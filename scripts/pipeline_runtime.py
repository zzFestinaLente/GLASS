"""Shared task definitions, logging, and progress for GLASS pipeline scripts."""

import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

from tqdm import tqdm


TASKS = {
    "lamp4": "LaMP_4",
    "lamp5": "LaMP_5",
    "longlamp3": "product_review_temporal",
    "longlamp4": "topic_writing_temporal",
}


def dump_new(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def elapsed_text(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:02d}:{seconds:02d}"


class StageProgress:
    """Render structured worker events and stage completion status."""

    def __init__(self, label, log_path, stream=None, clock=None, interval=30):
        self.label, self.log_path = label, log_path
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock if clock is not None else time.monotonic
        self.started = self.last_line = self.last_refresh = self.clock()
        self.interval = interval
        self.done, self.total, self.unit, self.status = 0, None, "items", "loading"
        self.observed = False
        self.workers_done = self.workers_total = self.workers_ready = None
        self.terminal = bool(getattr(self.stream, "isatty", lambda: False)())
        self.bar = None
        self.write(f"{self.label} starting; log: {self.log_path}")
        if self.terminal:
            self.bar = tqdm(total=None, desc=self.label, unit=self.unit,
                            file=self.stream, dynamic_ncols=True, leave=False,
                            bar_format="{desc} [{elapsed}] {postfix}")
            self.bar.set_postfix_str(self.status, refresh=False)

    def write(self, message):
        print(message, file=self.stream, flush=True)

    @staticmethod
    def count(value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if value < 0 or not float(value).is_integer():
            return None
        return int(value)

    def update(self, event):
        done = self.count(event.get("done"))
        total = self.count(event.get("total"))
        if done is None:
            return
        self.observed = True
        self.done, self.total = done, total
        self.unit = str(event.get("unit") or self.unit)
        self.status = str(event.get("status") or "running")
        self.workers_done = self.count(event.get("workers_done"))
        self.workers_total = self.count(event.get("workers_total"))
        self.workers_ready = self.count(event.get("workers_ready"))
        self.tick()

    def detail(self):
        detail = self.status
        if self.workers_total is not None:
            detail += f", workers {self.workers_done or 0}/{self.workers_total} finished"
            if self.workers_ready is not None and self.status == "loading":
                detail += f", {self.workers_ready}/{self.workers_total} ready"
        return detail

    def can_estimate(self):
        return (self.status == "running" and self.done > 0
                and self.total is not None and self.total > self.done)

    def summary(self):
        elapsed = self.clock() - self.started
        total = "?" if self.total is None else str(self.total)
        counts = f" {self.done}/{total} {self.unit};" if self.observed else ""
        message = f"{self.label}{counts} {self.detail()}; elapsed {elapsed_text(elapsed)}"
        if self.can_estimate() and elapsed > 0:
            message += f"; ETA ~{elapsed_text(elapsed * (self.total - self.done) / self.done)}"
        return message

    def tick(self):
        now = self.clock()
        if self.bar is not None:
            if now - self.last_refresh < 0.2:
                return
            self.bar.total, self.bar.unit = self.total, self.unit
            self.bar.bar_format = (
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} {unit} "
                "[{elapsed}<{remaining}, {rate_fmt}] {postfix}"
                if self.can_estimate() else
                "{desc}: {n_fmt}/{total_fmt} {unit} [{elapsed}] {postfix}"
            )
            self.bar.set_postfix_str(self.detail(), refresh=False)
            self.bar.update(self.done - self.bar.n)
            self.bar.refresh()
            self.last_refresh = now
        elif now - self.last_line >= self.interval:
            self.write(self.summary())
            self.last_line = now

    def finish(self, returncode=None, interrupted=False):
        # The stage exits after writing and merging its outputs.
        self.status = "interrupted" if interrupted else "complete" if returncode == 0 else "failed"
        if self.bar is not None:
            self.bar.close()
        suffix = "" if returncode in (None, 0) else f"; exit code {returncode}"
        self.write(f"{self.summary()}{suffix}; log: {self.log_path}")


def stop_stage(process, interrupt=False):
    """Give parallel's parent a chance to clean workers, then stop its session."""
    if process.poll() is None:
        process.send_signal(signal.SIGINT if interrupt else signal.SIGTERM)
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    # The parent may have exited before cleaning every worker. Its separate
    # session lets us kill remaining descendants without touching the runner.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def run_stage(command, log_path, *, cwd=None, env=None, label, stage, stream=None):
    """Keep the complete stage log while rendering only structured progress."""
    process = reader = None
    lines = queue.Queue()

    def read_lines():
        try:
            for line in process.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    with Path(log_path).open("xb") as log:
        display = StageProgress(label, log_path, stream=stream)
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            reader = threading.Thread(target=read_lines, daemon=True)
            reader.start()
            while True:
                try:
                    line = lines.get(timeout=1)
                except queue.Empty:
                    display.tick()
                    continue
                if line is None:
                    break
                log.write(line)
                log.flush()
                if line.startswith(b"GLASS_PROGRESS "):
                    try:
                        event = json.loads(line[len(b"GLASS_PROGRESS "):])
                    except (ValueError, UnicodeError):
                        event = None
                    if isinstance(event, dict) and event.get("stage") == stage:
                        display.update(event)
            returncode = process.wait()
        except BaseException as error:
            if process is not None:
                stop_stage(process, interrupt=isinstance(error, KeyboardInterrupt))
                if reader is not None:
                    reader.join(timeout=2)
                while True:
                    try:
                        line = lines.get_nowait()
                    except queue.Empty:
                        break
                    if line is not None:
                        log.write(line)
                log.flush()
            display.finish(process.returncode if process is not None else None,
                           interrupted=isinstance(error, KeyboardInterrupt))
            raise
        finally:
            if process is not None and process.stdout is not None:
                process.stdout.close()
    display.finish(returncode)
    if returncode:
        raise subprocess.CalledProcessError(returncode, command)
