"""Log files: everything a training or evaluation run prints is also appended to a file in its folder."""

import contextlib
import sys
import time
import traceback
from pathlib import Path


class _Tee:
    """Writes to a stream and a log file."""

    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, text: str) -> int:
        self.log.write(text)
        self.log.flush()
        return self.stream.write(text)

    def flush(self) -> None:
        self.stream.flush()
        self.log.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


@contextlib.contextmanager
def logged(path: Path):
    """Copy everything printed (stdout and stderr) to ``path``, appending, with a header per run. An error's
    traceback goes into the log too."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as log:
        log.write(f"\n== {time.strftime('%Y-%m-%dT%H:%M:%S%z')}: {' '.join(sys.argv)}\n")
        out, err = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = _Tee(out, log), _Tee(err, log)
        try:
            yield
        except BaseException:
            log.write(traceback.format_exc())
            raise
        finally:
            sys.stdout, sys.stderr = out, err
