from __future__ import annotations

import atexit
import logging
import os
import re
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import BinaryIO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_ROOT = PROJECT_ROOT / "logs"

_LOG_FILE_HANDLE: BinaryIO | None = None
_STDOUT_MIRROR: "_FdMirror | None" = None
_STDERR_MIRROR: "_FdMirror | None" = None


class _FdMirror:
    def __init__(self, fd: int, log_file: BinaryIO):
        self._fd = fd
        self._log_file = log_file
        self._orig_fd = os.dup(fd)
        self._pipe_r, pipe_w = os.pipe()
        os.dup2(pipe_w, fd)
        os.close(pipe_w)
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        while True:
            try:
                chunk = os.read(self._pipe_r, 8192)
            except OSError:
                break
            if not chunk:
                break
            try:
                os.write(self._orig_fd, chunk)
            except OSError:
                pass
            try:
                self._log_file.write(chunk)
                self._log_file.flush()
            except Exception:  # noqa: BLE001
                pass

    def close(self) -> None:
        try:
            os.dup2(self._orig_fd, self._fd)
        except OSError:
            pass
        try:
            os.close(self._orig_fd)
        except OSError:
            pass
        try:
            os.close(self._pipe_r)
        except OSError:
            pass
        self._thread.join(timeout=0.2)


def _safe_model_name(model_dir_name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z._-]+", "_", model_dir_name.strip())
    return cleaned or "model"


def _shutdown_logging_capture() -> None:
    global _STDOUT_MIRROR, _STDERR_MIRROR, _LOG_FILE_HANDLE

    if _STDOUT_MIRROR is not None:
        _STDOUT_MIRROR.close()
        _STDOUT_MIRROR = None
    if _STDERR_MIRROR is not None:
        _STDERR_MIRROR.close()
        _STDERR_MIRROR = None
    if _LOG_FILE_HANDLE is not None:
        try:
            _LOG_FILE_HANDLE.close()
        except Exception:  # noqa: BLE001
            pass
        _LOG_FILE_HANDLE = None


def setup_cli_logging(model_dir_name: str) -> Path:
    global _LOG_FILE_HANDLE, _STDOUT_MIRROR, _STDERR_MIRROR

    LOGS_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = LOGS_ROOT / f"{timestamp}_{_safe_model_name(model_dir_name)}.log"

    _LOG_FILE_HANDLE = log_path.open("ab", buffering=0)
    _STDOUT_MIRROR = _FdMirror(1, _LOG_FILE_HANDLE)
    _STDERR_MIRROR = _FdMirror(2, _LOG_FILE_HANDLE)
    atexit.register(_shutdown_logging_capture)

    # Core engine/library debug visibility.
    os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"
    os.environ["TRANSFORMERS_VERBOSITY"] = "debug"
    os.environ["HF_HUB_VERBOSITY"] = "debug"

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    for logger_name in ("vllm", "transformers", "datasets", "openai", "httpx", "urllib3"):
        logging.getLogger(logger_name).setLevel(logging.DEBUG)

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_debug()
    except Exception:  # noqa: BLE001
        pass

    try:
        from datasets.utils import logging as ds_logging

        ds_logging.set_verbosity_debug()
    except Exception:  # noqa: BLE001
        pass

    logging.getLogger(__name__).debug("CLI log file: %s", log_path)
    return log_path
