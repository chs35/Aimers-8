from __future__ import annotations

import json
import logging

from dotenv import load_dotenv

from src.runtime.logging_capture import setup_cli_logging

load_dotenv()

from .app.runner import run
from .cli.args import parse_args


def main() -> int:
    args = parse_args()
    log_path = setup_cli_logging(args.model_path)
    logger = logging.getLogger(__name__)
    logger.debug("Benchmark started. model_dir_name=%s", args.model_path)

    try:
        payload = run(args)
    except Exception:  # noqa: BLE001
        logger.exception("Benchmark failed")
        raise

    payload["cli_log_file"] = str(log_path)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
