from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from streamlit_app import run_app  # type: ignore  # pylint: disable=wrong-import-position


def main() -> None:
    """Boot the Streamlit application from the canonical entry point.

    Args:
        None

    Returns:
        None
    """

    run_app()


if __name__ == "__main__":
    main()