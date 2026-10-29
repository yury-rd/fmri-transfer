"""Run the response prediction examples and read saved results."""
from pathlib import Path

from fmri_response.cli import main


if __name__ == "__main__":
    main(Path(__file__).resolve().parent)
