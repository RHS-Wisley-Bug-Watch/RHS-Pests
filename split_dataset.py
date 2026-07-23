from pathlib import Path
import runpy


SCRIPT_PATH = Path(__file__).resolve().parent / "insect_detection" / "split_dataset_script.py"


def main() -> None:
    runpy.run_path(str(SCRIPT_PATH), run_name="__main__")


if __name__ == "__main__":
    main()