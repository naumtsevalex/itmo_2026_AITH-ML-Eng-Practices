"""Run from the repository root after downloading the Titanic ARFF."""

from pathlib import Path

from evidently import Report
from evidently.presets import DataDriftPreset

from src.prepare import read_titanic


def main():
    reference = read_titanic()[["fare", "age"]]
    current = reference.copy()
    output = Path("artifacts/evidently")
    output.mkdir(parents=True, exist_ok=True)

    for name, batch in [
        ("control", current),
        ("shifted", current.assign(fare=current["fare"] + 30)),
    ]:
        report = Report([DataDriftPreset(num_method="wasserstein", num_threshold=0.1)])
        result = report.run(current_data=batch, reference_data=reference)
        result.save_html(str(output / f"{name}.html"))
        print(name, result.json())


if __name__ == "__main__":
    main()
