# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = ["evidently==0.7.23", "pandas==2.3.3"]
# ///
"""Run from the repository root after preparing the Titanic CSV."""

from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset


def main():
    data = pd.read_csv("data/prepared/titanic.csv", usecols=["fare", "age"])
    sample = data.dropna().sample(n=400, random_state=42)
    reference = sample.iloc[:200]
    current = sample.iloc[200:]
    output = Path("artifacts/evidently")
    output.mkdir(parents=True, exist_ok=True)

    for name, batch in [
        ("control", current),
        ("shifted", current.assign(fare=current["fare"] + 30)),
    ]:
        report = Report([DataDriftPreset(num_method="ks", num_threshold=0.05)])
        result = report.run(current_data=batch, reference_data=reference)
        result.save_html(str(output / f"{name}.html"))
        print(name, result.json())


if __name__ == "__main__":
    main()
