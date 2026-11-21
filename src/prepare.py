"""Prepare a tabular dataset from the OpenML Titanic ARFF."""

from pathlib import Path

import pandas as pd

from src.get_data import DATA_PATH

PREPARED_PATH = Path("data/prepared/titanic.csv")
NUMERIC = ["age", "fare", "sibsp", "parch"]
CATEGORICAL = ["pclass", "sex"]
TARGET = "survived"


def read_titanic():
    with DATA_PATH.open(encoding="utf-8") as data_file:
        columns = []
        for line in data_file:
            if line.startswith("@attribute "):
                columns.append(line.split("'", 2)[1])
            elif line.strip().lower() == "@data":
                break
        return pd.read_csv(data_file, names=columns, na_values="?")


def main():
    data = read_titanic()[NUMERIC + CATEGORICAL + [TARGET]]
    PREPARED_PATH.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(PREPARED_PATH, index=False)
    print(f"Prepared {len(data):,} passengers in {PREPARED_PATH}")


if __name__ == "__main__":
    main()
