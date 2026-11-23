"""Compare two days of NYC yellow taxi trips and simulate two data issues."""

from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

OUTPUT = Path("artifacts/evidently")
SOURCE = OUTPUT / "yellow_tripdata_2023-11.parquet"


def trips_on(date):
    start = pd.Timestamp(date)
    trips = pd.read_parquet(
        SOURCE,
        columns=[
            "fare_amount",
            "payment_type",
            "trip_distance",
            "passenger_count",
            "PULocationID",
        ],
        filters=[
            ("tpep_pickup_datetime", ">=", start),
            ("tpep_pickup_datetime", "<", start + pd.Timedelta(days=1)),
            ("fare_amount", ">", 0),
            ("fare_amount", "<=", 200),
            ("payment_type", "in", [1, 2]),
        ],
    )
    trips["payment_type"] = trips["payment_type"].map({1: "card", 2: "cash"})
    trips["PULocationID"] = trips["PULocationID"].astype(str)
    print(date, len(trips), "trips; median fare:", trips["fare_amount"].median())
    print(trips["payment_type"].value_counts(normalize=True).round(4))
    return trips


def main():
    reference = trips_on("2023-11-08")
    current = trips_on("2023-11-15")

    surcharge = np.random.default_rng(42).normal(loc=6, scale=2, size=len(current))

    for name, batch in [
        ("control", current),
        ("shifted", current.assign(fare_amount=current["fare_amount"] + surcharge)),
        ("payment", current.assign(payment_type="cash")),
    ]:
        report = Report(
            [
                DataDriftPreset(
                    num_method="wasserstein",
                    num_threshold=0.1,
                    cat_method="jensenshannon",
                    cat_threshold=0.1,
                )
            ]
        )
        result = report.run(current_data=batch, reference_data=reference)
        result.save_html(str(OUTPUT / f"{name}.html"))
        print(name, result.json())


if __name__ == "__main__":
    main()
