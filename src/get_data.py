"""Download and verify the public OpenML Titanic dataset."""

import hashlib
from pathlib import Path
from urllib.request import urlopen

DATA_PATH = Path("data/raw/titanic.arff")
DATA_URL = "https://www.openml.org/data/v1/download/16826755/Titanic.arff"
DATA_SHA256 = "c56e3785c96cd8aceac85f11bff21c2bf72841426f50903a6f27be8f334a7d03"


def main():
    with urlopen(DATA_URL, timeout=30) as response:
        data = response.read()
    if hashlib.sha256(data).hexdigest() != DATA_SHA256:
        raise ValueError("Downloaded file does not match OpenML Titanic 40945.")

    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_PATH.write_bytes(data)
    print(f"Downloaded and verified {DATA_PATH} ({len(data):,} bytes)")
    print(f"SHA-256: {DATA_SHA256}")


if __name__ == "__main__":
    main()
