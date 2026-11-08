import zipfile
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "data" / "raw"
OUT_CSV = HERE / "data" / "cfpb_bank_complaints_2025.csv"

BANK_PRODUCTS = [
    "Checking or savings account",
    "Credit card",
    "Money transfer, virtual currency, or money service",
    "Mortgage",
    "Vehicle loan or lease",
]

COLUMNS = {
    "Complaint ID": "complaint_id",
    "Date received": "date_received",
    "Product": "product",
    "Sub-product": "sub_product",
    "Issue": "issue",
    "Sub-issue": "sub_issue",
    "Consumer complaint narrative": "narrative",
    "Company": "company",
    "State": "state",
    "Tags": "tags",
    "Company response to consumer": "company_response",
    "Timely response?": "timely_response",
}

SAMPLE_SIZE = 60_000
SEED = 42


def read_export(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        with z.open(z.namelist()[0]) as fh:
            for chunk in pd.read_csv(fh, dtype=str, chunksize=200_000, keep_default_na=False):
                has_story = chunk["Consumer complaint narrative"].str.strip() != ""
                in_scope = chunk["Product"].isin(BANK_PRODUCTS)
                yield chunk.loc[has_story & in_scope, list(COLUMNS)]


def main():
    exports = sorted(RAW_DIR.glob("CCDB_Export_*_2025.zip"))
    frames = []
    for path in exports:
        part = pd.concat(read_export(path), ignore_index=True)
        print(f"{path.name}: {len(part):,} bank-product complaints with a narrative")
        frames.append(part)

    df = pd.concat(frames, ignore_index=True).rename(columns=COLUMNS)
    df = df[df["date_received"].str.startswith("2025")]
    print(f"In scope for 2025: {len(df):,}")

    sample = df.sample(n=SAMPLE_SIZE, random_state=SEED).sort_values(["date_received", "complaint_id"])
    sample.to_csv(OUT_CSV, index=False)
    print(f"Saved {len(sample):,} rows to {OUT_CSV} ({OUT_CSV.stat().st_size / 1e6:.1f} MB)")
    print(sample["product"].value_counts().to_string())


if __name__ == "__main__":
    main()
