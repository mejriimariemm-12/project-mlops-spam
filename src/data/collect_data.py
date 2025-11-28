import os
import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

URL = "https://raw.githubusercontent.com/justmarkham/DAT8/master/data/sms.tsv"

def collect_data(output_path: str):
    logging.info("Starting data collection...")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        df = pd.read_csv(URL, sep="\t", names=["label", "text"])
        logging.info(f"Downloaded dataset, shape = {df.shape}")
    except Exception as e:
        logging.error("Error while downloading dataset")
        raise e

    if "label" not in df.columns or "text" not in df.columns:
        raise ValueError("Dataset missing required columns: label, text")

    df.to_csv(output_path, index=False)
    logging.info(f"Dataset saved to {output_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collect raw SMS spam dataset")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path")

    args = parser.parse_args()
    collect_data(args.out)
