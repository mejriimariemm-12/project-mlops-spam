import os
import pandas as pd
import logging
import requests
import zipfile
import io

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def collect_data(output_path: str):
    logging.info("Starting data collection...")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # URL du vrai dataset SMS Spam
    URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    try:
        # Télécharger le zip
        response = requests.get(URL)
        response.raise_for_status()
        
        # Extraire le fichier
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
            # Chercher le fichier SMS
            df = None
            for file_name in zip_file.namelist():
                if file_name.endswith('.txt') or 'SMSSpamCollection' in file_name:
                    with zip_file.open(file_name) as f:
                        # Lire le fichier avec tab comme séparateur
                        df = pd.read_csv(f, sep='\t', header=None, names=["label", "text"])
                        logging.info(f"Found file: {file_name}")
                        break
            
            if df is None:
                raise ValueError("No SMS file found in the zip archive")
        
        logging.info(f"Downloaded SMS Spam dataset, shape = {df.shape}")
        
        # Sauvegarder
        df.to_csv(output_path, index=False)
        logging.info(f"Dataset saved to {output_path}")
        
    except Exception as e:
        logging.error(f"Error while downloading dataset: {e}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect raw SMS spam dataset")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()
    collect_data(args.out)