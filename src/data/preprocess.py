# src/data/preprocess.py - Version corrigée
import os
import pandas as pd
import logging
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def clean_text(text):
    """Nettoyage plus léger - garde certains caractères spéciaux"""
    if not isinstance(text, str):
        return ""
    
    # Convertir en minuscules
    text = text.lower()
    
    # Supprimer URLs
    text = re.sub(r"http\S+|www\S+", " ", text)
    
    # Garder les caractères importants pour le spam
    # Garde: lettres, chiffres, !, ?, $, €, £, %, @
    text = re.sub(r"[^a-z0-9\s!?$€£%@]", " ", text)
    
    # Remplacer les espaces multiples
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def preprocess(input_path, output_path):
    logging.info("Loading raw dataset...")
    df = pd.read_csv(input_path)
    
    logging.info(f"Initial shape: {df.shape}")
    
    # Afficher des exemples avant nettoyage
    print("\n📝 EXEMPLES AVANT NETTOYAGE:")
    for i, (label, text) in enumerate(df[['label', 'text']].head(3).values):
        print(f"  {i+1}. [{label}] {text[:60]}...")
    
    df = df.drop_duplicates()
    logging.info(f"After removing duplicates: {df.shape}")
    
    # Nettoyer le texte
    df["text"] = df["text"].astype(str).apply(clean_text)
    
    # Afficher des exemples après nettoyage
    print("\n📝 EXEMPLES APRÈS NETTOYAGE:")
    for i, (label, text) in enumerate(df[['label', 'text']].head(3).values):
        print(f"  {i+1}. [{label}] {text[:60]}...")
    
    # Supprimer textes vides
    df = df[df["text"].str.strip() != ""]
    logging.info(f"After removing empty text: {df.shape}")
    
    # Convertir labels (ham=0, spam=1)
    df["label"] = df["label"].map({"ham": 0, "spam": 1})
    logging.info("Converted labels: ham→0, spam→1")
    
    # Statistiques
    print(f"\n📊 DISTRIBUTION DES LABELS:")
    print(f"  Ham (0): {sum(df['label'] == 0)}")
    print(f"  Spam (1): {sum(df['label'] == 1)}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logging.info(f"Processed dataset saved at {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess SMS dataset")
    parser.add_argument("--in", dest="input_path", required=True)
    parser.add_argument("--out", dest="output_path", required=True)
    args = parser.parse_args()
    preprocess(args.input_path, args.output_path)