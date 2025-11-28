import argparse
import pandas as pd
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main(model_path, data_path, output_path):
    # Charger le dictionnaire contenant le modèle et le vectorizer
    model_dict = joblib.load(model_path)
    model = model_dict['model']
    vectorizer = model_dict['vectorizer']

    # Charger le dataset
    df = pd.read_csv(data_path)
    X = df['text']
    y = df['label']

    # Transformer les données avec le vectorizer
    X_vect = vectorizer.transform(X)

    # Faire les prédictions
    y_pred = model.predict(X_vect)

    # Calculer les métriques
    metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred),
        'recall': recall_score(y, y_pred),
        'f1_score': f1_score(y, y_pred)
    }

    # Sauvegarder les métriques dans un fichier JSON
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    print(f"✅ Evaluation terminée. Métriques sauvegardées dans {output_path}")
    print(metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument('--model', type=str, required=True, help="Path to the trained model (joblib file)")
    parser.add_argument('--data', type=str, required=True, help="Path to the processed dataset CSV")
    parser.add_argument('--out', type=str, required=True, help="Path to save the metrics JSON")
    args = parser.parse_args()

    main(args.model, args.data, args.out)
