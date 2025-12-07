# src/models/evaluate.py - Version améliorée
import pandas as pd
import joblib
import json
import argparse
import mlflow
import yaml
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def load_config(config_path="config/config.yaml"):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model_path, data_path, output_path, config):
    """Évalue un modèle sur un jeu de données"""
    print(f"Chargement du modèle depuis {model_path}")
    model_dict = joblib.load(model_path)
    model = model_dict['model']
    vectorizer = model_dict['vectorizer']
    
    print(f"Chargement des données depuis {data_path}")
    data = pd.read_csv(data_path)
    X = data['text'].astype(str)
    y = data['label']
    
    print(f"Données: {len(X)} échantillons")
    print(f"Distribution des labels:\n{y.value_counts()}")
    
    # Vectorisation
    X_vect = vectorizer.transform(X)
    
    # Prédictions
    y_pred = model.predict(X_vect)
    y_pred_proba = model.predict_proba(X_vect)[:, 1] if hasattr(model, "predict_proba") else None
    
    # Calcul des métriques
    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred),
        "recall": recall_score(y, y_pred),
        "f1_score": f1_score(y, y_pred)
    }
    
    # AUC-ROC si disponible
    if y_pred_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y, y_pred_proba)
    
    # Rapport de classification détaillé
    report = classification_report(y, y_pred, output_dict=True)
    
    # Matrice de confusion
    cm = confusion_matrix(y, y_pred)
    
    # Sauvegarder les résultats
    results = {
        "model_path": model_path,
        "data_path": data_path,
        "evaluation_date": datetime.now().isoformat(),
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "data_stats": {
            "n_samples": len(X),
            "class_distribution": y.value_counts().to_dict(),
            "positive_ratio": y.mean()
        }
    }
    
    # Créer le dossier de sortie
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Sauvegarder en JSON
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Créer des visualisations
    if config.get('evaluation', {}).get('save_plots', True):
        create_evaluation_plots(y, y_pred, y_pred_proba, output_path)
    
    return results

def create_evaluation_plots(y_true, y_pred, y_pred_proba, output_path):
    """Crée des visualisations pour l'évaluation"""
    plots_dir = os.path.join(os.path.dirname(output_path), "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Matrice de confusion
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Ham', 'Spam'], 
                yticklabels=['Ham', 'Spam'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 2. Courbe ROC si disponible
    if y_pred_proba is not None:
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'roc_curve.png'))
        plt.close()
    
    # 3. Distribution des prédictions
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    y_true.value_counts().plot(kind='bar', color=['blue', 'red'])
    plt.title('Distribution des Vraies Labels')
    plt.xlabel('Label')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    plt.subplot(1, 2, 2)
    pd.Series(y_pred).value_counts().plot(kind='bar', color=['lightblue', 'pink'])
    plt.title('Distribution des Prédictions')
    plt.xlabel('Label Prédit')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'distributions.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Evaluate spam detection model")
    parser.add_argument("--model", type=str, required=True, help="Path to model file")
    parser.add_argument("--data", type=str, required=True, help="Path to data file")
    parser.add_argument("--out", type=str, required=True, help="Path to save metrics")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                       help="Path to config file")
    
    args = parser.parse_args()
    
    # Charger configuration
    config = load_config(args.config)
    
    # Configurer MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Évaluer le modèle
        results = evaluate_model(args.model, args.data, args.out, config)
        
        # Log dans MLflow
        mlflow.log_param("model_path", args.model)
        mlflow.log_param("data_path", args.data)
        
        for metric_name, metric_value in results['metrics'].items():
            mlflow.log_metric(metric_name, metric_value)
        
        mlflow.set_tag("stage", "evaluation")
        mlflow.set_tag("data_size", results['data_stats']['n_samples'])
        
        # Log des artifacts
        mlflow.log_artifact(args.out)
        
        # Log des plots si existent
        plots_dir = os.path.join(os.path.dirname(args.out), "plots")
        if os.path.exists(plots_dir):
            mlflow.log_artifacts(plots_dir, "evaluation_plots")
        
        # Affichage des résultats
        print("\n" + "="*50)
        print("RÉSULTATS DE L'ÉVALUATION")
        print("="*50)
        for metric_name, metric_value in results['metrics'].items():
            print(f"{metric_name:<15}: {metric_value:.4f}")
        
        print(f"\nRapport sauvegardé dans: {args.out}")
        print(f"Run MLflow ID: {mlflow.active_run().info.run_id}")
        print("="*50)
    
    print("✅ Évaluation terminée avec succès!")

if __name__ == "__main__":
    main()