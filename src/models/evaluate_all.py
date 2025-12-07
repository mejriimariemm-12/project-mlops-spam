 
# src/models/evaluate_all.py
import pandas as pd
import joblib
import json
import argparse
import mlflow
import yaml
import os
import glob
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def load_config(config_path="config/config.yaml"):
    """Charge la configuration YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_single_model(model_path, X, y, model_name):
    """Évalue un seul modèle"""
    print(f"  Évaluation de {model_name}...")
    
    try:
        # Charger le modèle
        model_dict = joblib.load(model_path)
        model = model_dict.get('model')
        vectorizer = model_dict.get('vectorizer')
        
        if model is None or vectorizer is None:
            print(f"  ⚠️  Modèle ou vectorizer manquant dans {model_path}")
            return None
        
        # Vectoriser les données
        X_vect = vectorizer.transform(X)
        
        # Prédictions
        y_pred = model.predict(X_vect)
        
        # Calcul des métriques de base
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1_score': f1_score(y, y_pred)
        }
        
        # AUC-ROC si le modèle supporte predict_proba
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_vect)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y, y_pred_proba)
        
        return {
            'metrics': metrics,
            'model_type': model_dict.get('model_type', 'unknown'),
            'timestamp': model_dict.get('timestamp', 'unknown'),
            'config': model_dict.get('config', {}),
            'vectorizer_shape': X_vect.shape
        }
        
    except Exception as e:
        print(f"  ❌ Erreur avec {model_name}: {str(e)}")
        return None

def create_comparison_plot(results, output_dir):
    """Crée un graphique de comparaison des modèles"""
    if not results:
        return
    
    # Préparer les données pour le graphique
    model_names = []
    f1_scores = []
    accuracies = []
    
    for model_name, model_info in results.items():
        if model_info and 'metrics' in model_info:
            model_names.append(model_name)
            f1_scores.append(model_info['metrics']['f1_score'])
            accuracies.append(model_info['metrics']['accuracy'])
    
    if not model_names:
        return
    
    # Graphique comparatif
    plt.figure(figsize=(12, 6))
    
    x = range(len(model_names))
    width = 0.35
    
    plt.subplot(1, 2, 1)
    plt.bar(x, f1_scores, width, label='F1-Score', color='skyblue')
    plt.xlabel('Modèles')
    plt.ylabel('F1-Score')
    plt.title('Comparaison des F1-Scores')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.subplot(1, 2, 2)
    plt.bar(x, accuracies, width, label='Accuracy', color='lightgreen')
    plt.xlabel('Modèles')
    plt.ylabel('Accuracy')
    plt.title('Comparaison des Accuracies')
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.ylim(0, 1)
    
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return plot_path

def main():
    parser = argparse.ArgumentParser(description="Compare multiple spam detection models")
    parser.add_argument("--models_dir", type=str, required=True, 
                       help="Directory containing model files (.pkl)")
    parser.add_argument("--data", type=str, required=True,
                       help="Path to evaluation data CSV")
    parser.add_argument("--out", type=str, required=True,
                       help="Output JSON file for comparison results")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to config file")
    
    args = parser.parse_args()
    
    # Charger configuration
    config = load_config(args.config)
    
    # Charger les données
    print(f"📊 Chargement des données depuis {args.data}")
    data = pd.read_csv(args.data)
    X = data['text'].astype(str)
    y = data['label']
    
    print(f"📈 Données chargées: {len(X)} échantillons")
    print(f"📊 Distribution des labels: {y.value_counts().to_dict()}")
    
    # Trouver tous les fichiers de modèle
    model_patterns = ["*.pkl", "model_*.pkl"]
    model_files = []
    
    for pattern in model_patterns:
        model_files.extend(glob.glob(os.path.join(args.models_dir, pattern)))
    
    model_files = [f for f in model_files if 'vectorizer' not in f.lower()]
    
    if not model_files:
        print(f"❌ Aucun fichier de modèle trouvé dans {args.models_dir}")
        return
    
    print(f"🔍 {len(model_files)} modèles trouvés:")
    for mf in model_files:
        print(f"   - {os.path.basename(mf)}")
    
    # Évaluer chaque modèle
    results = {}
    
    for model_file in model_files:
        model_name = os.path.splitext(os.path.basename(model_file))[0]
        model_name = model_name.replace('model_', '')
        
        print(f"\n{'='*50}")
        print(f"🧪 ÉVALUATION: {model_name}")
        print('='*50)
        
        result = evaluate_single_model(model_file, X, y, model_name)
        results[model_name] = result
    
    # Identifier les modèles valides
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if not valid_results:
        print("❌ Aucun modèle n'a pu être évalué avec succès")
        return
    
    # Trouver le meilleur modèle (par F1-Score)
    best_model_name = max(valid_results.items(), 
                         key=lambda x: x[1]['metrics']['f1_score'])[0]
    best_model_info = valid_results[best_model_name]
    
    # Créer le rapport de comparaison
    comparison_report = {
        'comparison_date': datetime.now().isoformat(),
        'data_info': {
            'path': args.data,
            'n_samples': len(X),
            'class_distribution': y.value_counts().to_dict(),
            'positive_ratio': float(y.mean())
        },
        'models_evaluated': len(model_files),
        'models_successful': len(valid_results),
        'best_model': {
            'name': best_model_name,
            'file': next(f for f in model_files if best_model_name in f),
            'metrics': best_model_info['metrics'],
            'model_type': best_model_info['model_type']
        },
        'all_results': {}
    }
    
    # Ajouter les résultats détaillés
    for model_name, model_info in valid_results.items():
        comparison_report['all_results'][model_name] = {
            'metrics': model_info['metrics'],
            'model_type': model_info['model_type'],
            'timestamp': model_info['timestamp']
        }
    
    # Classement par F1-Score
    ranking = sorted(
        [(name, info['metrics']['f1_score']) 
         for name, info in valid_results.items()],
        key=lambda x: x[1],
        reverse=True
    )
    
    comparison_report['ranking_by_f1'] = [
        {'model': name, 'f1_score': score} for name, score in ranking
    ]
    
    # Sauvegarder le rapport
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(comparison_report, f, indent=2, ensure_ascii=False)
    
    # Créer des visualisations
    plots_dir = os.path.join(os.path.dirname(args.out), 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_path = create_comparison_plot(valid_results, plots_dir)
    
    # MLflow tracking
    print(f"\n📊 Configuration MLflow...")
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    with mlflow.start_run(run_name=f"model_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        # Log des paramètres
        mlflow.log_param("models_dir", args.models_dir)
        mlflow.log_param("data_path", args.data)
        mlflow.log_param("n_models", len(model_files))
        mlflow.log_param("best_model", best_model_name)
        
        # Log des métriques du meilleur modèle
        for metric_name, metric_value in best_model_info['metrics'].items():
            mlflow.log_metric(f"best_{metric_name}", metric_value)
        
        # Log du classement
        for i, (model_name, f1_score) in enumerate(ranking, 1):
            mlflow.log_metric(f"rank_{i}_f1", f1_score)
            mlflow.log_param(f"rank_{i}_model", model_name)
        
        # Tags
        mlflow.set_tag("stage", "model_comparison")
        mlflow.set_tag("run_type", "evaluation")
        
        # Log des artifacts
        mlflow.log_artifact(args.out)
        if plot_path and os.path.exists(plot_path):
            mlflow.log_artifact(plot_path)
        
        # Log du rapport de comparaison
        mlflow.log_dict(comparison_report, "comparison_report.json")
    
    # Affichage des résultats
    print(f"\n{'='*60}")
    print("🏆 RÉSULTATS DE LA COMPARAISON")
    print('='*60)
    print(f"📊 Modèles évalués: {len(model_files)}")
    print(f"✅ Modèles réussis: {len(valid_results)}")
    print(f"🏅 Meilleur modèle: {best_model_name}")
    print(f"   F1-Score: {best_model_info['metrics']['f1_score']:.4f}")
    print(f"   Accuracy: {best_model_info['metrics']['accuracy']:.4f}")
    
    print(f"\n📈 CLASSEMENT (par F1-Score):")
    for i, (model_name, f1_score) in enumerate(ranking, 1):
        print(f"   {i:2d}. {model_name:<20} - F1: {f1_score:.4f}")
    
    print(f"\n💾 Rapport sauvegardé: {args.out}")
    if plot_path:
        print(f"📊 Graphique sauvegardé: {plot_path}")
    
    print(f"\n🔍 Pour voir les résultats dans MLflow:")
    print(f"   mlflow ui --port 5000")
    print('='*60)
    
    print("\n✅ Comparaison terminée avec succès!")

if __name__ == "__main__":
    main()