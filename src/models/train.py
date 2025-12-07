# src/models/train.py - Version finale corrigée
import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC  # CHANGÉ: LinearSVC → SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import argparse
import os
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV

def load_config(config_path="config/config.yaml"):
    """Charge la configuration depuis le fichier YAML"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_model(model_type, model_params):
    """Factory pour créer le modèle approprié"""
    if model_type == "logistic":
        return LogisticRegression(**model_params)
    
    elif model_type == "svm":
        # SVC avec probability=True pour avoir predict_proba
        params = model_params.copy()
        if 'probability' not in params:
            params['probability'] = True
        if 'max_iter' in params:
            del params['max_iter']
        return SVC(**params)
    
    elif model_type == "nb":
        return MultinomialNB(**model_params)
    
    elif model_type == "rf":
        # Random Forest optimisé pour le spam
        params = model_params.copy()
        if 'class_weight' not in params:
            params['class_weight'] = 'balanced'
        if 'n_jobs' not in params:
            params['n_jobs'] = -1
        # Limiter la profondeur pour éviter l'overfitting
        if 'max_depth' not in params or params['max_depth'] is None:
            params['max_depth'] = 20
        return RandomForestClassifier(**params)
    
    else:
        raise ValueError(f"Modèle non supporté: {model_type}")

def setup_mlflow(config, model_type):
    """Configure MLflow avec tags et paramètres"""
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    run_name = f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return mlflow.start_run(run_name=run_name)

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calcule toutes les métriques"""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0)
    }
    
    if y_pred_proba is not None:
        try:
            metrics["roc_auc"] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics["roc_auc"] = 0.0
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train spam detection model")
    parser.add_argument("--data", type=str, required=True, help="Path to processed data")
    parser.add_argument("--out", type=str, required=True, help="Path to save model")
    parser.add_argument("--model_type", type=str, default="logistic", 
                       choices=["logistic", "svm", "nb", "rf"], 
                       help="Type of model to train")
    parser.add_argument("--config", type=str, default="config/config.yaml", 
                       help="Path to config file")
    parser.add_argument("--test_size", type=float, default=0.2, 
                       help="Test set size")
    
    args = parser.parse_args()
    
    # Charger configuration
    config = load_config(args.config)
    
    # Créer répertoire de sortie
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    
    # Charger données
    print(f"📊 Chargement des données depuis {args.data}")
    data = pd.read_csv(args.data)
    X = data['text'].astype(str)
    y = data['label']
    
    # Statistiques
    print(f"📈 Statistiques des données:")
    print(f"   Total: {len(X)}")
    print(f"   Ham (0): {sum(y == 0)} ({sum(y == 0)/len(y):.1%})")
    print(f"   Spam (1): {sum(y == 1)} ({sum(y == 1)/len(y):.1%})")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=args.test_size, 
        random_state=config['data']['random_state'],
        stratify=y
    )
    
    # Vectorisation
    print(f"\n🔧 Vectorisation TF-IDF...")
    vectorizer_params = config['preprocessing']['vectorizer']
    vectorizer = TfidfVectorizer(
        max_features=vectorizer_params['max_features'],
        ngram_range=tuple(vectorizer_params['ngram_range']),
        stop_words=vectorizer_params['stop_words'],
        min_df=vectorizer_params['min_df'],
        max_df=vectorizer_params['max_df']
    )
    
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)
    
    # Création du modèle
    print(f"🤖 Création du modèle {args.model_type.upper()}...")
    model_params = config['models']['parameters'][args.model_type]
    model = get_model(args.model_type, model_params)
    
    # MLflow tracking
    with setup_mlflow(config, args.model_type) as run:
        # Log des paramètres
        mlflow.log_param("model_type", args.model_type)
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_params(model_params)
        
        # Tags
        mlflow.set_tag("stage", "training")
        mlflow.set_tag("task", "spam_detection")
        
        # Entraînement
        print("🏋️  Entraînement du modèle...")
        model.fit(X_train_vect, y_train)
        
        # Prédictions
        y_pred_train = model.predict(X_train_vect)
        y_pred_test = model.predict(X_test_vect)
        
        # Probabilités si disponibles
        if hasattr(model, "predict_proba"):
            y_pred_proba_train = model.predict_proba(X_train_vect)[:, 1]
            y_pred_proba_test = model.predict_proba(X_test_vect)[:, 1]
        else:
            y_pred_proba_train = None
            y_pred_proba_test = None
        
        # Métriques
        train_metrics = calculate_metrics(y_train, y_pred_train, y_pred_proba_train)
        test_metrics = calculate_metrics(y_test, y_pred_test, y_pred_proba_test)
        
        # Log des métriques
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        for metric_name, metric_value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", metric_value)
        
        # Log du modèle
        mlflow.sklearn.log_model(model, "model")
        mlflow.sklearn.log_model(vectorizer, "vectorizer")
        
        # Affichage
        print("\n" + "="*60)
        print("🎯 RÉSULTATS DE L'ENTRAÎNEMENT")
        print("="*60)
        print(f"Modèle: {args.model_type.upper()}")
        print(f"Run ID: {run.info.run_id}")
        print("\n📊 Métriques TEST:")
        for metric, value in test_metrics.items():
            print(f"   {metric:>12}: {value:.4f}")
        
        print("\n📈 Métriques TRAIN:")
        for metric, value in train_metrics.items():
            print(f"   {metric:>12}: {value:.4f}")
        print("="*60)
    
    # Sauvegarde locale
    print(f"\n💾 Sauvegarde du modèle dans {args.out}")
    model_dict = {
        'model': model,
        'vectorizer': vectorizer,
        'model_type': args.model_type,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'train': train_metrics,
            'test': test_metrics
        },
        'data_stats': {
            'n_train': len(X_train),
            'n_test': len(X_test),
            'class_distribution': dict(y.value_counts())
        }
    }
    
    joblib.dump(model_dict, args.out)
    
    print("✅ Entraînement terminé avec succès!")
    print(f"📁 Modèle sauvegardé: {args.out}")
    print(f"🔍 Pour visualiser: mlflow ui --port 5000")

if __name__ == "__main__":
    main()