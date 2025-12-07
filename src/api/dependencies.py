# src/api/dependencies.py
import joblib
import os
import time
from typing import Optional, Tuple, Dict, Any, List
from src.data.preprocess import clean_text
import numpy as np
from datetime import datetime

# Cache pour les modèles
_loaded_models: Dict[str, Dict[str, Any]] = {}
_default_model_type: str = "svm"
_startup_time: float = time.time()

def initialize_models() -> Dict[str, bool]:
    """
    Initialise tous les modèles disponibles au démarrage
    Retourne un dict avec le statut de chargement de chaque modèle
    """
    models_dir = "models"
    models_to_load = ["logistic", "svm", "nb", "rf"]
    results = {}
    
    for model_type in models_to_load:
        model_path = os.path.join(models_dir, f"model_{model_type}.pkl")
        if os.path.exists(model_path):
            try:
                model_dict = joblib.load(model_path)
                _loaded_models[model_type] = model_dict
                results[model_type] = True
                print(f"[MODEL] Modèle {model_type} chargé avec succès")
            except Exception as e:
                print(f"[ERROR] Erreur chargement modèle {model_type}: {e}")
                results[model_type] = False
        else:
            print(f"[WARN] Fichier modèle non trouvé: {model_path}")
            results[model_type] = False
    
    return results

def load_model(model_type: str = "svm") -> bool:
    """Charge un modèle spécifique dans le cache"""
    try:
        model_path = f"models/model_{model_type}.pkl"
        if not os.path.exists(model_path):
            print(f"[ERROR] Fichier modèle non trouvé: {model_path}")
            return False
        
        model_dict = joblib.load(model_path)
        _loaded_models[model_type] = model_dict
        print(f"[MODEL] Modèle {model_type} chargé dynamiquement")
        return True
    except Exception as e:
        print(f"[ERROR] Erreur chargement modèle {model_type}: {e}")
        return False

def get_model(model_type: str = "svm") -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Retourne le modèle, vectorizer et metadata pour un type donné
    
    Args:
        model_type: Type de modèle (logistic, svm, nb, rf)
    
    Returns:
        Tuple: (model, vectorizer, metadata)
    """
    # Si le modèle n'est pas chargé, essayer de le charger
    if model_type not in _loaded_models:
        if not load_model(model_type):
            # Fallback au modèle par défaut
            if _default_model_type in _loaded_models:
                print(f"[WARN] Modèle {model_type} non disponible, utilisation de {_default_model_type}")
                model_type = _default_model_type
            else:
                # Essayer de charger le premier modèle disponible
                for mt in ["svm", "logistic", "nb", "rf"]:
                    if mt in _loaded_models:
                        model_type = mt
                        break
    
    if model_type in _loaded_models:
        model_dict = _loaded_models[model_type]
        return (
            model_dict.get('model'),
            model_dict.get('vectorizer'),
            {
                'model_type': model_type,
                'timestamp': model_dict.get('timestamp', 'unknown'),
                'metrics': model_dict.get('metrics', {}),
                'config': model_dict.get('config', {})
            }
        )
    
    # Aucun modèle disponible
    return None, None, {}

def predict_text(text: str, model_type: str = "svm") -> Dict[str, Any]:
    """
    Prédit si un texte est spam avec un modèle spécifique
    
    Args:
        text: Texte à classifier
        model_type: Type de modèle à utiliser
    
    Returns:
        Dictionnaire avec les résultats de prédiction
    """
    start_time = time.time()
    
    # Obtenir le modèle
    model, vectorizer, metadata = get_model(model_type)
    
    if model is None or vectorizer is None:
        raise ValueError(f"Modèle {model_type} non disponible")
    
    # Nettoyage du texte
    cleaned_text = clean_text(text)
    
    # Vectorisation
    features = vectorizer.transform([cleaned_text])
    
    # Prédiction
    prediction = model.predict(features)[0]
    
    # Probabilités si disponibles
    probabilities = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(features)[0]
        probabilities = {"ham": float(proba[0]), "spam": float(proba[1])}
        confidence = float(max(proba))
    else:
        # Pour les modèles sans predict_proba (SVM avec probability=False)
        confidence = 1.0 if prediction == model.predict(features)[0] else 0.0
    
    # Calcul du temps de traitement
    processing_time_ms = (time.time() - start_time) * 1000
    
    # Résultat
    return {
        "text": text,
        "cleaned_text": cleaned_text,
        "prediction": "spam" if prediction == 1 else "ham",
        "is_spam": bool(prediction == 1),
        "confidence": confidence,
        "model_used": model_type,
        "model_version": metadata.get('timestamp', 'v1.0'),
        "processing_time_ms": processing_time_ms,
        "probabilities": probabilities,
        "timestamp": datetime.now().isoformat()
    }

def get_available_models() -> List[str]:
    """Retourne la liste des modèles disponibles"""
    return list(_loaded_models.keys())

def get_model_info(model_type: str = "svm") -> Dict[str, Any]:
    """Retourne les informations détaillées d'un modèle"""
    if model_type not in _loaded_models:
        return {}
    
    model_dict = _loaded_models[model_type]
    model = model_dict.get('model')
    
    info = {
        "model_type": model_type,
        "model_version": model_dict.get('timestamp', 'unknown'),
        "training_date": model_dict.get('timestamp', 'unknown'),
        "model_class": model.__class__.__name__ if model else "Unknown",
        "metrics": model_dict.get('metrics', {}),
        "config": model_dict.get('config', {}),
        "parameters": model_dict.get('config', {}).get('models', {}).get('parameters', {}).get(model_type, {})
    }
    
    # Ajouter la taille du vocabulaire si disponible
    vectorizer = model_dict.get('vectorizer')
    if vectorizer and hasattr(vectorizer, 'vocabulary_'):
        info["vocabulary_size"] = len(vectorizer.vocabulary_)
        info["features_count"] = vectorizer.get_feature_names_out().shape[0]
    
    return info

def get_uptime() -> float:
    """Retourne le temps d'activité de l'API en secondes"""
    return time.time() - _startup_time
def get_health_info() -> Dict[str, Any]:
    """Retourne les informations pour le health check"""
    available_models = get_available_models()
    
    # Obtenir des infos sur le modèle par défaut
    default_model_info = {}
    if available_models:
        model_type = "svm" if "svm" in available_models else available_models[0]
        default_model_info = get_model_info(model_type)
    
    return {
        "model_loaded": len(available_models) > 0,
        "available_models": available_models,
        "default_model": "svm" if "svm" in available_models else (available_models[0] if available_models else None),
        "model_version": default_model_info.get("model_version", "unknown") if default_model_info else None,
        "uptime_seconds": get_uptime(),
        "models_loaded_count": len(available_models)
    }
# Alias pour la compatibilité
initialize_model = initialize_models  # Alias pour la compatibilité