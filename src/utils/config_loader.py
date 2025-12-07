# src/utils/config_loader.py
import yaml
import os
from typing import Dict, Any, Optional
import json

class ConfigLoader:
    """Chargeur de configuration centralisé"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
    
    def load_config(self) -> Dict[str, Any]:
        """Charge la configuration depuis le fichier YAML"""
        if not os.path.exists(self.config_path):
            print(f"⚠️  Fichier de configuration non trouvé: {self.config_path}")
            return {}
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            print(f"✅ Configuration chargée depuis {self.config_path}")
            return config or {}
            
        except yaml.YAMLError as e:
            print(f"❌ Erreur YAML: {e}")
            return {}
        except Exception as e:
            print(f"❌ Erreur chargement configuration: {e}")
            return {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Récupère une valeur de configuration avec chemin pointé"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                if isinstance(value, dict):
                    value = value.get(k, {})
                else:
                    return default
            
            return value if value != {} else default
            
        except (KeyError, AttributeError):
            return default
    
    def get_path(self, key: str, default: str = "") -> str:
        """Récupère un chemin en le résolvant par rapport au dossier base"""
        path_value = self.get(key, default)
        
        if not path_value:
            return ""
        
        # Si le chemin est absolu, le retourner tel quel
        if os.path.isabs(path_value):
            return path_value
        
        # Sinon, le résoudre par rapport au dossier base
        base_path = self.get("paths.base", ".")
        return os.path.join(base_path, path_value)
    
    def get_model_config(self, model_type: str) -> Dict[str, Any]:
        """Récupère la configuration d'un modèle spécifique"""
        models_config = self.get("models.parameters", {})
        return models_config.get(model_type, {})
    
    def get_vectorizer_config(self) -> Dict[str, Any]:
        """Récupère la configuration du vectorizer"""
        return self.get("preprocessing.vectorizer", {})
    
    def update(self, updates: Dict[str, Any]) -> None:
        """Met à jour la configuration avec de nouvelles valeurs"""
        def _update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = _update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        self.config = _update_dict(self.config, updates)
    
    def save(self, path: Optional[str] = None) -> None:
        """Sauvegarde la configuration dans un fichier"""
        save_path = path or self.config_path
        
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            
            print(f"✅ Configuration sauvegardée: {save_path}")
            
        except Exception as e:
            print(f"❌ Erreur sauvegarde configuration: {e}")
    
    def print_summary(self) -> None:
        """Affiche un résumé de la configuration"""
        print("\n" + "="*50)
        print("📋 RÉSUMÉ DE LA CONFIGURATION")
        print("="*50)
        
        # Informations projet
        app_name = self.get("app.name", "N/A")
        app_version = self.get("app.version", "N/A")
        print(f"📁 Projet: {app_name} v{app_version}")
        
        # Chemins
        data_raw = self.get_path("paths.data.raw")
        data_processed = self.get_path("paths.data.processed")
        models_dir = self.get_path("paths.models.directory")
        
        print(f"\n📂 Chemins:")
        print(f"   Données brutes: {data_raw}")
        print(f"   Données traitées: {data_processed}")
        print(f"   Modèles: {models_dir}")
        
        # Modèles disponibles
        default_model = self.get("models.default", "N/A")
        available_models = self.get("models.available", [])
        
        print(f"\n🤖 Modèles:")
        print(f"   Modèle par défaut: {default_model}")
        print(f"   Modèles disponibles: {', '.join(available_models)}")
        
        # MLflow
        mlflow_exp = self.get("mlflow.experiment_name", "N/A")
        print(f"\n📊 MLflow:")
        print(f"   Expérience: {mlflow_exp}")
        
        print("="*50)

# Instance globale pour usage facile
config_loader = ConfigLoader()

# Fonctions d'accès rapide
def get_config(key: str = "", default: Any = None) -> Any:
    """Fonction utilitaire pour récupérer une valeur de configuration"""
    if not key:
        return config_loader.config
    return config_loader.get(key, default)

def get_path(key: str, default: str = "") -> str:
    """Fonction utilitaire pour récupérer un chemin"""
    return config_loader.get_path(key, default)

def get_model_config(model_type: str) -> Dict[str, Any]:
    """Fonction utilitaire pour récupérer la config d'un modèle"""
    return config_loader.get_model_config(model_type)

if __name__ == "__main__":
    # Test du chargeur de configuration
    loader = ConfigLoader()
    loader.print_summary()
    
    # Exemple d'utilisation
    print("\n🧪 Exemples d'utilisation:")
    print(f"   Modèle par défaut: {get_config('models.default')}")
    print(f"   Chemin des données: {get_path('paths.data.raw')}")
    print(f"   Config logistic: {get_model_config('logistic')}") 
