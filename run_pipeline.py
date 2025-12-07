# run_pipeline.py
import subprocess
import sys
import os
import time
from pathlib import Path
import argparse

def print_header(title):
    """Affiche un en-tête stylisé"""
    print("\n" + "="*60)
    print(f"🚀 {title}")
    print("="*60)

def run_command(cmd, description=None):
    """Exécute une commande shell"""
    if description:
        print(f"\n▶️  {description}")
        print(f"   Commande: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
    
    if result.returncode == 0:
        print(f"   ✅ Succès")
        if result.stdout.strip():
            print(f"   Sortie: {result.stdout.strip()[:200]}...")
        return True
    else:
        print(f"   ❌ Erreur (code: {result.returncode})")
        if result.stderr.strip():
            print(f"   Erreur: {result.stderr.strip()[:200]}...")
        return False

def check_dependencies():
    """Vérifie les dépendances"""
    print_header("VÉRIFICATION DES DÉPENDANCES")
    
    dependencies = [
        ("Python", ["python", "--version"]),
        ("DVC", ["dvc", "--version"]),
        ("MLflow", ["mlflow", "--version"]),
        ("pip", ["pip", "--version"])
    ]
    
    all_ok = True
    for name, cmd in dependencies:
        if run_command(cmd, f"Vérification {name}"):
            print(f"   ✓ {name} installé")
        else:
            print(f"   ✗ {name} non installé")
            all_ok = False
    
    return all_ok

def run_dvc_pipeline():
    """Exécute le pipeline DVC complet"""
    print_header("EXÉCUTION DU PIPELINE DVC")
    
    print("\n📊 État actuel du pipeline:")
    run_command(["dvc", "dag"], "Visualisation du pipeline")
    
    print("\n🔄 Exécution du pipeline...")
    if run_command(["dvc", "repro"], "Pipeline DVC"):
        print("\n✅ Pipeline exécuté avec succès!")
        
        # Afficher les résultats
        print("\n📁 FICHIERS GÉNÉRÉS:")
        if os.path.exists("models/"):
            models = os.listdir("models")
            print(f"   models/: {len(models)} fichiers")
        
        if os.path.exists("reports/"):
            reports = os.listdir("reports")
            print(f"   reports/: {len(reports)} fichiers")
        
        return True
    else:
        print("\n❌ Erreur lors de l'exécution du pipeline")
        return False

def train_single_model(model_type="logistic"):
    """Entraîne un modèle spécifique"""
    print_header(f"ENTRAÎNEMENT MODÈLE: {model_type.upper()}")
    
    model_file = f"models/model_{model_type}.pkl"
    
    cmd = [
        "python", "src/models/train.py",
        "--data", "data/processed/sms_clean.csv",
        "--out", model_file,
        "--model_type", model_type
    ]
    
    if run_command(cmd, f"Entraînement {model_type}"):
        print(f"\n✅ Modèle sauvegardé: {model_file}")
        return True
    return False

def train_all_models():
    """Entraîne tous les modèles disponibles"""
    print_header("ENTRAÎNEMENT MULTI-MODÈLES")
    
    models = ["logistic", "svm", "nb", "rf"]
    results = {}
    
    for model_type in models:
        print(f"\n🧠 {model_type.upper()}...")
        success = train_single_model(model_type)
        results[model_type] = success
        time.sleep(1)  # Pause entre les entraînements
    
    # Résumé
    print("\n📊 RÉSUMÉ DE L'ENTRAÎNEMENT:")
    successful = [m for m, s in results.items() if s]
    failed = [m for m, s in results.items() if not s]
    
    if successful:
        print(f"   ✅ Réussi: {', '.join(successful)}")
    if failed:
        print(f"   ❌ Échoué: {', '.join(failed)}")
    
    return len(failed) == 0

def compare_models():
    """Compare tous les modèles entraînés"""
    print_header("COMPARAISON DES MODÈLES")
    
    if not os.path.exists("models/"):
        print("❌ Aucun modèle trouvé. Entraînez d'abord des modèles.")
        return False
    
    cmd = [
        "python", "src/models/evaluate_all.py",
        "--models_dir", "models/",
        "--data", "data/processed/sms_clean.csv",
        "--out", "reports/model_comparison.json"
    ]
    
    if run_command(cmd, "Comparaison des modèles"):
        print("\n📊 RÉSULTATS DE COMPARAISON:")
        
        # Lire et afficher le rapport
        report_path = "reports/model_comparison.json"
        if os.path.exists(report_path):
            import json
            with open(report_path, 'r') as f:
                report = json.load(f)
            
            if "best_model" in report:
                best = report["best_model"]
                print(f"   🏆 Meilleur modèle: {best['name']}")
                print(f"   📈 F1-Score: {best['metrics']['f1_score']:.4f}")
                print(f"   🎯 Accuracy: {best['metrics']['accuracy']:.4f}")
        
        return True
    return False

def start_mlflow_ui():
    """Démarre l'interface MLflow"""
    print_header("INTERFACE MLFLOW")
    
    print("🌐 Démarrage de MLflow UI sur http://localhost:5000")
    print("   Appuyez sur Ctrl+C pour arrêter")
    
    try:
        # Démarrer MLflow en arrière-plan
        import threading
        
        def run_mlflow():
            subprocess.run(["mlflow", "ui", "--port", "5000", "--host", "0.0.0.0"])
        
        thread = threading.Thread(target=run_mlflow, daemon=True)
        thread.start()
        
        print("✅ MLflow démarré")
        print("   Accédez à: http://localhost:5000")
        print("\n⏳ Attente de 5 secondes pour le démarrage...")
        time.sleep(5)
        
        # Essayer d'ouvrir le navigateur
        try:
            import webbrowser
            webbrowser.open("http://localhost:5000")
        except:
            pass
        
        # Garder le script en vie
        input("\n🎯 Appuyez sur Entrée pour continuer...")
        
    except KeyboardInterrupt:
        print("\n🛑 MLflow arrêté")
    except Exception as e:
        print(f"❌ Erreur: {e}")

def clean_outputs():
    """Nettoie les répertoires de sortie"""
    print_header("NETTOYAGE DES SORTIES")
    
    dirs_to_clean = ["models", "reports", "artifacts", "logs"]
    
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            import shutil
            try:
                shutil.rmtree(dir_name)
                print(f"   🗑️  Supprimé: {dir_name}/")
            except Exception as e:
                print(f"   ❌ Erreur suppression {dir_name}: {e}")
        else:
            print(f"   ✓ Déjà propre: {dir_name}/")
    
    print("\n✅ Nettoyage terminé")

def main_menu():
    """Menu principal interactif"""
    while True:
        print_header("MLOps PIPELINE - SPAM DETECTION")
        
        print("1. 🔄 Exécuter le pipeline DVC complet")
        print("2. 🤖 Entraîner tous les modèles")
        print("3. 🧠 Entraîner un modèle spécifique")
        print("4. 📊 Comparer les modèles")
        print("5. 📈 Démarrer MLflow UI")
        print("6. 🧹 Nettoyer les sorties")
        print("7. ✅ Vérifier les dépendances")
        print("8. 🚪 Quitter")
        
        choice = input("\n👉 Choix (1-8): ").strip()
        
        if choice == "1":
            if check_dependencies():
                run_dvc_pipeline()
        elif choice == "2":
            train_all_models()
        elif choice == "3":
            model_type = input("Type de modèle (logistic/svm/nb/rf): ").strip().lower()
            if model_type in ["logistic", "svm", "nb", "rf"]:
                train_single_model(model_type)
            else:
                print("❌ Type de modèle invalide")
        elif choice == "4":
            compare_models()
        elif choice == "5":
            start_mlflow_ui()
        elif choice == "6":
            clean_outputs()
        elif choice == "7":
            check_dependencies()
        elif choice == "8":
            print("\n👋 Au revoir!")
            break
        else:
            print("❌ Choix invalide")
        
        input("\n⏎ Appuyez sur Entrée pour continuer...")

def main():
    """Point d'entrée principal"""
    parser = argparse.ArgumentParser(description="Pipeline MLOps pour détection de spam")
    parser.add_argument("--mode", choices=["auto", "menu", "dvc", "train", "compare", "clean"],
                       default="menu", help="Mode d'exécution")
    parser.add_argument("--model", choices=["logistic", "svm", "nb", "rf"],
                       help="Type de modèle pour l'entraînement")
    
    args = parser.parse_args()
    
    if args.mode == "auto":
        # Mode automatique: vérif → pipeline → comparaison
        print_header("MODE AUTOMATIQUE")
        if check_dependencies():
            run_dvc_pipeline()
            compare_models()
    
    elif args.mode == "dvc":
        run_dvc_pipeline()
    
    elif args.mode == "train":
        if args.model:
            train_single_model(args.model)
        else:
            train_all_models()
    
    elif args.mode == "compare":
        compare_models()
    
    elif args.mode == "clean":
        clean_outputs()
    
    else:  # menu par défaut
        main_menu()

if __name__ == "__main__":
    main() 
