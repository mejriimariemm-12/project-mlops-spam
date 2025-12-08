# 🚀 Projet MLOps - Détection de Spam SMS

[![CI/CD Pipeline](https://github.com/mejriimariemm-12/project-mlops-spam/actions/workflows/mlops-ci-cd.yml/badge.svg)](https://github.com/mejriimariemm-12/project-mlops-spam/actions)

## 📊 Description
Système complet MLOps pour la détection de spam SMS avec pipeline automatisé.

## ✅ Fonctionnalités
- **API REST** avec FastAPI et Swagger
- **4 modèles ML** prêts à l'emploi
- **Containerisation** Docker
- **CI/CD** automatisé (GitHub Actions)
- **Tracking** MLflow
- **Versioning** DVC

## 🏗️ Architecture
\\\
project-mlops-spam/
├── src/api/              # API FastAPI
├── src/models/           # Modèles ML
├── src/data/             # Préprocessing
├── models/              # Modèles entraînés (.pkl)
├── .github/workflows/   # CI/CD automatisé ✅
├── Dockerfile           # Image Docker
├── docker-compose.yml   # Orchestration
└── requirements.txt     # Dépendances Python
\\\

## 🚀 Installation Rapide
\\\ash
# 1. Cloner
git clone https://github.com/mejriimariemm-12/project-mlops-spam.git
cd project-mlops-spam

# 2. Installer
pip install -r requirements.txt

# 3. Lancer l'API
python -m src.api.main

# 4. Ouvrir dans le navigateur
# http://localhost:8000/docs
\\\

## 📡 Points de terminaison API
| Endpoint | Méthode | Description |
|----------|---------|-------------|
| / | GET | Page d'accueil |
| /health | GET | Vérification santé |
| /predict | POST | Prédiction spam/ham |
| /admin/models | GET | Liste des modèles |
| /docs | GET | Documentation Swagger |

## 📈 Résultats des modèles
| Modèle | Accuracy | F1-Score |
|--------|----------|----------|
| SVM | 0.9950 | 0.9798 |
| Logistic Regression | 0.9749 | 0.9555 |
| Naive Bayes | 0.9718 | 0.9472 |
| Random Forest | 0.9892 | 0.9733 |

## 🔧 Technologies utilisées
- **Python 3.11**
- **FastAPI** - Framework API
- **Scikit-learn** - Modèles ML
- **Docker** - Containerisation
- **GitHub Actions** - CI/CD
- **MLflow** - Tracking expériences
- **DVC** - Versioning données

## 👤 Auteur
**Marie Mejri**  
Projet MLOps - Pipeline complet de détection de spam

## 📄 Licence
Projet académique - Université
