# Utiliser Python 3.11
FROM python:3.11-slim

# Créer le dossier de travail
WORKDIR /app

# Copier le fichier de dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier tout le projet
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p models data reports logs

# Port de l'API
EXPOSE 8000

# Commande pour démarrer
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
