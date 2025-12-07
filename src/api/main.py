# src/api/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import time
from datetime import datetime
import uvicorn
import yaml
import os

from src.api.dependencies import initialize_models 

from src.api.routers import predict, health, admin
# Charger la configuration
def load_config():
    config_path = "config/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

# Créer l'application FastAPI
app = FastAPI(
    title="Spam Detection API",
    description="API de détection de spam SMS avec MLOps",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En production, spécifier les origines
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware de logging
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = (time.time() - start_time) * 1000
    formatted_time = "{0:.2f}".format(process_time)
    
    print(f"{request.client.host} - \"{request.method} {request.url.path}\" {response.status_code} - {formatted_time}ms")
    
    return response

# Inclure les routeurs
app.include_router(predict.router)
app.include_router(health.router)
app.include_router(admin.router)

# Événements de démarrage/arrêt
@app.on_event("startup")
async def startup_event():
    """Exécuté au démarrage de l'API"""
    print(">>> Démarrage de l'API Spam Detection...")
    
    # Initialiser le modèle
    print("[MODEL] Initialisation du modèle...")
    success = initialize_models()
    
    if success:
        print("[OK] Modèle chargé avec succès")
    else:
        print("[WARN] Modèle non chargé. L'API fonctionnera en mode limité.")
    
    print(f"[DATE] Démarrage terminé à {datetime.now().isoformat()}")

@app.on_event("shutdown")
async def shutdown_event():
    """Exécuté à l'arrêt de l'API"""
    print("[STOP] Arrêt de l'API...")

# Route racine
@app.get("/")
async def root():
    """Page d'accueil de l'API"""
    return {
        "message": "Bienvenue sur l'API de détection de spam SMS",
        "version": "1.0.0",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "predict": "/predict",
            "admin": "/admin"
        },
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

# Gestionnaire d'erreurs global
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error": str(exc),
            "path": request.url.path,
            "timestamp": datetime.now().isoformat()
        }
    )

# Point d'entrée pour l'exécution directe
if __name__ == "__main__":
    # Configuration depuis le fichier config
    api_config = config.get("api", {})
    host = api_config.get("host", "0.0.0.0")
    port = api_config.get("port", 8000)
    reload = api_config.get("reload", True)
    
    print(f"[NET] Démarrage sur {host}:{port}")
    print(f"[DOC] Documentation: http://{host}:{port}/docs")
    
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        reload=reload
    )
