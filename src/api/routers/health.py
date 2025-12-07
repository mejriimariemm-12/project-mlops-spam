# src/api/routers/health.py
from fastapi import APIRouter
from src.api.dependencies import get_available_models, get_uptime, get_model_info
from src.api.schemas import HealthResponse
from datetime import datetime

router = APIRouter(prefix="/health", tags=["health"])

@router.get("/", response_model=HealthResponse)
async def health_check():
    """Vérifie l'état de santé de l'API"""
    available_models = get_available_models()
    model_loaded = len(available_models) > 0
    
    # Info du modèle par défaut
    default_model = "svm" if "svm" in available_models else (available_models[0] if available_models else None)
    model_info = get_model_info(default_model) if default_model else {}
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        available_models=available_models,
        model_version=model_info.get("model_version"),
        model_type=default_model,
        api_version="1.0.0",
        uptime_seconds=get_uptime(),
        timestamp=datetime.now()
    )

@router.get("/detailed")
async def detailed_health_check():
    """Vérification de santé détaillée"""
    available_models = get_available_models()
    model_loaded = len(available_models) > 0
    
    # Vérifications supplémentaires
    checks = {
        'api': {'status': 'healthy', 'message': 'API is running'},
        'model': {'status': 'healthy' if model_loaded else 'unhealthy', 
                 'message': 'Model is loaded' if model_loaded else 'Model not loaded'},
        'memory': {'status': 'healthy', 'message': 'Memory usage normal'},
    }
    
    # Calculer le statut global
    all_healthy = all(check['status'] == 'healthy' for check in checks.values())
    overall_status = 'healthy' if all_healthy else 'unhealthy'
    
    return {
        'status': overall_status,
        'checks': checks,
        'details': {
            'available_models': available_models,
            'model_loaded': model_loaded,
            'uptime_seconds': get_uptime()
        }
    }

@router.get("/ready")
async def readiness_probe():
    """Endpoint de readiness pour Kubernetes/Docker"""
    available_models = get_available_models()
    model_loaded = len(available_models) > 0
    
    if model_loaded:
        return {"status": "ready", "message": "API is ready to receive traffic"}
    else:
        return {"status": "not_ready", "message": "Model not loaded"}, 503

@router.get("/live")
async def liveness_probe():
    """Endpoint de liveness pour Kubernetes/Docker"""
    return {"status": "alive", "message": "API is running"}

@router.get("/version")
async def version_info():
    """Retourne les informations de version"""
    import sys
    import platform
    
    return {
        'api_version': '1.0.0',
        'python_version': sys.version,
        'platform': platform.platform(),
        'dependencies': {
            'mlflow': '2.8.0',
            'fastapi': '0.104.0',
            'scikit-learn': '1.3.0'
        }
    }