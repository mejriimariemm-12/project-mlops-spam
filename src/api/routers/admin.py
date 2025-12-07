# src/api/routers/admin.py
from fastapi import APIRouter, HTTPException, BackgroundTasks
import subprocess
import os
import json
from typing import Dict, Any
from src.api.dependencies import initialize_model, get_model_info
from src.api.schemas import ModelInfoResponse, RetrainRequest

router = APIRouter(prefix="/admin", tags=["admin"])

@router.get("/model/info", response_model=ModelInfoResponse)
async def get_model_information():
    """
    Retourne les informations détaillées du modèle chargé
    """
    model_info = get_model_info()
    
    if not model_info:
        raise HTTPException(
            status_code=404,
            detail="No model information available"
        )
    
    return ModelInfoResponse(**model_info)

@router.post("/model/reload")
async def reload_model():
    """
    Recharge le modèle depuis le disque
    """
    try:
        success = initialize_model()
        
        if success:
            return {
                "status": "success",
                "message": "Model reloaded successfully",
                "model_info": get_model_info()
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to reload model"
            )
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reloading model: {str(e)}"
        )

@router.post("/model/retrain")
async def retrain_model(request: RetrainRequest, background_tasks: BackgroundTasks):
    """
    Ré-entraîne le modèle en arrière-plan
    
    - **model_type**: Type de modèle à entraîner
    - **data_path**: Chemin vers les nouvelles données (optionnel)
    - **save_model**: Sauvegarder le nouveau modèle
    """
    # Cette fonction est exécutée en arrière-plan
    def train_in_background(model_type: str, data_path: str = None):
        try:
            # Chemin par défaut des données
            if data_path is None:
                data_path = "data/processed/sms_clean.csv"
            
            # Chemin de sortie
            output_path = f"models/model_{model_type}_retrained.pkl"
            
            # Commande d'entraînement
            cmd = [
                "python", "src/models/train.py",
                "--data", data_path,
                "--out", output_path,
                "--model_type", model_type
            ]
            
            # Exécuter l'entraînement
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Ré-entraînement réussi: {output_path}")
            else:
                print(f"❌ Erreur ré-entraînement: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Exception ré-entraînement: {e}")
    
    # Démarrer l'entraînement en arrière-plan
    background_tasks.add_task(
        train_in_background, 
        request.model_type, 
        request.data_path
    )
    
    return {
        "status": "accepted",
        "message": f"Retraining {request.model_type} model started in background",
        "job_id": f"retrain_{request.model_type}_{os.urandom(4).hex()}",
        "estimated_time": "2-5 minutes"
    }

@router.get("/pipeline/status")
async def pipeline_status():
    """
    Vérifie l'état du pipeline DVC
    """
    try:
        # Vérifier l'état DVC
        status_result = subprocess.run(
            ["dvc", "status"], 
            capture_output=True, 
            text=True
        )
        
        dag_result = subprocess.run(
            ["dvc", "dag"], 
            capture_output=True, 
            text=True
        )
        
        return {
            "dvc_status": status_result.stdout if status_result.returncode == 0 else status_result.stderr,
            "pipeline_dag": dag_result.stdout if dag_result.returncode == 0 else dag_result.stderr,
            "data_files": {
                "raw": os.path.exists("data/raw/sms.csv"),
                "processed": os.path.exists("data/processed/sms_clean.csv")
            },
            "model_files": os.listdir("models") if os.path.exists("models") else []
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error checking pipeline: {str(e)}"
        )

@router.get("/metrics")
async def get_metrics():
    """
    Retourne les métriques du modèle
    """
    try:
        metrics_files = []
        if os.path.exists("reports"):
            for file in os.listdir("reports"):
                if file.endswith(".json"):
                    file_path = os.path.join("reports", file)
                    with open(file_path, 'r') as f:
                        metrics_files.append({
                            "file": file,
                            "content": json.load(f)
                        })
        
        return {
            "available_metrics": metrics_files,
            "model_metrics": get_model_info().get("metrics", {})
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error reading metrics: {str(e)}"
        ) 
