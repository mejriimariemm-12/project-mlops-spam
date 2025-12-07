# src/api/routers/predict.py - Version corrigée
from fastapi import APIRouter, HTTPException
from typing import List
import time
from src.api.dependencies import predict_text, get_model, get_available_models
from src.api.schemas import (
    TextRequest, PredictionResponse, BatchPredictionRequest, 
    BatchPredictionResponse, ErrorResponse
)

router = APIRouter(prefix="/predict", tags=["prediction"])

@router.post("/", response_model=PredictionResponse)
async def predict(request: TextRequest):
    """
    Prédit si un message SMS est du spam ou non
    
    - **text**: Le texte du message SMS à classifier
    - **model_type**: Type de modèle à utiliser (logistic, svm, nb, rf)
    """
    try:
        # Utiliser le model_type de la requête (c'est maintenant un str, pas un Enum)
        result = predict_text(request.text, request.model_type)
        
        return PredictionResponse(**result)
        
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@router.post("/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Prédictions par lots pour plusieurs messages SMS
    
    - **texts**: Liste des textes à classifier (max 1000)
    - **model_type**: Type de modèle à utiliser pour toutes les prédictions
    """
    try:
        start_time = time.time()
        predictions = []
        spam_count = 0
        ham_count = 0
        
        # model_type est déjà un str dans le schéma
        model_type = request.model_type or "svm"
        
        for text in request.texts:
            result = predict_text(text, model_type)
            predictions.append(PredictionResponse(**result))
            
            if result['is_spam']:
                spam_count += 1
            else:
                ham_count += 1
        
        processing_time = (time.time() - start_time) * 1000
        avg_time = processing_time / len(request.texts) if request.texts else 0
        
        return BatchPredictionResponse(
            predictions=predictions,
            total_texts=len(request.texts),
            spam_count=spam_count,
            ham_count=ham_count,
            model_used=model_type,
            processing_time_ms=processing_time,
            avg_processing_time_ms=avg_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch prediction error: {str(e)}"
        )

@router.get("/test")
async def test_prediction():
    """
    Endpoint de test avec des exemples prédéfinis
    """
    test_examples = [
        "WINNER! You have won a free ticket to Bahamas!",
        "Hey, are we still meeting tomorrow?",
        "URGENT: Your account has been compromised. Click here to secure.",
        "Thanks for your order. It will be delivered tomorrow.",
        "CONGRATULATIONS! You've been selected for a free iPhone."
    ]
    
    results = []
    for example in test_examples:
        try:
            result = predict_text(example, "svm")
            results.append(result)
        except Exception as e:
            results.append({
                "text": example,
                "error": str(e)
            })
    
    return {
        "test_examples": results,
        "total_tests": len(test_examples),
        "successful": len([r for r in results if "error" not in r])
    }

@router.get("/models")
async def list_models():
    """
    Liste tous les modèles disponibles
    """
    available_models = get_available_models()
    
    # Lire le fichier de comparaison pour le meilleur modèle
    best_model = "svm"  # Par défaut
    try:
        import json
        with open("reports/comparison.json", "r") as f:
            comparison = json.load(f)
            best_model = comparison.get("best_model", {}).get("name", "svm")
    except:
        pass
    
    return {
        "available_models": available_models,
        "default_model": "svm",
        "best_model": best_model,
        "count": len(available_models)
    }