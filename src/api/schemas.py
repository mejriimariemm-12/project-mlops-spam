# src/api/schemas.py - Version complète corrigée
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# Simple string pour model_type (pas d'Enum pour éviter les problèmes)
class TextRequest(BaseModel):
    """Requête pour la prédiction"""
    text: str = Field(..., min_length=1, max_length=1000, description="Texte SMS à classifier")
    model_type: Optional[str] = Field(default="svm", description="Type de modèle: logistic, svm, nb, rf")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "WINNER! You have won a free ticket to Bahamas!",
                "model_type": "svm"
            }
        }

class PredictionResponse(BaseModel):
    """Réponse de prédiction"""
    text: str
    cleaned_text: Optional[str] = Field(None, description="Texte après nettoyage")
    prediction: str = Field(..., description="ham ou spam")
    is_spam: bool
    confidence: float = Field(..., ge=0, le=1, description="Confiance de la prédiction")
    model_used: str = Field(..., description="Modèle utilisé pour la prédiction")
    model_version: str
    processing_time_ms: float
    probabilities: Optional[Dict[str, float]] = Field(None, description="Probabilités pour chaque classe")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "WINNER! You have won a free ticket to Bahamas!",
                "cleaned_text": "winner! you have won a free ticket to bahamas!",
                "prediction": "spam",
                "is_spam": True,
                "confidence": 0.95,
                "model_used": "svm",
                "model_version": "v1.0",
                "processing_time_ms": 12.5,
                "probabilities": {"ham": 0.05, "spam": 0.95}
            }
        }

class HealthResponse(BaseModel):
    """Réponse du health check"""
    status: str
    model_loaded: bool
    available_models: List[str]
    model_version: Optional[str]
    model_type: Optional[str]
    api_version: str
    uptime_seconds: float
    timestamp: datetime
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "available_models": ["logistic", "svm", "nb", "rf"],
                "model_version": "v1.0",
                "model_type": "svm",
                "api_version": "1.0.0",
                "uptime_seconds": 3600.5,
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

class ModelInfoResponse(BaseModel):
    """Informations sur le modèle"""
    model_type: str
    model_version: str
    training_date: Optional[str]
    features_count: Optional[int]
    vocabulary_size: Optional[int]
    metrics: Dict[str, float]
    parameters: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_type": "svm",
                "model_version": "v1.0",
                "training_date": "2024-01-15T10:00:00",
                "features_count": 5000,
                "vocabulary_size": 4800,
                "metrics": {"accuracy": 0.9798, "f1_score": 0.9798},
                "parameters": {"C": 1.0, "probability": True}
            }
        }

class RetrainRequest(BaseModel):
    """Requête pour ré-entraîner le modèle"""
    model_type: str = Field("svm", description="Type de modèle: logistic, svm, nb, rf")
    data_path: Optional[str] = Field(None, description="Chemin vers nouvelles données")
    test_size: Optional[float] = Field(0.2, ge=0.1, le=0.5, description="Taille du jeu de test")
    save_model: bool = Field(True, description="Sauvegarder le nouveau modèle")

class BatchPredictionRequest(BaseModel):
    """Requête pour prédiction par lots"""
    texts: List[str] = Field(..., min_items=1, max_items=1000)
    model_type: Optional[str] = Field(default="svm", description="Type de modèle à utiliser pour toutes les prédictions")
    
    class Config:
        json_schema_extra = {
            "example": {
                "texts": [
                    "Free entry to win a prize",
                    "Hey, are we still meeting tomorrow?"
                ],
                "model_type": "svm"
            }
        }

class BatchPredictionResponse(BaseModel):
    """Réponse pour prédiction par lots"""
    predictions: List[PredictionResponse]
    total_texts: int
    spam_count: int
    ham_count: int
    model_used: str
    processing_time_ms: float
    avg_processing_time_ms: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [],
                "total_texts": 2,
                "spam_count": 1,
                "ham_count": 1,
                "model_used": "svm",
                "processing_time_ms": 25.0,
                "avg_processing_time_ms": 12.5
            }
        }

class ErrorResponse(BaseModel):
    """Réponse d'erreur"""
    detail: str
    error_code: Optional[str] = Field(None, description="Code d'erreur personnalisé")
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_schema_extra = {
            "example": {
                "detail": "Model not loaded",
                "error_code": "MODEL_NOT_LOADED",
                "timestamp": "2024-01-15T10:30:00Z"
            }
        }

class AvailableModelsResponse(BaseModel):
    """Réponse avec liste des modèles disponibles"""
    available_models: List[str]
    default_model: str
    best_model: str
    models_info: Dict[str, Dict[str, Any]]
    
    class Config:
        json_schema_extra = {
            "example": {
                "available_models": ["logistic", "svm", "nb", "rf"],
                "default_model": "svm",
                "best_model": "svm",
                "models_info": {
                    "svm": {"f1_score": 0.9798, "accuracy": 0.9950},
                    "logistic": {"f1_score": 0.9555, "accuracy": 0.9749}
                }
            }
        }