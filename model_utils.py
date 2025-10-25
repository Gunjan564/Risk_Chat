import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
# from torch.cuda.amp import autocast 
from joblib import load 
import warnings

# Suppress warnings, useful for a clean deployed application
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- GLOBAL CONFIGURATION (ADJUSTED FOR LOCAL/REPLIT DEPLOYMENT) ---

# Base directory for the 'models' folder, relative to the project root (SentimentBot/)
MODEL_BASE_DIR = "models"

# The directory containing meta_model.joblib and ensemble_metadata.pt.
# Since your file structure shows meta_model.joblib and ensemble_metadata.pt *directly*
# under the 'models' folder, we simplify this path.
# Assuming the file structure from the image:
ENSEMBLE_SAVE_PATH = MODEL_BASE_DIR 


# ✅ FINAL CORRECT PATH MAPPING
# These paths now correctly point to the folders inside the 'models' directory.
LOCAL_MODEL_PATHS = {
    "xlnet": os.path.join(MODEL_BASE_DIR, "xlnet"),
    "distilbert": os.path.join(MODEL_BASE_DIR, "distill"), 
    "mental-roberta": os.path.join(MODEL_BASE_DIR, "aimh") 
}


class MentalHealthEnsemble:
    def __init__(self, X_val=None, y_val=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ["TOKENIZERS_PARALLELISM"] = "true"

        # Use local paths for model loading
        self.model_paths = LOCAL_MODEL_PATHS

        self.models = self._load_models()
        self.meta_model = None
        self.val_metrics = None
        self.confidence_thresholds = {
            'no risk': 0.85, 
            'low': 0.70,
            'moderate': 0.65,
            'high': 0.75
        }
        self.label2id = {"low": 0, "moderate": 1, "high": 2, "no risk": 3}
        self.id2label = {v: k for k, v in self.label2id.items()}
        
        if X_val is not None:
            self._initialize_validation(*self._ensure_text_format(X_val, y_val))

    def _ensure_text_format(self, texts, labels=None):
        """Convert input to list of strings and handle NaN values"""
        if isinstance(texts, (np.ndarray, pd.Series)):
            texts = texts.tolist()
        texts = [str(x) if pd.notna(x) else "" for x in texts]

        if labels is not None:
            if isinstance(labels, (np.ndarray, pd.Series)):
                labels = labels.tolist()
            return texts, labels
        return texts

    def _load_models(self):
        """Load models with better error handling"""
        models = {}
        for name, path in self.model_paths.items():
            try:
                print(f"Loading {name} from {path}...")
                model = AutoModelForSequenceClassification.from_pretrained(path)
                tokenizer = AutoTokenizer.from_pretrained(path)
                models[name] = {
                    "model": model.to(self.device).eval(),
                    "tokenizer": tokenizer
                }
                print(f"✅ {name} loaded successfully")
            except Exception as e:
                print(f"❌ Failed to load {name} from {path}: {str(e)}")
                continue
        return models

    def _initialize_validation(self, X_val, y_val):
        """Initialize validation metrics (implementation removed for brevity)"""
        print("Computing validation metrics...")
        pass

    def train_meta_model(self, X_train, y_train, X_val=None, y_val=None):
        """Trains logistic regression meta-model (implementation removed for brevity)"""
        print("Meta-model training stub - usually run offline.")
        pass

    def _predict_batch(self, model, tokenizer, texts, max_length):
        """Batch prediction helper, now with NaN/Error protection."""
        texts = self._ensure_text_format(texts)
        inputs = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(self.device)

        with torch.no_grad():
            outputs = model(**inputs)
            
            # --- CRITICAL FIX: Check for NaN Logits ---
            if outputs.logits is None or torch.isnan(outputs.logits).any():
                # If logits are NaN (a known issue with fine-tuned models on edge cases), 
                # we return None so the model is skipped in the ensemble.
                print(f"⚠️ Warning: Model output contained NaN logits for input. Skipping batch.")
                return None
            
            # --- End FIX ---
            
            return torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()

    def predict(self, texts, max_length=128):
        """
        Predicts risk levels with confidence scores using the ensemble.
        """
        if isinstance(texts, str):
            texts = [texts]

        all_probs = []
        for name, model_info in self.models.items():
            try:
                probs = self._predict_batch(model_info["model"], model_info["tokenizer"], texts, max_length)
                
                # Check for the None return value (due to NaN)
                if probs is None:
                    continue 
                
                all_probs.append(probs)
                
            except Exception as e:
                print(f"⚠️ Skipping {name} due to unexpected error: {str(e)}")
                continue

        if not all_probs:
            raise RuntimeError("All models failed - cannot make predictions")

        if self.meta_model:
            # Use meta-model for stacking prediction
            stacked_probs = np.hstack(all_probs)
            predictions = self.meta_model.predict(stacked_probs)
            confidences = np.max(self.meta_model.predict_proba(stacked_probs), axis=1)
        else:
            # Fallback to simple weighted average
            avg_probs = np.mean(all_probs, axis=0)
            predictions = np.argmax(avg_probs, axis=1)
            confidences = np.max(avg_probs, axis=1)

        # Convert to readable labels
        risk_labels = [self.id2label.get(p, "unknown") for p in predictions]

        # Ensure single output if input was single string
        if len(risk_labels) == 1 and isinstance(texts, list) and len(texts) == 1:
             return risk_labels[0], confidences.tolist()[0]
             
        return risk_labels, confidences.tolist()


# ------------------ LOAD FUNCTION (Required by app.py) ------------------

def load_ensemble_models(device=None):
    """
    Load the trained ensemble and its meta-model for inference.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # The save_dir is the 'models' directory itself, as per the file structure image.
    save_dir = ENSEMBLE_SAVE_PATH

    # Initialize empty ensemble instance
    ensemble = MentalHealthEnsemble.__new__(MentalHealthEnsemble)
    ensemble.device = device
    ensemble.model_paths = LOCAL_MODEL_PATHS 

    try:
        # Load metadata (ensemble_metadata.pt is directly in the save_dir)
        metadata = torch.load(os.path.join(save_dir, "ensemble_metadata.pt"), weights_only=False)
    except FileNotFoundError as e:
        print(f"❌ CRITICAL ERROR: Metadata file not found in {save_dir}. {e}")
        metadata = {}
        
    ensemble.id2label = metadata.get('id2label', {0: "low", 1: "moderate", 2: "high", 3: "no risk"})
    ensemble.label2id = metadata.get('label2id', {v: k for k, v in ensemble.id2label.items()})

    # Reload base models using LOCAL_MODEL_PATHS
    ensemble.models = {}
    for name, path in ensemble.model_paths.items():
        full_path = path 
        try:
            ensemble.models[name] = {
                "model": AutoModelForSequenceClassification.from_pretrained(full_path).to(device).eval(),
                "tokenizer": AutoTokenizer.from_pretrained(full_path)
            }
            print(f"✅ Reloaded base model: {name} from {full_path}")
        except Exception as e:
            print(f"❌ Failed to reload base model {name}: {str(e)}")
            continue

    # Load meta-model (meta_model.joblib is directly in the save_dir)
    meta_model_path = os.path.join(save_dir, "meta_model.joblib")
    if os.path.exists(meta_model_path):
        try:
            ensemble.meta_model = load(meta_model_path)
            print("✅ Loaded meta-model (Stacking Classifier)")
        except Exception as e:
            print(f"❌ Failed to load meta-model: {str(e)}")
            ensemble.meta_model = None
    else:
        ensemble.meta_model = None
        print("⚠️ Meta-model not found. Falling back to weighted average.")


    # Default confidence thresholds
    ensemble.confidence_thresholds = {
        'no risk': 0.85, 
        'low': 0.70,
        'moderate': 0.65,
        'high': 0.75
    }

    if not ensemble.models:
        raise Exception("Failed to load any base models. Check the 'models' directory content.")

    print("✅ Ensemble system loaded successfully for Streamlit.")
    return ensemble