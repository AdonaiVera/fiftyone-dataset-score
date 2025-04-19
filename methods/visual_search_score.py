import os
import torch
import numpy as np
import logging
import gc
import clip
import torch.nn as nn
import requests
import tempfile
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='dataset_difficulty.log'
)
logger = logging.getLogger("visual_search_score")

# Model constants
CLIP_MODEL_NAME = "ViT-B/32"
REGRESSOR_MODEL_NAME = "adonaivera/fiftyone-dataset-score-regressor"

# Global variables for models
clip_model = None
clip_preprocess = None
regressor_model = None

def get_device():
    """Get the appropriate device for model inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon) detected. Using CPU for inference to avoid type mismatch issues.")
        return torch.device("cpu")
    return torch.device("cpu")

def clean_memory():
    """Clean up GPU memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

class DifficultyPredictor(nn.Module):
    """A simple linear regressor for visual search time prediction."""
    def __init__(self, input_dim):
        super(DifficultyPredictor, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def load_models():
    """Load CLIP and regressor models."""
    global clip_model, clip_preprocess, regressor_model
    
    device = get_device()
    logger.info(f"Loading models on device: {device}")
    
    # Load CLIP model
    if clip_model is None:
        logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME}")
        clip_model, clip_preprocess = clip.load(CLIP_MODEL_NAME, device=device)
    
    # Load regressor model
    if regressor_model is None:
        logger.info(f"Loading regressor model: {REGRESSOR_MODEL_NAME}")
        
        # Create a temporary directory to download the model
        with tempfile.TemporaryDirectory() as temp_dir:
            # Download model file
            logger.info("Downloading model file...")
            
            # Get CLIP feature dimension
            input_dim = clip_model.visual.output_dim
            logger.info(f"CLIP feature dimension: {input_dim}")
            
            # Create the model
            regressor_model = DifficultyPredictor(input_dim=input_dim)
            
            # Download model weights
            weights_url = f"https://huggingface.co/{REGRESSOR_MODEL_NAME}/resolve/main/model.pt"
            weights_path = os.path.join(temp_dir, "model.pt")
            response = requests.get(weights_url)
            with open(weights_path, "wb") as f:
                f.write(response.content)
            
            # Load weights
            state_dict = torch.load(weights_path, map_location=device)
            regressor_model.load_state_dict(state_dict)
        
        # Move model to device and set to eval mode
        regressor_model.to(device)
        regressor_model.eval()
    
    return device

def visual_search_score(image, device):
    """
    Compute visual search time difficulty score for an image.
    
    Args:
        image: PIL Image
        device: torch device
        
    Returns:
        float: Visual search time difficulty score
    """
    global clip_model, clip_preprocess, regressor_model
    
    logger.info("Starting visual search score calculation")
    
    try:
        # Ensure models are loaded
        if clip_model is None or regressor_model is None:
            logger.info("Models not loaded, loading now")
            device = load_models()
        
        # Preprocess image for CLIP
        logger.info("Preprocessing image for CLIP")
        image_input = clip_preprocess(image).unsqueeze(0).to(device)
        
        # Get CLIP image features
        logger.info("Getting CLIP image features")
        with torch.no_grad():
            image_features = clip_model.encode_image(image_input)
            # Ensure consistent data type
            image_features = image_features.to(torch.float32)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Pass features through regressor
        logger.info("Passing features through regressor")
        with torch.no_grad():
            # Ensure consistent data type
            image_features = image_features.to(torch.float32)
            outputs = regressor_model(image_features)
            score = outputs.item()
            logger.info(f"Final score: {score}")
        
        logger.info("Visual search score calculation complete")
        return score
    
    except Exception as e:
        logger.error(f"Error in visual_search_score: {str(e)}")
        # Return a default score in case of error
        return 0.5 