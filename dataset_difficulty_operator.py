import os
os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'

# Set matplotlib backend to Agg to avoid GUI issues
import matplotlib
matplotlib.use('Agg')

import fiftyone.operators as foo
from fiftyone.operators import types
import torch
import numpy as np
import logging
from tqdm import tqdm
import gc
from typing import List, Dict, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image
import clip
import torch.nn as nn
import torch.nn.functional as F
import requests
import tempfile
import json
import os

# Import utility functions
from .utils import _execution_mode

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='dataset_difficulty.log'
)
logger = logging.getLogger("dataset_difficulty_operator")

# Batch processing constants
DEFAULT_BATCH_SIZE = 8
MAX_MEMORY_THRESHOLD = 0.9  

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
    
    # Ensure models are loaded
    if clip_model is None or regressor_model is None:
        device = load_models()
    
    # Preprocess image for CLIP
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    
    # Get CLIP image features
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input)
        # Ensure consistent data type
        image_features = image_features.to(torch.float32)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    
    # Pass features through regressor
    with torch.no_grad():
        # Ensure consistent data type
        image_features = image_features.to(torch.float32)
        outputs = regressor_model(image_features)
        score = outputs.item()
        logger.info(f"Final score: {score}")
    
    return score

def batch_generator(dataset, batch_size: int):
    """Generate batches of samples from the dataset."""
    samples = list(dataset.iter_samples())  
    for i in range(0, len(samples), batch_size):
        yield samples[i:i + batch_size]

def plot_to_base64(fig):
    """Convert a matplotlib figure to a base64 encoded string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_str

class DatasetDifficultyScoring(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="dataset_difficulty_scoring",
            label="Dataset Difficulty Scoring",
            description="Score dataset samples based on Visual Search Time difficulty metric",
            icon="/assets/difficulty-icon.svg", 
            dynamic=True,
        )

    def resolve_input(self, ctx):
        inputs = types.Object()

        # Add dataset percentage input
        inputs.float(
            "dataset_percentage",
            default=1.0,
            label="Dataset Percentage",
            description="Percentage of the dataset to analyze (0.0-1.0)",
            min=0.0,
            max=1.0
        )

        # # Add difficulty metrics selection  
        # metrics_radio = types.RadioGroup()
        # metrics_radio.add_choice("all", label="All Metrics")
        # metrics_radio.add_choice("visual", label="Visual Search Time")
        # metrics_radio.add_choice("odd", label="Object Detection Difficulty")
        # metrics_radio.add_choice("cnn", label="CNN Difficulty Predictor")
        # metrics_radio.add_choice("zigzag", label="Zigzag Learning")
        # 
        # inputs.enum(
        #     "difficulty_metrics",
        #     metrics_radio.values(),
        #     label="Difficulty Metrics",
        #     description="Select which difficulty metrics to compute",
        #     default="all"
        # )

        # Add visualization options
        inputs.bool(
            "show_histograms",
            default=True,
            label="Show Histograms",
            description="Generate histograms of difficulty scores"
        )

        # # Add scatter plots option  
        # inputs.bool(
        #     "show_scatter_plots",
        #     default=True,
        #     label="Show Scatter Plots",
        #     description="Generate scatter plots of difficulty metrics"
        # )

        # Add output fields
        inputs.str(
            "output_field",
            default="difficulty_score",
            label="Difficulty Score Field",
            description="Field to store the final difficulty score"
        )

        # # Add metrics field  
        # inputs.str(
        #     "metrics_field",
        #     default="difficulty_metrics",
        #     label="Metrics Field",
        #     description="Field to store individual difficulty metrics"
        # )

        # # Add recommendation field  
        # inputs.str(
        #     "recommendation_field",
        #     default="difficulty_recommendation",
        #     label="Recommendation Field",
        #     description="Field to store recommendations based on difficulty"
        # )

        # Add execution mode input
        _execution_mode(ctx, inputs)

        # Add batch size input
        inputs.int(
            "batch_size",
            default=8,
            label="Batch Size",
            description="Number of samples to process in each batch",
            min=1,
            max=32
        )

        return types.Property(inputs)

    def resolve_delegation(self, ctx):
        return ctx.params.get("delegate", False)

    def execute(self, ctx):
        logger.info("Starting dataset difficulty scoring operation")
        
        # Get parameters
        dataset_percentage = ctx.params.get("dataset_percentage", 1.0)
        # difficulty_metrics = ctx.params.get("difficulty_metrics", "all")  # Commented for future use
        show_histograms = ctx.params.get("show_histograms", True)
        # show_scatter_plots = ctx.params.get("show_scatter_plots", True)  # Commented for future use
        output_field = ctx.params.get("output_field", "difficulty_score")
        # metrics_field = ctx.params.get("metrics_field", "difficulty_metrics")  # Commented for future use
        # recommendation_field = ctx.params.get("recommendation_field", "difficulty_recommendation")  # Commented for future use
        batch_size = ctx.params.get("batch_size", DEFAULT_BATCH_SIZE)
        
        logger.info(f"Parameters: dataset_percentage={dataset_percentage}, "
                   f"show_histograms={show_histograms}")

        # Load models and get device
        device = load_models()
        
        try:
            # Get dataset
            dataset = ctx.dataset
            
            # Apply dataset percentage if needed
            if dataset_percentage < 1.0:
                num_samples = int(len(dataset) * dataset_percentage)
                dataset = dataset.limit(num_samples)
                logger.info(f"Limited dataset to {num_samples} samples ({dataset_percentage*100}%)")
            
            total_samples = len(dataset)
            logger.info(f"Processing {total_samples} samples in batches of {batch_size}")
            
            # Initialize metrics storage
            all_scores = []
            
            # # Initialize metrics storage for multiple metrics  
            # all_metrics = {
            #     "visual": [],
            #     "odd": [],
            #     "cnn": [],
            #     "zigzag": [],
            #     "final_score": []
            # }
            
            # Process dataset in batches
            with torch.no_grad():  # Disable gradient computation
                for batch in tqdm(batch_generator(dataset, batch_size), total=(total_samples + batch_size - 1) // batch_size):
                    batch_images = []
                    batch_filepaths = []
                    
                    # Prepare batch data
                    for sample in batch:
                        try:
                            image = Image.open(sample.filepath)
                            batch_images.append(image)
                            batch_filepaths.append(sample.filepath)
                        except (FileNotFoundError, OSError) as e:
                            logger.warning(f"Could not open image {sample.filepath}: {str(e)}. Skipping...")
                            continue
                    
                    if not batch_images:
                        continue
                    
                    # Compute Visual Search Time difficulty for batch
                    for i, (image, filepath) in enumerate(zip(batch_images, batch_filepaths)):
                        try:
                            # Use the visual_search_score function to compute the score
                            visual_search_score_value = visual_search_score(image, device)
                            
                            # # Compute other difficulty metrics  
                            # odd_score = np.random.random()    
                            # cnn_score = np.random.random()    
                            # zigzag_score = np.random.random() 
                            # 
                            # # Compute final score (weighted average)
                            # final_score = 0.25 * visual_search_score + 0.25 * odd_score + 0.25 * cnn_score + 0.25 * zigzag_score
                            # 
                            # # Store metrics
                            # metrics = {
                            #     "visual": float(visual_search_score),
                            #     "odd": float(odd_score),
                            #     "cnn": float(cnn_score),
                            #     "zigzag": float(zigzag_score),
                            #     "final_score": float(final_score)
                            # }
                            
                            # Update sample with score
                            sample = dataset[filepath]
                            sample[output_field] = float(visual_search_score_value)
                            
                            # # Update sample with all metrics  
                            # sample[metrics_field] = metrics
                            
                            # # Generate recommendation based on score  
                            # if visual_search_score < 3.0:
                            #     recommendation = "Easy: Ready for training"
                            # elif visual_search_score < 4.0:
                            #     recommendation = "Medium: Consider augmentations or super-resolution"
                            # else:
                            #     recommendation = "Hard: Suggest manual labeling or curriculum learning"
                            # 
                            # sample[recommendation_field] = recommendation
                            sample.save()
                            
                            # Store score for visualization
                            all_scores.append(visual_search_score_value)
                            
                            # # Store metrics for visualization  
                            # all_metrics["visual"].append(visual_search_score)
                            # all_metrics["odd"].append(odd_score)
                            # all_metrics["cnn"].append(cnn_score)
                            # all_metrics["zigzag"].append(zigzag_score)
                            # all_metrics["final_score"].append(final_score)
                            
                        except Exception as e:
                            logger.error(f"Error processing sample {filepath}: {str(e)}")
                            continue
                    
                    # Clean up memory
                    clean_memory()
            
            # Generate visualizations
            visualizations = {}
            
            if show_histograms:
                # Create histogram of difficulty scores
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(all_scores, bins=20, alpha=0.7)
                ax.set_title('Distribution of Visual Search Time Difficulty Scores')
                ax.set_xlabel('Score')
                ax.set_ylabel('Count')
                visualizations["difficulty_histogram"] = plot_to_base64(fig)
                
                # # Create histograms for each metric  
                # for metric_name, values in all_metrics.items():
                #     fig, ax = plt.subplots(figsize=(10, 6))
                #     ax.hist(values, bins=20, alpha=0.7)
                #     ax.set_title(f'Distribution of {metric_name} Scores')
                #     ax.set_xlabel('Score')
                #     ax.set_ylabel('Count')
                #     visualizations[f"{metric_name}_histogram"] = plot_to_base64(fig)
            
            # # Create scatter plots for metric pairs  
            # if show_scatter_plots:
            #     metric_pairs = [
            #         ("visual", "odd"),
            #         ("visual", "cnn"),
            #         ("visual", "zigzag"),
            #         ("odd", "cnn"),
            #         ("odd", "zigzag"),
            #         ("cnn", "zigzag")
            #     ]
            #     
            #     for metric1, metric2 in metric_pairs:
            #         fig, ax = plt.subplots(figsize=(10, 6))
            #         ax.scatter(all_metrics[metric1], all_metrics[metric2], alpha=0.5)
            #         ax.set_title(f'{metric1} vs {metric2} Scores')
            #         ax.set_xlabel(metric1)
            #         ax.set_ylabel(metric2)
            #         visualizations[f"{metric1}_{metric2}_scatter"] = plot_to_base64(fig)
            
            # Compute summary statistics
            summary_stats = {
                "mean": float(np.mean(all_scores)),
                "median": float(np.median(all_scores)),
                "std": float(np.std(all_scores)),
                "min": float(np.min(all_scores)),
                "max": float(np.max(all_scores))
            }
            
            # # Compute summary statistics for multiple metrics  
            # summary_stats = {}
            # for metric_name, values in all_metrics.items():
            #     summary_stats[metric_name] = {
            #         "mean": float(np.mean(values)),
            #         "median": float(np.median(values)),
            #         "std": float(np.std(values)),
            #         "min": float(np.min(values)),
            #         "max": float(np.max(values))
            #     }
            
            # # Generate overall recommendations  
            # mean_score = summary_stats["mean"]
            # if mean_score < 3.0:
            #     overall_recommendation = "Dataset is generally easy. Consider using standard training approaches."
            # elif mean_score < 4.0:
            #     overall_recommendation = "Dataset has moderate difficulty. Consider using data augmentation and curriculum learning."
            # else:
            #     overall_recommendation = "Dataset is challenging. Consider manual labeling for difficult samples and specialized model architectures."
            
            logger.info("Dataset difficulty scoring completed successfully")
            
            # Return results
            return {
                "success": True,
                "summary_stats": summary_stats,
                # "overall_recommendation": overall_recommendation,  # Commented for future use
                "visualizations": visualizations
            }
            
        except Exception as e:
            logger.error(f"Error during dataset difficulty scoring: {str(e)}")
            raise
        
        finally:
            # Clean up
            clean_memory()
            ctx.ops.reload_dataset()

def register(plugin):
    plugin.register(DatasetDifficultyScoring) 