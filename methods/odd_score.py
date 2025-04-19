import numpy as np
import logging
import torch
import torchvision
from PIL import Image
import torchvision.transforms as transforms
import cv2
from ultralytics import FastSAM
import os
import requests
from tqdm import tqdm

# Set up logging
logger = logging.getLogger("odd_score")

# Global variables for model and device
_model = None
_device = None

def download_file(url, filename):
    """
    Download a file with progress bar
    
    Args:
        url: URL to download from
        filename: Local filename to save to
    """
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            pbar.update(size)

def load_model():
    """
    Load the FastSAM model for object detection
    
    Returns:
        model: The loaded model
        device: The device the model is on
    """
    global _model, _device
    
    if _model is not None:
        return _model, _device
    
    # Set device
    _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Load FastSAM model
        _model = FastSAM('FastSAM-s.pt')
        _model.to(_device)
        
        logger.info("Successfully loaded FastSAM model")
    except Exception as e:
        logger.error(f"Error loading FastSAM model: {str(e)}")
        raise
    
    return _model, _device

def detect_objects(image):
    """
    Run FastSAM-based object detection on an image
    
    Args:
        image: PIL Image
        
    Returns:
        List of [x1, y1, x2, y2] for each detection
    """
    global _model, _device
    
    # Load model if not already loaded
    if _model is None:
        load_model()
    
    # Convert PIL Image to numpy array for processing
    image_np = np.array(image)
    
    # Process image with FastSAM
    results = _model(image_np)
    
    # Convert masks to bounding boxes
    boxes = []
    for result in results:
        if hasattr(result, 'masks') and result.masks is not None:
            for mask in result.masks.data:
                # Convert mask to binary
                binary_mask = (mask.cpu().numpy() > 0).astype(np.uint8)
                
                # Find contours in the binary mask
                contours, _ = cv2.findContours(
                    binary_mask, 
                    cv2.RETR_EXTERNAL, 
                    cv2.CHAIN_APPROX_SIMPLE
                )
                
                # Get the largest contour (main object)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)
                    
                    # Convert to [x1, y1, x2, y2] format
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    
                    # Normalize coordinates to [0, 1] range
                    height, width = image_np.shape[:2]
                    x1, x2 = x1 / width, x2 / width
                    y1, y2 = y1 / height, y2 / height
                    
                    boxes.append([x1, y1, x2, y2])
    
    return boxes

def calculate_iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU value
    """
    # Calculate intersection coordinates
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Calculate intersection area
    intersection_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou

def calculate_weighted_metrics(predictions, ground_truths):
    """
    Calculate weighted precision and recall based on IoU thresholds.
    For each ground truth, find the best matching prediction.
    
    Args:
        predictions: List of predictions, each with [x1, y1, x2, y2]
        ground_truths: List of ground truth boxes, each with [x1, y1, x2, y2]
        
    Returns:
        weighted_precision, weighted_recall
    """
    # Initialize counters for perfect (P), medium (M), near (N), and no good (NG) matches
    P_count = 0  # Perfect match (IoU > 0.8)
    M_count = 0  # Medium match (0.5 < IoU <= 0.8)
    N_count = 0  # Near match (0.3 < IoU <= 0.5)
    NG_count = 0  # No good match (IoU <= 0.3)
    
    # Track which predictions have been matched
    matched_predictions = [False] * len(predictions)
    
    # For each ground truth, find the best matching prediction
    for gt in ground_truths:
        best_iou = 0
        best_pred_idx = -1
        
        # Find the prediction with highest IoU for this ground truth
        for i, pred in enumerate(predictions):
            if not matched_predictions[i]:  # Only consider unmatched predictions
                iou = calculate_iou(pred, gt)
                if iou > best_iou:
                    best_iou = iou
                    best_pred_idx = i
        
        # Categorize the match based on IoU
        if best_iou > 0.8:
            P_count += 1
            if best_pred_idx >= 0:
                matched_predictions[best_pred_idx] = True
        elif 0.5 < best_iou <= 0.8:
            M_count += 1
            if best_pred_idx >= 0:
                matched_predictions[best_pred_idx] = True
        elif 0.3 < best_iou <= 0.5:
            N_count += 1
            if best_pred_idx >= 0:
                matched_predictions[best_pred_idx] = True
        else:
            NG_count += 1
    
    # Calculate total predictions and ground truths
    total_predictions = len(predictions)
    total_ground_truths = len(ground_truths)
    
    # Calculate weighted precision (how well predictions match ground truths)
    weighted_precision = (1.0 * P_count + 0.5 * M_count + 0.3 * N_count) / total_predictions if total_predictions > 0 else 0
    
    # Calculate weighted recall (how well ground truths are matched by predictions)
    weighted_recall = (1.0 * P_count + 0.5 * M_count + 0.3 * N_count) / total_ground_truths if total_ground_truths > 0 else 0
    
    # Log detailed matching information
    logger.info(f"Perfect matches (IoU > 0.8): {P_count}")
    logger.info(f"Medium matches (0.5 < IoU <= 0.8): {M_count}")
    logger.info(f"Near matches (0.3 < IoU <= 0.5): {N_count}")
    logger.info(f"No good matches (IoU <= 0.3): {NG_count}")
    logger.info(f"Total ground truths: {total_ground_truths}")
    logger.info(f"Total predictions: {total_predictions}")
    
    return weighted_precision, weighted_recall

def odd_score(image, ground_truths):
    """
    Compute ODD (Object Detection Difficulty) score for an image.
    
    Args:
        image: PIL Image
        device: torch device
        
    Returns:
        float: ODD score
    """
    global _model, _device
    
    logger.info("Starting ODD score calculation")
    
    try:
        # Ensure FastSAM model is loaded
        if _model is None:
            logger.info("FastSAM model not loaded, loading now")
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            load_model()
        
        # Run object detection
        logger.info("Running object detection with FastSAM model")
        objects = detect_objects(image)
        logger.info(f"Found {len(objects)} objects")
        
        if not objects:
            logger.warning("No objects detected, returning default score")
            return 0.5
        
        # Calculate weighted metrics
        logger.info("Calculating weighted metrics")
        weighted_precision, weighted_recall = calculate_weighted_metrics(objects, ground_truths)
        
        logger.info(f"Weighted precision: {weighted_precision:.4f}")
        logger.info(f"Weighted recall: {weighted_recall:.4f}")
        
        # Calculate final score
        score = (weighted_precision + weighted_recall) / 2
        logger.info(f"Final ODD score: {score:.4f}")
        
        return score
    
    except Exception as e:
        logger.error(f"Error in odd_score: {str(e)}")
        # Return a default score in case of error
        return 0.5

def clean_memory():
    """Clean up memory by removing the model"""
    global _model, _device
    _model = None
    _device = None 