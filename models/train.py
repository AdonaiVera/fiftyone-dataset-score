#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Train a CLIP + Linear Regressor model to predict Visual Search Time difficulty scores.
Based on the approach described in Ionescu et al. 2016.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import clip
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
import json
from pathlib import Path

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("models/training.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("visual_search_time_training")

# Constants
DEFAULT_CLIP_MODEL = "ViT-B/32"
DEFAULT_BATCH_SIZE = 32
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_EPOCHS = 50
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_VAL_SPLIT = 0.1
DEFAULT_TEST_SPLIT = 0.1
DEFAULT_IMAGE_SIZE = 224
DEFAULT_CSV_PATH = "models/data/voc2012_difficulty_scores.csv"
DEFAULT_IMAGES_DIR = "models/data/images"
DEFAULT_MODEL_SAVE_DIR = "models/saved_models"
DEFAULT_RESULTS_DIR = "models/results"

class DifficultyDataset(Dataset):
    def __init__(self, csv_path, image_dir, clip_model, transform=None):
        self.data = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.clip_model = clip_model
        self.transform = transform
        
        # Preprocess all images with CLIP
        logger.info("Preprocessing images with CLIP...")
        self.features = []
        self.scores = []
        
        for idx, row in tqdm(self.data.iterrows(), total=len(self.data)):
            image_path = os.path.join(image_dir, row['image_path'] + '.jpg')
            try:
                image = Image.open(image_path).convert('RGB')
                if self.transform:
                    image = self.transform(image)
                
                # Get CLIP features
                with torch.no_grad():
                    features = self.clip_model.encode_image(image.unsqueeze(0))
                
                self.features.append(features.squeeze().cpu().numpy())
                self.scores.append(row['difficulty_score'])
            except Exception as e:
                logger.warning(f"Error processing image {image_path}: {str(e)}")
                continue
        
        self.features = np.array(self.features)
        self.scores = np.array(self.scores)
        
        logger.info(f"Processed {len(self.features)} images successfully")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.scores[idx]])

class DifficultyPredictor(nn.Module):
    def __init__(self, input_dim):
        super(DifficultyPredictor, self).__init__()
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    val_losses = []
    val_kendall_taus = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        for features, scores in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            features, scores = features.to(device), scores.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, scores)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0
        all_preds = []
        all_scores = []
        
        with torch.no_grad():
            for features, scores in val_loader:
                features, scores = features.to(device), scores.to(device)
                outputs = model(features)
                loss = criterion(outputs, scores)
                val_loss += loss.item()
                
                all_preds.extend(outputs.cpu().numpy())
                all_scores.extend(scores.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calculate Kendall's Tau
        kendall_tau, _ = kendalltau(all_preds, all_scores)
        val_kendall_taus.append(kendall_tau)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val Kendall's Tau: {kendall_tau:.4f}")
    
    return train_losses, val_losses, val_kendall_taus

def plot_training_history(train_losses, val_losses, val_kendall_taus, output_dir):
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot Kendall's Tau
    plt.subplot(1, 2, 2)
    plt.plot(val_kendall_taus, label="Kendall's Tau")
    plt.xlabel('Epoch')
    plt.ylabel("Kendall's Tau")
    plt.title("Validation Kendall's Tau")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close()

def plot_predictions(model, val_loader, device, output_dir):
    model.eval()
    all_preds = []
    all_scores = []
    
    with torch.no_grad():
        for features, scores in val_loader:
            features = features.to(device)
            outputs = model(features)
            all_preds.extend(outputs.cpu().numpy())
            all_scores.extend(scores.numpy())
    
    plt.figure(figsize=(8, 8))
    plt.scatter(all_scores, all_preds, alpha=0.5)
    plt.plot([min(all_scores), max(all_scores)], [min(all_scores), max(all_scores)], 'r--')
    plt.xlabel('Actual Difficulty Score')
    plt.ylabel('Predicted Difficulty Score')
    plt.title('Predicted vs. Actual Difficulty Scores')
    plt.savefig(os.path.join(output_dir, 'predictions.png'))
    plt.close()

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Train a CLIP + Linear Regressor model for Visual Search Time difficulty prediction')
    
    # Data arguments
    parser.add_argument('--csv_path', type=str, default=DEFAULT_CSV_PATH, help='Path to the CSV file with difficulty scores')
    parser.add_argument('--image_dir', type=str, default=DEFAULT_IMAGES_DIR, help='Directory containing the images')
    
    # Model arguments
    parser.add_argument('--clip_model', type=str, default=DEFAULT_CLIP_MODEL, help='CLIP model to use')
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=DEFAULT_LEARNING_RATE, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of epochs to train for')
    
    # Split arguments
    parser.add_argument('--train_split', type=float, default=DEFAULT_TRAIN_SPLIT, help='Proportion of data to use for training')
    parser.add_argument('--val_split', type=float, default=DEFAULT_VAL_SPLIT, help='Proportion of data to use for validation')
    parser.add_argument('--test_split', type=float, default=DEFAULT_TEST_SPLIT, help='Proportion of data to use for testing')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default=DEFAULT_RESULTS_DIR, help='Directory to save model outputs')
    parser.add_argument('--model_save_dir', type=str, default=DEFAULT_MODEL_SAVE_DIR, help='Directory to save model checkpoints')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create directories
    os.makedirs(args.model_save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load CLIP model
    logger.info(f"Loading CLIP model: {args.clip_model}")
    clip_model, preprocess = clip.load(args.clip_model, device=device)
    
    # Create dataset
    logger.info(f"Creating dataset from {args.csv_path}")
    dataset = DifficultyDataset(args.csv_path, args.image_dir, clip_model, preprocess)
    
    # Split dataset
    train_size = int(args.train_split * len(dataset))
    val_size = int(args.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    logger.info(f"Dataset split: {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # Create model
    input_dim = dataset.features.shape[1]
    model = DifficultyPredictor(input_dim).to(device)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Train model
    logger.info("Starting training")
    train_losses, val_losses, val_kendall_taus = train_model(
        model, train_loader, val_loader, criterion, optimizer, args.epochs, device
    )
    
    # Plot training history
    logger.info("Plotting training history")
    plot_training_history(train_losses, val_losses, val_kendall_taus, args.output_dir)
    
    # Plot predictions
    logger.info("Plotting predictions")
    plot_predictions(model, val_loader, device, args.output_dir)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(args.model_save_dir, 'model.pt'))
    logger.info(f"Model saved to {os.path.join(args.model_save_dir, 'model.pt')}")
    
    # Save results
    logger.info("Saving results")
    results = {
        'args': vars(args),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'val_kendall_taus': val_kendall_taus
    }
    
    with open(os.path.join(args.output_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    logger.info("Done!")

if __name__ == "__main__":
    main() 


'''
python models/train.py --image_dir models/data/images --output_dir outputs --model_save_dir outputs/models
'''