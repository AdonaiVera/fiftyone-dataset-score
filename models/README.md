# Visual Search Time Difficulty Model Training

This directory contains scripts for training and evaluating the CLIP + Linear Regressor model for predicting Visual Search Time difficulty scores.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Prepare your dataset:
   - Place your CSV file containing image paths and difficulty scores in the `data` directory
   - Ensure all images referenced in the CSV are accessible
   - The CSV should have columns: `image_path` and `difficulty_score`

## Training

To train the model with default parameters:
```bash
python train.py
```

### Command Line Arguments

- `--csv_path`: Path to the CSV file containing image paths and difficulty scores (default: "models/data/voc2012_difficulty_scores.csv")
- `--image_dir`: Directory containing the images (default: "models/data/VOC2012/JPEGImages")
- `--clip_model`: CLIP model to use (default: "ViT-B/32")
- `--batch_size`: Batch size for training (default: 32)
- `--learning_rate`: Learning rate for the optimizer (default: 1e-4)
- `--epochs`: Number of training epochs (default: 50)
- `--output_dir`: Directory to save model outputs (default: "models/results")
- `--model_save_dir`: Directory to save model checkpoints (default: "models/saved_models")
- `--train_split`: Proportion of data to use for training (default: 0.8)
- `--val_split`: Proportion of data to use for validation (default: 0.1)
- `--test_split`: Proportion of data to use for testing (default: 0.1)
- `--seed`: Random seed for reproducibility (default: 42)

Example with custom parameters:
```bash
python train.py --csv_path data/my_scores.csv --image_dir data/my_images --clip_model ViT-B/16 --batch_size 64 --learning_rate 1e-5 --epochs 100 --output_dir outputs
```

## Output

The training script generates the following outputs in the specified output directory:

1. `training_history.png`: Plot of training metrics over time
2. `predictions.png`: Scatter plot of predicted vs. actual difficulty scores
3. `results.json`: JSON file containing training results and parameters

The trained model is saved in the model save directory:

1. `model.pt`: The trained model weights

## Model Architecture

The model consists of two main components:
1. CLIP image encoder (ViT-B/32 by default)
2. Linear regression head

The CLIP encoder is frozen during training, and only the linear regression head is trained to predict difficulty scores.

## Evaluation

The model is evaluated using:
- Mean Squared Error (MSE)
- Kendall's Tau correlation coefficient

These metrics are logged during training and plotted in the output visualizations. 