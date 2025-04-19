import os
os.environ['FIFTYONE_ALLOW_LEGACY_ORCHESTRATORS'] = 'true'


import fiftyone.operators as foo
from fiftyone.operators import types
import torch
import logging
from PIL import Image
import numpy as np

# Import utility functions
from .methods.utils import (
    _execution_mode,
    batch_generator,
    compute_summary_stats,
    DEFAULT_BATCH_SIZE
)
from .methods.visual_search_score import visual_search_score, load_models, clean_memory as clean_vs_memory
from .methods.odd_score import odd_score, clean_memory as clean_odd_memory

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='dataset_difficulty.log'
)
logger = logging.getLogger("dataset_difficulty_operator")

def extract_boxes_and_classes(detections):
    """
    Extract bounding boxes and classes from FiftyOne detections
    
    Args:
        detections: FiftyOne detections field
        
    Returns:
        List of [x1, y1, x2, y2, class] for each detection
    """
    boxes = []
    for det in detections:
        # Get bounding box coordinates
        bbox = det.bounding_box  # [x, y, w, h] format
        x1 = bbox[0]
        y1 = bbox[1]
        x2 = x1 + bbox[2]  # x + w
        y2 = y1 + bbox[3]  # y + h
        
        # Get class label
        class_label = det.label
        
        boxes.append([x1, y1, x2, y2, class_label])
    
    return boxes

class DatasetDifficultyScoring(foo.Operator):
    @property
    def config(self):
        return foo.OperatorConfig(
            name="dataset_difficulty_scoring",
            label="Dataset Difficulty Scoring",
            description="Score dataset samples based on Visual Search Time and Object Detection Difficulty metrics",
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

        # Add difficulty metrics selection  
        metrics_radio = types.RadioGroup()
        metrics_radio.add_choice("all", label="All Metrics")
        metrics_radio.add_choice("visual", label="Visual Search Time")
        metrics_radio.add_choice("odd", label="Object Detection Difficulty")
        
        inputs.enum(
            "difficulty_metrics",
            metrics_radio.values(),
            label="Difficulty Metrics",
            description="Select which difficulty metrics to compute",
            default="all"
        )

        # Add output fields for individual metrics
        inputs.str(
            "visual_score_field",
            default="visual_search_score",
            label="Visual Search Score Field",
            description="Field to store the visual search difficulty score"
        )
        
        inputs.str(
            "odd_score_field",
            default="odd_score",
            label="Object Detection Difficulty Score Field",
            description="Field to store the object detection difficulty score"
        )

        # Add final score field
        inputs.str(
            "final_score_field",
            default="difficulty_score",
            label="Final Difficulty Score Field",
            description="Field to store the final combined difficulty score"
        )

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
        difficulty_metrics = ctx.params.get("difficulty_metrics", "all")
        visual_score_field = ctx.params.get("visual_score_field", "visual_search_score")
        odd_score_field = ctx.params.get("odd_score_field", "odd_score")
        final_score_field = ctx.params.get("final_score_field", "difficulty_score")
        batch_size = ctx.params.get("batch_size", DEFAULT_BATCH_SIZE)
        
        logger.info(f"Parameters: dataset_percentage={dataset_percentage}, "
                   f"difficulty_metrics={difficulty_metrics}")

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
            
            # Initialize metrics storage for multiple metrics  
            all_metrics = {
                "visual": [],
                "odd": [],
                "final_score": []
            }
            
            # Process dataset in batches
            with torch.no_grad():  # Disable gradient computation
                for batch in batch_generator(dataset, batch_size):
                    batch_images = []
                    batch_filepaths = []
                    batch_ground_truths = []
                    
                    # Prepare batch data
                    for sample in batch:
                        try:
                            # Load image for visual search score and ODD score
                            image = Image.open(sample.filepath)
                            batch_images.append(image)
                            batch_filepaths.append(sample.filepath)
                            
                            # Extract ground truth boxes
                            if hasattr(sample, 'ground_truth'):
                                gt_boxes = extract_boxes_and_classes(sample.ground_truth.detections)
                                batch_ground_truths.append(gt_boxes)
                            else:
                                logger.warning(f"Sample {sample.filepath} missing ground truth. Using empty boxes.")
                                batch_ground_truths.append([])
                                
                        except (FileNotFoundError, OSError) as e:
                            logger.warning(f"Could not open image {sample.filepath}: {str(e)}. Skipping...")
                            continue
                    
                    if not batch_images:
                        continue
                    
                    # Compute difficulty metrics for batch
                    for i, (image, filepath, ground_truths) in enumerate(zip(batch_images, batch_filepaths, batch_ground_truths)):
                        try:
                            # Initialize metrics dictionary
                            metrics = {}
                            
                            # Calculate scores based on selected metrics
                            if difficulty_metrics == "visual":
                                score = visual_search_score(image, device)
                                metrics["visual"] = float(score)
                                final_score = score
                                
                                # Update dataset sample with individual fields
                                sample = dataset[filepath]
                                sample[visual_score_field] = float(score)
                                sample[final_score_field] = float(final_score)
                                sample.save()
                                
                            elif difficulty_metrics == "odd":
                                score = odd_score(image, ground_truths)
                                metrics["odd"] = float(score)
                                final_score = score
                                
                                # Update dataset sample with individual fields
                                sample = dataset[filepath]
                                sample[odd_score_field] = float(score)
                                sample[final_score_field] = float(final_score)
                                sample.save()
                                
                            else:  # Combined metrics
                                vs_score = visual_search_score(image, device)
                                odd_score_val = odd_score(image, ground_truths)
                                metrics = {
                                    "visual": float(vs_score),
                                    "odd": float(odd_score_val)
                                }
                                final_score = 0.5 * vs_score + 0.5 * odd_score_val
                                
                                # Update dataset sample with individual fields
                                sample = dataset[filepath]
                                sample[visual_score_field] = float(vs_score)
                                sample[odd_score_field] = float(odd_score_val)
                                sample[final_score_field] = float(final_score)
                                sample.save()
                            
                            # Store scores for summary statistics
                            all_scores.append(final_score)
                            for metric, value in metrics.items():
                                all_metrics[metric].append(value)
                            all_metrics["final_score"].append(final_score)
                            
                        except Exception as e:
                            logger.error(f"Error processing sample {filepath}: {str(e)}")
                            continue
                    
                    # Clean up memory
                    clean_vs_memory()
                    clean_odd_memory()
            
            # Compute summary statistics
            summary_stats = compute_summary_stats(all_scores)
            
            logger.info("Dataset difficulty scoring completed successfully")
            
            # Return results
            return {
                "success": True,
                "summary_stats": summary_stats
            }
            
        except Exception as e:
            logger.error(f"Error during dataset difficulty scoring: {str(e)}")
            raise
        
        finally:
            # Clean up
            clean_vs_memory()
            clean_odd_memory()
            ctx.ops.reload_dataset()

def register(plugin):
    plugin.register(DatasetDifficultyScoring) 