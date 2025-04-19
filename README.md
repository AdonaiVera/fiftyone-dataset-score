# 🌍 FiftyOne Dataset Difficulty Scoring
A powerful toolkit for analyzing and scoring the difficulty of object detection datasets in FiftyOne

https://github.com/user-attachments/assets/6cea6d60-b54f-41f4-976d-6b412426cecd

## 🎯 Project Vision

In the field of computer vision and object detection, understanding the difficulty of your dataset is crucial for effective model training and evaluation. This plugin provides a comprehensive difficulty metric based on state-of-the-art research to help you understand your dataset's characteristics and make informed decisions about training strategies.

Our goal is to make advanced dataset analysis techniques accessible to everyone - from individual developers and hobbyists to large companies. We provide easy-to-use tools that help you understand your dataset's difficulty profile without requiring deep expertise in the field.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PBxHTrxoq34gErlRtko3uDaodm5Yf4kE?usp=sharing)

## 🚀 Key Features

### Dataset Difficulty Scoring

- **Visual Search Time (Perceptual Difficulty)** - Comprehensive analysis using research-backed difficulty measure:
  - Based on [Ionescu et al. 2016](https://arxiv.org/pdf/1705.08280)
  - Measures how difficult an image is for a human to label
  - Simulates human visual search behavior
  - Uses CLIP (Contrastive Language-Image Pre-training) for feature extraction
  - Trained regressor model to predict visual search time scores

- **Interactive Visualizations** - Generate histograms to understand the distribution of difficulty scores
- **Data-Driven Recommendations** - Get actionable insights based on your dataset's difficulty profile
- **Sample-Level Analysis** - Identify difficult samples that may need special attention during training
- **Dataset-Level Insights** - Understand overall dataset characteristics to inform training strategies

## 🛠️ Installation

If you haven't already, install FiftyOne:

```bash
pip install -U fiftyone transformers<=4.49 accelerate einops timm torch ultralytics
```

Then, install the plugin:

```bash
fiftyone plugins download https://github.com/AdonaiVera/fiftyone-dataset-score
```

## 📚 Research-Backed Technique

### Visual Search Time (Perceptual Difficulty)
- **Implementation**: CLIP + Linear Regressor approach
   - CLIP extracts image features (frozen)
   - Linear regression head predicts difficulty
   - Trained on PASCAL VOC 2012 with crowdsourced response times
   - Measures how difficult an image is for a human to label
   - Based on [Ionescu et al. 2016](https://arxiv.org/pdf/1705.08280)
   - Simulates human visual search behavior

### Individual Sample Scores (Object Detection Difficulty - ODD)
- **Implementation**: Advanced scoring system based on [Zhang et al. 2023](https://arxiv.org/pdf/2308.11327)
   - **Core Components**:
     - FastSAM Object Detection: Runs FastSAM to predict object bbox
     - IoU-based Classification: Categorizes predictions into P/M/N/NG based on IoU thresholds
     - Weighted Metrics: Computes precision and recall with category-specific weights
     - Harmonic Mean Scoring: Combines metrics into final ODD score
   
   - **Scoring Mechanism**:
     - Categorizes predictions into four classes based on IoU:
       - Positive (P): IoU > 0.8, weight = 1.0
       - Multi-positive (M): 0.5 < IoU ≤ 0.8, weight = 0.5
       - Near-positive (N): 0.3 < IoU ≤ 0.5, weight = 0.3
       - Negative (NG): IoU ≤ 0.3, weight = 0.0
     - Computes weighted precision and recall
     - Final score = 1 - harmonic_mean(Wpr, Wrc)

   - **Applications**:
     - Ranks images by detection difficulty
     - Identifies challenging samples for model training
     - Helps balance dataset difficulty distribution
     - Guides data augmentation strategies
     - Supports quality control in dataset curation

## 🧠 Visual Search Time Difficulty Scoring

### Dataset Foundation
The Visual Search Time difficulty metric is based on the PASCAL VOC 2012 dataset (11,540 samples), which has been annotated with human difficulty scores through a carefully designed crowdsourcing process.

### How Visual Search Time Scores Are Collected
1. **Crowdsourced Visual Search Task**:
   - Workers are shown an image and asked: "Is there a {class} in this image?"
   - Process: Ready → image appears → decision made → Done
   - Response time is recorded in seconds

2. **Two Questions Per Image**:
   - One positive and one negative question to prevent bias
   - Ensures balanced assessment of difficulty

3. **Post-processing**:
   - Outlier removal (e.g., times > 20 seconds)
   - Normalization per annotator
   - Final score = geometric mean of response times per image
   - Continuous values (typically 2.5 to 5.0)

### Implementation Approach
Our implementation uses a modern, scalable approach:

1. **CLIP + Linear Regressor**:
   - CLIP (ViT-B/32) extracts image features
   - Pretrained on image-text data (frozen)
   - Small regression head (nn.Linear) on top of CLIP embeddings
   - Only the linear layer is trained

2. **Training Process**:
   - Input: image paths from VOC2012
   - Target: difficulty scores from Ionescu et al.
   - Loss: Mean Squared Error (MSE) or Huber Loss
   - Evaluation: MSE and Kendall's τ for ranking performance

This approach provides a fast, interpretable baseline that leverages strong semantic priors from CLIP, matching the structure of the original paper but with modern architecture.


## 🔮 Future Work
We plan to implement additional difficulty metrics in future releases:

1. **CNN Difficulty Predictor**
   - Lightweight CNN+SVR to predict detection difficulty
   - Based on [Soviany et al. 2018](https://arxiv.org/pdf/1803.08707)
   - Separates easy from hard images for detector selection

2. **Zigzag Learning (Localization Difficulty)**
   - Quantifies how dispersed/confused the detector is about object location
   - Based on [Zhang et al. 2018](https://arxiv.org/pdf/1804.09466)
   - Uses energy accumulation from region proposals

## 📊 Usage Examples

### Basic Usage

```python
import fiftyone as fo

# Load your dataset
dataset = fo.load_dataset("your_dataset_name")

# Launch the FiftyOne app
session = fo.launch_app(dataset)

# Use the Dataset Difficulty Scoring operator from the FiftyOne UI
# 1. Select your dataset
# 2. Click on the "Dataset Difficulty Scoring" operator
# 3. Configure parameters (dataset percentage, visualization options)
# 4. Run the operator
```

## 📈 Interpreting Results

The Dataset Difficulty Scoring operator provides several types of insights:

1. **Individual Sample Scores**
   - Each sample in your dataset receives a difficulty score
   - Scores are stored in the `difficulty_score` field
   - Recommendations are stored in the `difficulty_recommendation` field

2. **Visualizations**
   - Histograms show the distribution of difficulty scores
   - These help identify patterns and outliers in your dataset

3. **Recommendations**
   - Based on the overall difficulty profile of your dataset
   - Suggests appropriate training strategies
   - Identifies potential areas for improvement

## 🤝 Contributing

We welcome contributions! If you're working on dataset difficulty analysis and have techniques that could improve our plugin, please reach out.

## 📚 Citation

```bibtex
@inproceedings{ionescu2016hard,
  title={How hard can it be? Estimating the difficulty of visual search in an image},
  author={Ionescu, Radu Tudor and Alexe, Bogdan and Leordeanu, Marius and Popescu, Marius and Papadopoulos, Dimitrios and Ferrari, Vittorio},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2157--2166},
  year={2016}
}
```

```bibtex
@inproceedings{zhang2023object,
  title={Object Detection Difficulty: Suppressing Over-aggregation for Faster and Better Video Object Detection},
  author={Zhang, Bingqing and Wang, Sen and Liu, Yifan and Kusy, Brano and Li, Xue and Liu, Jiajun},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={1768--1778},
  year={2023}
}
```

## 🙏 Acknowledgments

- OpenAI for the CLIP model
- Hugging Face for hosting the regressor model
- FiftyOne team for the amazing dataset visualization platform 
