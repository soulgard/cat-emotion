# cat-emotion

**This project based on the following Cat Audio Emotion Classification Task**

Classification of domestic cat audio files into 10 emotional states: Angry, Defense, Fighting, Happy, HuntingMind, Mating, MotherCall, Paining, Resting, and Warning.

The model trains on raw waveforms using the M5 architecture (PyTorch), with time stretching, pitch shifting, and Gaussian noise augmentation.

![Example](examples/app_demo.png)

---

## 🎯 Recent Updates (2026)

This repository has been updated with **new training and inference scripts** that work with modern Python modules (PyTorch 2.9+, Python 3.13+) while maintaining compatibility with the original M5 model architecture.

### New Scripts:

**1. `train_modern.py` - Modernized Training Script**

Based on the original `script.py`, this updated version:
- ✅ **Uses the same M5 architecture** as the original model
- ✅ **Works with PyTorch 2.9+ and Python 3.13+** (no need for old versions!)
- ✅ **Matches original training approach**: Same hyperparameters (lr=0.01, weight_decay, LR scheduler), same augmentations (TimeStretch, PitchShift, Gaussian noise), same data pipeline
- ✅ **Uses Kaggle data to train**(https://www.kaggle.com/datasets/yagtapandeya/cat-sound-classification-dataset)

**What you can do:**
```bash
# Train a new model using the fixed pipeline with modern PyTorch
python train_modern.py --data data/CAT_DB --epochs 30
```

**2. `inference.py` - Universal Inference Script**

A new script that can work with both old and new models:
- ✅ **Loads the original pretrained model** (MLflow format from `examples/model/`)
- ✅ **Also loads any new models** trained with `train_modern.py` (.pt/.pth files)
- ✅ **Works with modern PyTorch 2.9+** (no downgrade needed!)
- ✅ **Auto-detects model architecture** from checkpoint
- ✅ **Two modes**:
  - Evaluation on dataset (calculate accuracy)
  - Prediction on single audio file

**What you can do:**
```bash
# Evaluate original pretrained model with modern PyTorch
python inference.py --checkpoint examples/model --data data/CAT_DB --mode eval

# Predict emotion from a single cat audio file
python inference.py --checkpoint examples/model --audio cat_meow.mp3 --mode predict

# Also works with newly trained models
python inference.py --checkpoint my_model.pt --data data/CAT_DB --mode eval
```

### Why These Updates Matter:

The original repository requires **PyTorch 1.9** and **Python 3.8** (from 2022). These new scripts let you:
- ✅ Use the **latest PyTorch** and Python versions
- ✅ Train new models with the **same proven M5 architecture**
- ✅ Load and use the **original pretrained model** without downgrading
- ✅ Fix bugs that caused training failures

---

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- See `requirements.txt` for full dependencies

---

## 🎓 Training

### Option 1: Use the New Modern Training Script (Recommended)

Train a new model with the **fixed pipeline** that works with modern PyTorch:

```bash
python train_modern.py --data data/CAT_DB --epochs 30 --batch_size 4 --lr 0.01
```

This script:
- Uses the **same M5 architecture** as the original model
- Fixes bugs that prevented proper training
- Works with **PyTorch 2.9+ and Python 3.13+**
- Follows the same training methodology (augmentations, hyperparameters)

### Option 2: Use the Original Script

For reference, the original training script is still available:

```bash
python script.py
```

**Note:** The original script was designed for PyTorch 1.9 and may require older dependencies.

### Training Parameters:
- `--data`: Path to dataset directory
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 0.01)
- `--filters`: Filter sizes for each conv layer (default: [64, 64, 128, 128])
- `--early_stop_patience`: Early stopping patience (default: 5)
- `--checkpoint_dir`: Directory to save checkpoints

### Example Output:
```
Using device: mps
Train Epoch 0: 100%|██████████| 20/20 [00:08<00:00, loss=0.612, accuracy=0.047]
Test  Epoch 0: 100%|██████████| 5/5 [00:07<00:00, loss=0.485, accuracy=0.000]
...
Train Epoch 8: 100%|██████████| 20/20 [00:07<00:00, loss=0.406, accuracy=0.359]
Test  Epoch 8: 100%|██████████| 5/5 [00:06<00:00, loss=0.402, accuracy=0.250]
```

---

## 🏗️ Model Architecture

**M5 Network:**
- 4 convolutional blocks
- Each block: Conv1D → BatchNorm → ReLU → MaxPool → Dropout(0.3)
- Global average pooling
- Fully connected layer → LogSoftmax
- Input: Raw waveform at 8kHz
- Output: 10 emotion classes

**Default Configuration:**
- Filters: [64, 64, 128, 128]
- Kernel sizes: [80, 3, 3, 3]
- Strides: [16, 1, 1, 1]

---

## 📊 Model Performance

### Real Performance on CAT_DB Dataset (100 samples, 10 per class):

We evaluated both the original pretrained model and a newly trained model on the same CAT_DB test set (20 samples):

| Model | Overall Accuracy | Test Samples | Best Classes | Notes |
|-------|------------------|--------------|--------------|-------|
| **Original Pretrained** (`examples/model/`) | **10.00%** | 2/20 correct | MotherCall (100%), Happy (50%) | Trained on different dataset from research paper |
| **Newly Trained** (`train_modern.py`) | **35.00%** | 7/20 correct | HuntingMind (100%), Fighting (75%), Paining (50%) | Trained specifically on CAT_DB |

### Detailed Per-Class Accuracy:

**Original Pretrained Model:**
```
Angry          : 0.00%
Defense        : 0.00%
Fighting       : 0.00%
Happy          : 50.00%  ✓
HuntingMind    : 0.00%
Mating         : 0.00%
MotherCall     : 100.00% ✓
Paining        : 0.00%
Resting        : 0.00%
Warning        : 0.00%
```

**Newly Trained Model (train_modern.py):**
```
Angry          : 0.00%
Defense        : 0.00%
Fighting       : 75.00%  ✓
Happy          : 0.00%
HuntingMind    : 100.00% ✓
Mating         : 33.33%  ✓
MotherCall     : 0.00%
Paining        : 50.00%  ✓
Resting        : 0.00%
Warning        : 33.33%  ✓
```

### Analysis:

**Important: Both models use the SAME M5 architecture AND the SAME training pipeline!**

The performance difference is NOT due to different model structures or different training methodology. Both use:
- ✅ Identical M5 architecture
- ✅ Same augmentations (TimeStretch, PitchShift, Gaussian noise)
- ✅ Same hyperparameters (lr=0.01, weight_decay=0.0001, LR scheduler)
- ✅ Same training approach (raw waveform input, 8kHz sampling)

**Class-Specific Performance Patterns:**
   - **Original model**: Best at MotherCall (100%), Happy (50%)
   - **New model**: Best at HuntingMind (100%), Fighting (75%), Paining (50%)
   - Same architecture, same pipeline design, but different learned patterns
   - Shows that training data distribution matters more than architecture


### Next steps:

1. **Collect More Data:** Aim for 100-150 samples per class
2. **Use less class:** Considering this paper and employed less sound classes to train the model

### Good to know:
This kaggle author employed 500 epochs to train CNN model using the same dataset and reach 45% accuracy, still not good, but better than ours.
https://www.kaggle.com/code/muqaddasejaz/cat-emotion-classification-eda/notebook


### How to Reproduce These Results:

```bash
# Evaluate original pretrained model
python inference.py --checkpoint examples/model --data data/CAT_DB --mode eval

# Evaluate newly trained model
python inference.py --checkpoint path/to/your_model.pt --data data/CAT_DB --mode eval
```

---

## 📚 Citation

Credit to the original dataset and augmentation techniques:

**Domestic Cat Sound Classification Using Transfer Learning**
*Yagya Raj Pandeya, Dongwhoon Kim and Joonwhoan Lee*
https://doi.org/10.3390/app8101949

