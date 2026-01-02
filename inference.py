"""
Inference script for pretrained M5 cat emotion classifier.

Usage:
  # Evaluate on test set
  python inference.py --checkpoint modern_m5.pt --data data/CAT_DB --mode eval

  # Inference on single audio file
  python inference.py --checkpoint modern_m5.pt --audio path/to/audio.mp3 --mode predict
"""

import argparse
from pathlib import Path

import torch
import torchaudio
import numpy as np
from tqdm import tqdm

from models.m5 import M5
from preprocessing.pipeline import load_dataset


# Class names mapping (must match training order)
CLASS_NAMES = {
    0: "Angry",
    1: "Defense",
    2: "Fighting",
    3: "Happy",
    4: "HuntingMind",
    5: "Mating",
    6: "MotherCall",
    7: "Paining",
    8: "Resting",
    9: "Warning",
}


def detect_model_architecture(state_dict: dict) -> tuple:
    """Detect model architecture from state dict."""
    # Check for old architecture (features/classifier) vs new (model/fc)
    has_features = any(k.startswith("features.") for k in state_dict.keys())

    if has_features:
        # Old architecture - detect filter sizes
        filter_sizes = []
        for i in [0, 5, 10, 15]:  # Conv layer indices in old architecture
            key = f"features.{i}.weight"
            if key in state_dict:
                filter_sizes.append(state_dict[key].shape[0])

        return filter_sizes, "old"
    else:
        # New architecture - detect from model keys
        filter_sizes = []
        for i in [0, 5, 10, 15]:
            key = f"model.{i}.weight"
            if key in state_dict:
                filter_sizes.append(state_dict[key].shape[0])

        return filter_sizes, "new"


def convert_old_to_new_state_dict(old_state_dict: dict) -> dict:
    """Convert old M5 state dict (features/classifier) to new format (model/fc)."""
    new_state_dict = {}

    for key, value in old_state_dict.items():
        if key.startswith("features."):
            # features.X -> model.X
            new_key = key.replace("features.", "model.")
            new_state_dict[new_key] = value
        elif key.startswith("classifier.2."):
            # classifier.2 -> fc
            new_key = key.replace("classifier.2.", "fc.")
            new_state_dict[new_key] = value

    return new_state_dict


def load_pretrained_model(checkpoint_path: Path, num_classes: int = 10) -> torch.nn.Module:
    """Load a pretrained M5 model from checkpoint.

    Supports:
    - MLflow format (directory with MLmodel file)
    - PyTorch .pth/.pt files (direct model or state_dict)
    """
    # Check if it's an MLflow model directory
    if checkpoint_path.is_dir() and (checkpoint_path / "MLmodel").exists():
        print(f"Loading MLflow model from {checkpoint_path}...")
        try:
            import mlflow.pytorch
            model = mlflow.pytorch.load_model(str(checkpoint_path), map_location="cpu")
            print("MLflow model loaded successfully!")
            model.eval()
            return model
        except ImportError:
            print("Warning: mlflow not installed. Install with: pip install mlflow[extras]")
            raise

    # Otherwise, load as PyTorch checkpoint
    print(f"Loading PyTorch checkpoint from {checkpoint_path}...")

    # Try loading as pickled model first (for compatibility with old saves)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # If it's already a model instance, return it directly
        if isinstance(checkpoint, torch.nn.Module):
            print("Loaded model instance directly")
            checkpoint.eval()
            return checkpoint
    except Exception as e:
        print(f"Could not load as pickled model: {e}")
        # Fall through to state_dict loading
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Detect architecture from state dict
    filter_sizes, arch_type = detect_model_architecture(state_dict)

    print(f"Detected architecture: {arch_type}")
    print(f"Filter sizes: {filter_sizes}")

    # Create model with detected architecture
    model = M5(
        kernel_sizes=[80, 3, 3, 3],
        filters=filter_sizes,
        strides=[16, 1, 1, 1],
        num_classes=num_classes,
    )

    # Convert old state dict to new format if needed
    if arch_type == "old":
        state_dict = convert_old_to_new_state_dict(state_dict)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_audio(audio_path: Path, target_sample_rate: int = 8000) -> torch.Tensor:
    """Preprocess a single audio file for inference."""
    # Try soundfile first, fallback to torchaudio
    try:
        import soundfile as sf
        wav_np, sample_rate = sf.read(audio_path)
        waveform = torch.from_numpy(wav_np).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.T
    except Exception:
        waveform, sample_rate = torchaudio.load(audio_path)

    # Convert to mono
    if waveform.shape[0] == 2:
        waveform = torch.mean(waveform, axis=0, keepdim=True)

    # Resample to target sample rate
    if sample_rate != target_sample_rate:
        resampler = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=target_sample_rate
        )
        waveform = resampler(waveform)

    # Add batch dimension
    waveform = waveform.unsqueeze(0)  # [1, 1, samples]

    return waveform


def predict_single(model: torch.nn.Module, audio_path: Path, device: torch.device) -> dict:
    """Run inference on a single audio file."""
    model.to(device)
    model.eval()

    # Preprocess audio
    waveform = preprocess_audio(audio_path)
    waveform = waveform.to(device)

    # Run inference
    with torch.no_grad():
        log_probs = model(waveform)
        probs = torch.exp(log_probs).squeeze().cpu().numpy()

    # Get prediction
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    # Build results
    results = {
        "predicted_class": pred_class,
        "predicted_index": pred_idx,
        "confidence": confidence,
        "all_probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(len(probs))}
    }

    return results


def evaluate_dataset(
    model: torch.nn.Module,
    data_dir: Path,
    batch_size: int,
    device: torch.device
) -> dict:
    """Evaluate model on test dataset."""
    model.to(device)
    model.eval()

    # Load dataset
    _, test_loader = load_dataset(str(data_dir), batch_size=batch_size)

    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    print("Evaluating on test set...")
    with torch.no_grad():
        for wavs, labels in tqdm(test_loader, desc="Evaluation"):
            wavs, labels = wavs.to(device), labels.to(device)

            # Forward pass
            log_probs = model(wavs)
            preds = torch.argmax(log_probs.squeeze(), dim=-1)

            # Accumulate results
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total if total > 0 else 0.0

    # Per-class accuracy
    per_class_correct = {i: 0 for i in range(10)}
    per_class_total = {i: 0 for i in range(10)}

    for pred, label in zip(all_preds, all_labels):
        per_class_total[label] += 1
        if pred == label:
            per_class_correct[label] += 1

    per_class_acc = {
        CLASS_NAMES[i]: per_class_correct[i] / per_class_total[i] if per_class_total[i] > 0 else 0.0
        for i in range(10)
    }

    results = {
        "overall_accuracy": accuracy,
        "total_samples": total,
        "correct_predictions": correct,
        "per_class_accuracy": per_class_acc,
    }

    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference with pretrained M5 model")
    parser.add_argument("--checkpoint", required=True, type=Path, help="Path to .pt checkpoint file")
    parser.add_argument("--mode", choices=["eval", "predict"], required=True, help="Evaluation or prediction mode")
    parser.add_argument("--data", type=Path, help="Data directory for evaluation mode")
    parser.add_argument("--audio", type=Path, help="Audio file path for prediction mode")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for evaluation")
    parser.add_argument("--num_classes", type=int, default=10, help="Number of output classes")
    return parser.parse_args()


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    device = pick_device()
    print(f"Using device: {device}")

    # Load pretrained model
    print(f"Loading model from {args.checkpoint}...")
    model = load_pretrained_model(args.checkpoint, num_classes=args.num_classes)
    print("Model loaded successfully!")

    if args.mode == "eval":
        if not args.data:
            raise ValueError("--data is required for evaluation mode")

        results = evaluate_dataset(model, args.data, args.batch_size, device)

        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        print(f"Overall Accuracy: {results['overall_accuracy']:.2%}")
        print(f"Correct: {results['correct_predictions']}/{results['total_samples']}")
        print("\nPer-Class Accuracy:")
        for class_name, acc in sorted(results['per_class_accuracy'].items()):
            print(f"  {class_name:15s}: {acc:.2%}")

    elif args.mode == "predict":
        if not args.audio:
            raise ValueError("--audio is required for prediction mode")

        if not args.audio.exists():
            raise FileNotFoundError(f"Audio file not found: {args.audio}")

        print(f"\nRunning inference on: {args.audio}")
        results = predict_single(model, args.audio, device)

        print("\n" + "="*50)
        print("PREDICTION RESULTS")
        print("="*50)
        print(f"Predicted Class: {results['predicted_class']}")
        print(f"Confidence: {results['confidence']:.2%}")
        print("\nAll Probabilities:")
        sorted_probs = sorted(
            results['all_probabilities'].items(),
            key=lambda x: x[1],
            reverse=True
        )
        for class_name, prob in sorted_probs:
            print(f"  {class_name:15s}: {prob:.2%}")


if __name__ == "__main__":
    main()
