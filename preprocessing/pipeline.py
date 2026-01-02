import torch
import torchaudio
import numpy as np

from pathlib import Path
from torch.utils.data import DataLoader, Dataset, random_split
from audiomentations import Compose, TimeStretch, PitchShift, AddGaussianNoise

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except Exception:
    sf = None
    HAS_SOUNDFILE = False


def select_audio_backend() -> str:
    try:
        backends = torchaudio.list_audio_backends()
    except AttributeError:
        try:
            backends = torchaudio.backend.list_audio_backends()
        except Exception:
            backends = []
    for name in ("soundfile", "sox_io"):
        if name in backends:
            try:
                torchaudio.set_audio_backend(name)
            except Exception:
                pass
            return name
    return ""


class SoundDataset(Dataset):
    def __init__(self, sound_dir, transforms=None, target_sample_rate=8000, file_list=None):
        self.sound_dir = sound_dir
        self.target_sample_rate = target_sample_rate

        # Filter and sort directories first, then enumerate
        valid_dirs = sorted([
            x for x in sound_dir.iterdir()
            if x.is_dir() and not x.name.startswith('.')
        ], key=lambda x: x.name)

        self.classes = {d.name: i for i, d in enumerate(valid_dirs)}

        self.transforms = transforms
        self.backend = select_audio_backend()
        self.use_soundfile = self.backend == "soundfile" or HAS_SOUNDFILE

        if file_list is not None:
            self.files = file_list
        else:
            self.files = []
            for ext in ("*.wav", "*.mp3"):
                self.files.extend(Path(self.sound_dir).rglob(ext))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        class_name = Path(self.files[idx]).parent.name
        label = self.classes[class_name]
        if self.use_soundfile and sf is not None:
            wav_np, sample_rate = sf.read(self.files[idx])
            waveform = torch.from_numpy(wav_np).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T
        else:
            waveform, sample_rate = torchaudio.load(self.files[idx])

        # Convert to mono
        if waveform.shape[0] == 2:
            waveform = torch.mean(waveform, axis=0, keepdim=True)

        # Resample to target sample rate
        if sample_rate != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=self.target_sample_rate
            )
            waveform = resampler(waveform)

        # Run augmentations (audiomentations expects numpy array with shape (samples,))
        if self.transforms:
            waveform_np = waveform.squeeze(0).numpy()
            waveform_np = self.transforms(samples=waveform_np, sample_rate=self.target_sample_rate)
            waveform = torch.from_numpy(waveform_np).unsqueeze(0)

        return waveform, label


def collate_fn(wavs):
    max_len = max([w[0].shape[-1] for w in wavs])

    labels = torch.LongTensor([w[1] for w in wavs])
    features = torch.zeros((len(wavs), 1, max_len))

    for i, (wav, _) in enumerate(wavs):
        features[i, 0, :wav.shape[-1]] = wav

    return features, labels


def load_dataset(sound_dir, batch_size=4, train_ratio=0.8):
    augmentations = Compose([
        TimeStretch(0.81, 1.23, p=0.5),
        PitchShift(-2, 2, p=0.5),
        AddGaussianNoise(0.001, 0.015, p=0.5),
    ])

    data_root = Path(sound_dir)
    subdirs = {x.name for x in data_root.iterdir() if x.is_dir()}

    if subdirs == {"test", "train"}:
        train_set = SoundDataset(data_root / "train", augmentations)
        test_set = SoundDataset(data_root / "test")
    else:
        # Collect all files first
        all_files = []
        for ext in ("*.wav", "*.mp3"):
            all_files.extend(data_root.rglob(ext))
        all_files = sorted(all_files)  # Sort for reproducibility

        # Split files into train/test
        torch.manual_seed(42)
        indices = torch.randperm(len(all_files)).tolist()
        train_size = int(len(all_files) * train_ratio)
        train_indices = indices[:train_size]
        test_indices = indices[train_size:]

        train_files = [all_files[i] for i in train_indices]
        test_files = [all_files[i] for i in test_indices]

        # Create separate datasets with different transforms
        train_set = SoundDataset(data_root, transforms=augmentations, file_list=train_files)
        test_set = SoundDataset(data_root, transforms=None, file_list=test_files)

    train_dataset = DataLoader(
        train_set, batch_size=batch_size, collate_fn=collate_fn, num_workers=1, shuffle=True)
    test_dataset = DataLoader(
        test_set, batch_size=batch_size, collate_fn=collate_fn, num_workers=1)

    return train_dataset, test_dataset
