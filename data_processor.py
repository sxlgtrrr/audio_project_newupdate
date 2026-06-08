import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import os
import config
import random
import scipy.io.wavfile as wav
from scipy import signal

class AudioProcessor:
    def __init__(self, sample_rate=config.SAMPLE_RATE, duration=config.DURATION):
        self.sample_rate = sample_rate
        self.duration = duration
        self.n_samples = int(sample_rate * duration)

    def load_audio(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.sample_rate, duration=self.duration)
        except Exception:
            try:
                sr, data = wav.read(file_path)
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                elif data.dtype == np.int32:
                    data = data.astype(np.float32) / 2147483648.0
                else:
                    data = data.astype(np.float32)
                if data.ndim > 1:
                    data = np.mean(data, axis=1)
                if sr != self.sample_rate:
                    data = signal.resample(data, int(len(data) * self.sample_rate / sr))
                num_samples = int(self.sample_rate * self.duration)
                if len(data) < num_samples:
                    data = np.pad(data, (0, num_samples - len(data)), mode='constant')
                else:
                    data = data[:num_samples]
                audio = data
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                return np.zeros(self.n_samples)

        if len(audio) < self.n_samples:
            padding = self.n_samples - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        else:
            audio = audio[:self.n_samples]
        return audio

    def normalize_audio(self, audio):
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        return audio

    def pre_emphasis(self, audio, coeff=0.97):
        return np.append(audio[0], audio[1:] - coeff * audio[:-1])

    def add_noise(self, audio, noise_factor=0.005):
        noise = np.random.randn(len(audio))
        augmented_audio = audio + noise_factor * noise
        return augmented_audio

    def time_shift(self, audio, shift_max=0.2):
        shift = int(random.uniform(-shift_max, shift_max) * len(audio))
        if shift > 0:
            audio = np.roll(audio, shift)
        elif shift < 0:
            audio = np.roll(audio, shift)
        return audio

    def pitch_shift(self, audio, n_steps=2):
        return librosa.effects.pitch_shift(y=audio, sr=self.sample_rate, n_steps=n_steps)

    def time_stretch(self, audio, rate=1.0):
        return librosa.effects.time_stretch(y=audio, rate=rate)


class EmotionDataset(Dataset):
    def __init__(self, data_dir=config.DATA_DIR, mode='train', transform=None):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        self.processor = AudioProcessor()
        self.audio_files = []
        self.labels = []

        self._load_dataset()

    def _load_dataset(self):
        for idx, emotion in enumerate(config.EMOTIONS):
            emotion_dir = os.path.join(self.data_dir, self.mode, emotion)
            if os.path.exists(emotion_dir):
                for file_name in os.listdir(emotion_dir):
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(emotion_dir, file_name)
                        self.audio_files.append(file_path)
                        self.labels.append(idx)
        print(f"Loaded {len(self.audio_files)} samples for {self.mode} set")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]

        audio = self.processor.load_audio(audio_path)
        audio = self.processor.normalize_audio(audio)

        if self.transform and self.mode == 'train':
            if random.random() > 0.5:
                audio = self.processor.add_noise(audio)
            if random.random() > 0.4:
                audio = self.processor.time_shift(audio)
            if random.random() > 0.5:
                audio = self.processor.pitch_shift(audio, n_steps=random.uniform(-3, 3))

        mfcc = self._extract_mfcc(audio)
        mel_spec = self._extract_mel_spectrogram(audio)
        chroma = self._extract_chroma(audio)

        if self.transform and self.mode == 'train':
            mfcc = self._spec_augment(mfcc, freq_mask=10, time_mask=10)
            mel_spec = self._spec_augment(mel_spec, freq_mask=16, time_mask=10)

        mfcc = (mfcc - mfcc.mean(axis=1, keepdims=True)) / (mfcc.std(axis=1, keepdims=True) + 1e-8)
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-8)

        features = {
            'mfcc': torch.FloatTensor(mfcc),
            'mel_spectrogram': torch.FloatTensor(mel_spec),
            'chroma': torch.FloatTensor(chroma),
            'label': torch.LongTensor([label])
        }
        return features

    def _extract_mfcc(self, audio):
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.processor.sample_rate,
            n_mfcc=config.N_MFCC,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            win_length=config.WIN_LENGTH
        )
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        combined = np.concatenate([mfcc, delta, delta2], axis=0)
        return combined

    def _extract_mel_spectrogram(self, audio):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.processor.sample_rate,
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    def _extract_chroma(self, audio):
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.processor.sample_rate,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH
        )
        return chroma

    def _spec_augment(self, spec, freq_mask=8, time_mask=10):
        if spec.ndim == 2:
            num_freq, num_time = spec.shape
        else:
            _, num_freq, num_time = spec.shape
        freq_masked = spec.copy()

        for _ in range(random.randint(1, 3)):
            f = random.randint(0, max(0, num_freq - freq_mask))
            f_len = random.randint(1, min(freq_mask, num_freq - f))
            freq_masked[f:f+f_len, :] = freq_masked.mean()

        for _ in range(random.randint(1, 3)):
            t = random.randint(0, max(0, num_time - time_mask))
            t_len = random.randint(1, min(time_mask, num_time - t))
            freq_masked[:, t:t+t_len] = freq_masked.mean()

        return freq_masked


def collate_fn(batch):
    mfccs = []
    mel_specs = []
    chromas = []
    labels = []

    for item in batch:
        mfccs.append(item['mfcc'])
        mel_specs.append(item['mel_spectrogram'])
        chromas.append(item['chroma'])
        labels.append(item['label'])

    mfccs = torch.stack(mfccs)
    mel_specs = torch.stack(mel_specs)
    chromas = torch.stack(chromas)
    labels = torch.cat(labels)

    return {
        'mfcc': mfccs.unsqueeze(1),
        'mel_spectrogram': mel_specs.unsqueeze(1),
        'chroma': chromas.unsqueeze(1),
        'label': labels
    }


def get_dataloaders(data_dir=config.DATA_DIR, batch_size=config.BATCH_SIZE):
    import multiprocessing
    nw = min(4, multiprocessing.cpu_count() // 2)

    train_dataset = EmotionDataset(data_dir=data_dir, mode='train', transform=True)
    val_dataset = EmotionDataset(data_dir=data_dir, mode='val', transform=False)
    test_dataset = EmotionDataset(data_dir=data_dir, mode='test', transform=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=collate_fn, num_workers=nw,
                             pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=collate_fn, num_workers=nw,
                           pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=nw,
                            pin_memory=True)

    return train_loader, val_loader, test_loader


class Wav2Vec2EmotionDataset(Dataset):
    def __init__(self, data_dir=config.DATA_DIR, mode='train'):
        self.data_dir = data_dir
        self.mode = mode
        self.processor = AudioProcessor()
        self.audio_files = []
        self.labels = []
        self._load_dataset()

    def _load_dataset(self):
        for idx, emotion in enumerate(config.EMOTIONS):
            emotion_dir = os.path.join(self.data_dir, self.mode, emotion)
            if os.path.exists(emotion_dir):
                for file_name in os.listdir(emotion_dir):
                    if file_name.endswith('.wav'):
                        file_path = os.path.join(emotion_dir, file_name)
                        self.audio_files.append(file_path)
                        self.labels.append(idx)
        print(f"Loaded {len(self.audio_files)} samples for {self.mode} set")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        label = self.labels[idx]
        audio = self.processor.load_audio(audio_path)
        audio = self.processor.normalize_audio(audio)
        return {
            'audio': torch.FloatTensor(audio),
            'label': torch.LongTensor([label])
        }


def wav2vec2_collate_fn(batch):
    audios = torch.stack([item['audio'] for item in batch])
    labels = torch.cat([item['label'] for item in batch])
    return {'audio': audios, 'label': labels}


def get_wav2vec2_dataloaders(data_dir=config.DATA_DIR, batch_size=config.WAV2VEC2_BATCH_SIZE):
    import multiprocessing
    nw = min(2, multiprocessing.cpu_count() // 4)

    train_dataset = Wav2Vec2EmotionDataset(data_dir=data_dir, mode='train')
    val_dataset = Wav2Vec2EmotionDataset(data_dir=data_dir, mode='val')
    test_dataset = Wav2Vec2EmotionDataset(data_dir=data_dir, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             collate_fn=wav2vec2_collate_fn, num_workers=nw,
                             pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           collate_fn=wav2vec2_collate_fn, num_workers=nw,
                           pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=wav2vec2_collate_fn, num_workers=nw,
                            pin_memory=True)

    return train_loader, val_loader, test_loader