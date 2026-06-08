import numpy as np
import librosa
from scipy import signal
import config


class FeatureExtractor:
    def __init__(self, sample_rate=config.SAMPLE_RATE):
        self.sample_rate = sample_rate

    def extract_mfcc(self, audio, n_mfcc=config.N_MFCC):
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=n_mfcc,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            win_length=config.WIN_LENGTH,
            window='hann'
        )
        return mfcc

    def extract_delta_features(self, features):
        delta = librosa.feature.delta(features)
        delta2 = librosa.feature.delta(features, order=2)
        return np.concatenate([features, delta, delta2], axis=0)

    def extract_mel_spectrogram(self, audio, n_mels=config.N_MELS):
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=n_mels,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            power=2.0
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return log_mel_spec

    def extract_chroma_features(self, audio):
        chroma = librosa.feature.chroma_stft(
            y=audio,
            sr=self.sample_rate,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH
        )
        return chroma

    def extract_spectral_contrast(self, audio):
        spectral_contrast = librosa.feature.spectral_contrast(
            y=audio,
            sr=self.sample_rate,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH
        )
        return spectral_contrast

    def extract_tonnetz(self, audio):
        tonnetz = librosa.feature.tonnetz(
            y=librosa.effects.harmonic(audio),
            sr=self.sample_rate
        )
        return tonnetz

    def extract_zero_crossing_rate(self, audio):
        zcr = librosa.feature.zero_crossing_rate(audio, frame_length=config.N_FFT,
                                                  hop_length=config.HOP_LENGTH)
        return zcr

    def extract_rms_energy(self, audio):
        rms = librosa.feature.rms(y=audio, frame_length=config.N_FFT,
                                  hop_length=config.HOP_LENGTH)
        return rms

    def extract_all_features(self, audio):
        features_dict = {}

        mfcc = self.extract_mfcc(audio)
        mfcc_with_delta = self.extract_delta_features(mfcc)
        features_dict['mfcc'] = mfcc_with_delta

        features_dict['mel_spectrogram'] = self.extract_mel_spectrogram(audio)

        features_dict['chroma'] = self.extract_chroma_features(audio)

        features_dict['mfcc'] = (features_dict['mfcc'] - features_dict['mfcc'].mean(axis=1, keepdims=True)) / (features_dict['mfcc'].std(axis=1, keepdims=True) + 1e-8)
        features_dict['mel_spectrogram'] = (features_dict['mel_spectrogram'] - features_dict['mel_spectrogram'].mean()) / (features_dict['mel_spectrogram'].std() + 1e-8)

        features_dict['spectral_contrast'] = self.extract_spectral_contrast(audio)

        try:
            features_dict['tonnetz'] = self.extract_tonnetz(audio)
        except:
            pass

        features_dict['zcr'] = self.extract_zero_crossing_rate(audio)
        features_dict['rms'] = self.extract_rms_energy(audio)

        return features_dict


def compute_statistics(features):
    mean = np.mean(features, axis=1)
    std = np.std(features, axis=1)
    median = np.median(features, axis=1)
    max_val = np.max(features, axis=1)
    min_val = np.min(features, axis=1)

    stats = np.concatenate([mean, std, median, max_val, min_val])
    return stats


class FrameProcessor:
    def __init__(self, frame_size=config.WIN_LENGTH, hop_size=config.HOP_LENGTH,
                 sample_rate=config.SAMPLE_RATE):
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.sample_rate = sample_rate
        self.window = signal.windows.hann(frame_size, sym=False)

    def frame_signal(self, audio):
        n_frames = 1 + (len(audio) - self.frame_size) // self.hop_size
        frames = np.zeros((n_frames, self.frame_size))

        for i in range(n_frames):
            start = i * self.hop_size
            end = start + self.frame_size
            frames[i] = audio[start:end] * self.window

        return frames

    def short_time_fourier_transform(self, audio):
        _, _, stft = signal.stft(
            audio,
            fs=self.sample_rate,
            window='hann',
            nperseg=self.frame_size,
            noverlap=self.frame_size - self.hop_size
        )
        return stft

    def compute_power_spectrum(self, stft):
        power_spectrum = np.abs(stft) ** 2
        return power_spectrum

    def apply_filter_bank(self, power_spectrum, n_filters=26):
        low_freq_mel = 0
        high_freq_mel = 2595 * np.log10(1 + (self.sample_rate / 2) / 700)
        mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
        hz_points = 700 * (10 ** (mel_points / 2595) - 1)
        bin_points = np.floor((self.frame_size + 1) * hz_points / self.sample_rate).astype(int)

        filter_bank = np.zeros((n_filters, self.frame_size // 2 + 1))

        for i in range(1, n_filters + 1):
            left = bin_points[i-1]
            center = bin_points[i]
            right = bin_points[i+1]

            for j in range(left, center):
                if j < len(filter_bank[i-1]):
                    filter_bank[i-1][j] = (j - left) / (center - left) if center != left else 0

            for j in range(center, right):
                if j < len(filter_bank[i-1]):
                    filter_bank[i-1][j] = (right - j) / (right - center) if right != center else 0

        filter_banks = np.dot(power_spectrum[:self.frame_size//2+1], filter_bank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
        filter_banks = 20 * np.log10(filter_banks)

        return filter_banks