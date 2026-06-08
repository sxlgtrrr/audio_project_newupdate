import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import librosa
import librosa.display
import config
import os


def plot_training_history(history, save_path='./logs/training_curves.png'):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Loss Curve', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['lr'], 'g-', label='Learning Rate', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].axis('off')
    info_text = f"""
    Training Summary
    ─────────────────
    Total Epochs: {len(epochs)}
    
    Final Training Loss: {history['train_loss'][-1]:.4f}
    Final Validation Loss: {history['val_loss'][-1]:.4f}
    
    Best Training Accuracy: {max(history['train_acc']):.2f}%
    Best Validation Accuracy: {max(history['val_acc']):.2f}%
    
    Initial LR: {history['lr'][0]:.6f}
    Final LR: {history['lr'][-1]:.6f}
    """
    axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, family='monospace',
                   verticalalignment='center', transform=axes[1, 1].transAxes,
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"训练曲线已保存至: {save_path}")


def plot_confusion_matrix(y_true, y_pred, classes=config.EMOTIONS,
                         save_path='./logs/confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes,
               yticklabels=classes, ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')

    sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues', xticklabels=classes,
               yticklabels=classes, ax=ax2, cbar_kws={'label': 'Percentage'})
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"混淆矩阵已保存至: {save_path}")


def plot_audio_waveform(audio, sample_rate=config.SAMPLE_RATE,
                        save_path='./logs/waveform.png', title='Audio Waveform'):
    fig, ax = plt.subplots(figsize=(12, 4))
    time_axis = np.linspace(0, len(audio) / sample_rate, len(audio))
    ax.plot(time_axis, audio, color='blue', linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_mfcc(mfcc_data, save_path='./logs/mfcc_visualization.png',
             title='MFCC Features'):
    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(mfcc_data, sr=config.SAMPLE_RATE,
                                   hop_length=config.HOP_LENGTH,
                                   x_axis='time', y_axis='mel', ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(img, format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_mel_spectrogram(mel_spec, save_path='./logs/mel_spectrogram.png',
                        title='Mel Spectrogram'):
    fig, ax = plt.subplots(figsize=(12, 6))
    img = librosa.display.specshow(mel_spec, sr=config.SAMPLE_RATE,
                                   hop_length=config.HOP_LENGTH,
                                   x_axis='time', y_axis='mel', ax=ax)
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.colorbar(img, format='%+2.0f dB')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_emotion_distribution(predictions, probabilities, emotions=config.EMOTIONS,
                             save_path='./logs/emotion_distribution.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    emotion_counts = [predictions.tolist().count(i) for i in range(len(emotions))]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

    bars = axes[0].bar(emotions, emotion_counts, color=colors, edgecolor='black')
    axes[0].set_title('Emotion Distribution (Predictions)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Emotion')
    axes[0].set_ylabel('Count')

    for bar, count in zip(bars, emotion_counts):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{count}', ha='center', va='bottom', fontweight='bold')

    avg_probs = np.mean(probabilities, axis=0)
    bars2 = axes[1].bar(emotions, avg_probs * 100, color=colors, edgecolor='black')
    axes[1].set_title('Average Confidence per Emotion (%)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Emotion')
    axes[1].set_ylabel('Average Confidence (%)')

    for bar, prob in zip(bars2, avg_probs):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob*100:.1f}%', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"情感分布图已保存至: {save_path}")


def plot_feature_comparison(features_dict, save_path='./logs/feature_comparison.png'):
    n_features = len(features_dict)
    fig, axes = plt.subplots(n_features, 1, figsize=(14, 4*n_features))

    if n_features == 1:
        axes = [axes]

    for idx, (name, feature) in enumerate(features_dict.items()):
        im = axes[idx].imshow(feature, aspect='auto', cmap='viridis',
                             origin='lower')
        axes[idx].set_title(f'{name.upper()} Visualization', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Time Frames')
        axes[idx].set_ylabel('Feature Dimension')
        plt.colorbar(im, ax=axes[idx])

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def generate_complete_report(results, history, audio_sample=None,
                           save_dir='./logs/'):
    os.makedirs(save_dir, exist_ok=True)

    print("\n生成完整实验报告...")

    plot_training_history(history, os.path.join(save_dir, 'training_curves.png'))

    if results:
        plot_confusion_matrix(
            results['labels'],
            results['predictions'],
            save_path=os.path.join(save_dir, 'confusion_matrix.png')
        )

        plot_emotion_distribution(
            results['predictions'],
            results['probabilities'],
            save_path=os.path.join(save_dir, 'emotion_distribution.png')
        )

    if audio_sample is not None:
        from data_processor import AudioProcessor
        processor = AudioProcessor()

        plot_audio_waveform(audio_sample,
                          save_path=os.path.join(save_dir, 'waveform_example.png'))

        from feature_extractor import FeatureExtractor
        extractor = FeatureExtractor()

        mfcc = extractor.extract_mfcc(audio_sample)
        plot_mfcc(mfcc, save_path=os.path.join(save_dir, 'mfcc_example.png'))

        mel_spec = extractor.extract_mel_spectrogram(audio_sample)
        plot_mel_spectrogram(mel_spec,
                            save_path=os.path.join(save_dir, 'mel_spectrogram_example.png'))

    print(f"\n所有可视化图表已保存至: {save_dir}")