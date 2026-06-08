import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import numpy as np
import os
import json
import time
from datetime import datetime

import config
from models import get_model
from data_processor import get_dataloaders


class Trainer:
    def __init__(self, model_type='hybrid', model_name='emotion_recognition'):
        self.device = config.DEVICE
        self.model_type = model_type
        self.model_name = model_name

        self.model = get_model(model_type)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=config.LEARNING_RATE,
                                    weight_decay=1e-4)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                           patience=8, verbose=True, min_lr=1e-6)

        self.best_val_acc = 0.0
        self.train_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'lr': []
        }

        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        for batch in progress_bar:
            mfcc = batch['mfcc'].to(self.device)
            mel_spec = batch['mel_spectrogram'].to(self.device)
            chroma = batch['chroma'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            outputs = self.model(mfcc, mel_spec, chroma)
            loss = self.criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for batch in val_loader:
            mfcc = batch['mfcc'].to(self.device)
            mel_spec = batch['mel_spectrogram'].to(self.device)
            chroma = batch['chroma'].to(self.device)
            labels = batch['label'].to(self.device)

            outputs = self.model(mfcc, mel_spec, chroma)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)

    def train(self, train_loader, val_loader, epochs=config.EPOCHS):
        print(f"\n{'='*60}")
        print(f"开始训练 {self.model_type.upper()} 模型")
        print(f"设备: {self.device}")
        print(f"训练轮数: {epochs}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(epochs):
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print("-" * 40)

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.validate(val_loader)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['lr'].append(current_lr)

            print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"Learning Rate: {current_lr:.6f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(is_best=True)
                print(f"✓ 新的最佳模型! 验证准确率: {val_acc:.2f}%")

            if (epoch + 1) % 10 == 0:
                self.save_model(is_best=False)

        training_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"训练完成!")
        print(f"总训练时间: {training_time/60:.2f} 分钟")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        print(f"{'='*60}")

        self.save_training_history()
        return self.train_history

    def save_model(self, is_best=True):
        if is_best:
            save_path = config.MODEL_SAVE_PATH.replace('.pth', '_best.pth')
        else:
            save_path = config.MODEL_SAVE_PATH.replace('.pth', '_latest.pth')

        checkpoint = {
            'epoch': len(self.train_history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'model_type': self.model_type,
            'config': {
                'num_classes': config.NUM_CLASSES,
                'sample_rate': config.SAMPLE_RATE,
                'n_mfcc': config.N_MFCC
            }
        }

        torch.save(checkpoint, save_path)
        print(f"模型已保存至: {save_path}")

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"模型已从 {model_path} 加载")

    def save_training_history(self):
        history_path = os.path.join(config.LOG_DIR, f'{self.model_name}_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        print(f"训练历史已保存至: {history_path}")


class Wav2Vec2Trainer:
    def __init__(self, model_name='wav2vec2_emotion'):
        self.device = config.DEVICE
        self.model_name = model_name

        self.model = get_model('wav2vec2')

        feat_params = []
        enc_params = []
        head_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if 'classifier' in name:
                    head_params.append(param)
                elif 'feature_extractor' in name:
                    feat_params.append(param)
                else:
                    enc_params.append(param)

        self.optimizer = optim.AdamW([
            {'params': feat_params, 'lr': config.WAV2VEC2_LR * 0.5},
            {'params': enc_params, 'lr': config.WAV2VEC2_LR},
            {'params': head_params, 'lr': config.WAV2VEC2_LR * 10}
        ], weight_decay=0.01)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5,
                                           patience=4, verbose=True, min_lr=1e-7)

        self.best_val_acc = 0.0
        self.train_history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'lr': []
        }

        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        progress_bar = tqdm(train_loader, desc='Training', leave=False)
        for batch in progress_bar:
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(audio)
            loss = self.criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for batch in val_loader:
            audio = batch['audio'].to(self.device)
            labels = batch['label'].to(self.device)

            outputs = self.model(audio)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy, np.array(all_preds), np.array(all_labels)

    def train(self, train_loader, val_loader, epochs=config.WAV2VEC2_EPOCHS):
        print(f"\n{'='*60}")
        print(f"开始训练 WAV2VEC2 模型")
        print(f"设备: {self.device}")
        print(f"预训练模型: {config.WAV2VEC2_MODEL_NAME}")
        print(f"训练轮数: {epochs}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(epochs):
            print(f"\nEpoch [{epoch+1}/{epochs}]")
            print("-" * 40)

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc, _, _ = self.validate(val_loader)

            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']

            self.train_history['train_loss'].append(train_loss)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['lr'].append(current_lr)

            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"LR: feat={self.optimizer.param_groups[0]['lr']:.1e}  enc={self.optimizer.param_groups[1]['lr']:.1e}  head={self.optimizer.param_groups[2]['lr']:.1e}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(is_best=True)
                print(f"✓ 新的最佳模型! 验证准确率: {val_acc:.2f}%")

            if (epoch + 1) % 5 == 0:
                self.save_model(is_best=False)

        training_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"训练完成!")
        print(f"总训练时间: {training_time/60:.2f} 分钟")
        print(f"最佳验证准确率: {self.best_val_acc:.2f}%")
        print(f"{'='*60}")

        self.save_training_history()
        return self.train_history

    def save_model(self, is_best=True):
        if is_best:
            save_path = config.MODEL_SAVE_PATH.replace('.pth', '_wav2vec2_best.pth')
        else:
            save_path = config.MODEL_SAVE_PATH.replace('.pth', '_wav2vec2_latest.pth')

        checkpoint = {
            'epoch': len(self.train_history['train_loss']),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc,
            'model_type': 'wav2vec2',
        }

        torch.save(checkpoint, save_path)
        print(f"模型已保存至: {save_path}")

    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"模型已从 {model_path} 加载")

    def save_training_history(self):
        history_path = os.path.join(config.LOG_DIR, f'{self.model_name}_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.train_history, f, indent=2)
        print(f"训练历史已保存至: {history_path}")


@torch.no_grad()
def evaluate_model(model, test_loader, device=config.DEVICE):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    criterion = nn.CrossEntropyLoss()

    print("\n正在评估模型...")
    for batch in tqdm(test_loader, desc='Testing'):
        mfcc = batch['mfcc'].to(device)
        mel_spec = batch['mel_spectrogram'].to(device)
        chroma = batch['chroma'].to(device)
        labels = batch['label'].to(device)

        outputs = model(mfcc, mel_spec, chroma)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = 100. * correct / total

    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(all_labels, all_preds,
                                  target_names=config.EMOTIONS,
                                  digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n测试集结果:")
    print(f"损失: {test_loss:.4f}")
    print(f"准确率: {accuracy:.2f}%")
    print("\n分类报告:")
    print(report)
    print("\n混淆矩阵:")
    print(cm)

    return {
        'loss': test_loss,
        'accuracy': accuracy,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs),
        'classification_report': report,
        'confusion_matrix': cm
    }


@torch.no_grad()
def evaluate_wav2vec2_model(model, test_loader, device=config.DEVICE):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    criterion = nn.CrossEntropyLoss()

    print("\n正在评估 Wav2Vec2 模型...")
    for batch in tqdm(test_loader, desc='Testing'):
        audio = batch['audio'].to(device)
        labels = batch['label'].to(device)

        outputs = model(audio)
        loss = criterion(outputs, labels)

        test_loss += loss.item()
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    test_loss /= len(test_loader)
    accuracy = 100. * correct / total

    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(all_labels, all_preds,
                                  target_names=config.EMOTIONS,
                                  digits=4)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n测试集结果:")
    print(f"损失: {test_loss:.4f}")
    print(f"准确率: {accuracy:.2f}%")
    print("\n分类报告:")
    print(report)
    print("\n混淆矩阵:")
    print(cm)

    return {
        'loss': test_loss,
        'accuracy': accuracy,
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs),
        'classification_report': report,
        'confusion_matrix': cm
    }


def main():
    print("正在加载数据...")
    try:
        train_loader, val_loader, test_loader = get_dataloaders()
    except Exception as e:
        print(f"数据加载错误: {e}")
        print("请确保数据目录结构正确: ./data/train/, ./data/val/, ./data/test/")
        print("每个目录下应包含情感子文件夹: angry, disgust, fear, happy, neutral, sad")
        return

    trainer = Trainer(model_type='hybrid')
    history = trainer.train(train_loader, val_loader, epochs=config.EPOCHS)

    print("\n加载最佳模型进行最终评估...")
    best_model_path = config.MODEL_SAVE_PATH.replace('.pth', '_best.pth')
    if os.path.exists(best_model_path):
        trainer.load_model(best_model_path)
        results = evaluate_model(trainer.model, test_loader)
    else:
        print("未找到最佳模型，使用当前模型进行评估")
        results = evaluate_model(trainer.model, test_loader)


if __name__ == '__main__':
    main()