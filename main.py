import os
import sys
import argparse
import numpy as np
import torch

import config
from data_processor import AudioProcessor, EmotionDataset, get_dataloaders, get_wav2vec2_dataloaders
from feature_extractor import FeatureExtractor
from models import get_model
from train import Trainer, evaluate_model, Wav2Vec2Trainer, evaluate_wav2vec2_model
from visualize import (plot_training_history, plot_confusion_matrix,
                      plot_audio_waveform, plot_mfcc, plot_mel_spectrogram,
                      generate_complete_report)


def predict_emotion_wav2vec2(audio_path, model_path=None):
    processor = AudioProcessor()
    audio = processor.load_audio(audio_path)
    audio = processor.normalize_audio(audio)
    audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(config.DEVICE)

    model = get_model('wav2vec2')
    default_path = config.MODEL_SAVE_PATH.replace('.pth', '_wav2vec2_best.pth')
    load_path = model_path if (model_path and os.path.exists(model_path)) else default_path

    if os.path.exists(load_path):
        checkpoint = torch.load(load_path, map_location=config.DEVICE)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                          if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)
        print(f"已加载 Wav2Vec2 模型 (val_acc={checkpoint['best_val_acc']:.2f}%)")
    else:
        print("⚠ 未找到 wav2vec2 模型，使用随机权重")

    model.eval()
    with torch.no_grad():
        outputs = model(audio_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_emotion = config.EMOTIONS[predicted.item()]
    confidence_score = confidence.item() * 100

    print("\n" + "="*50)
    print("情感识别结果 (Wav2Vec2)")
    print("="*50)
    print(f"音频文件: {audio_path}")
    print(f"识别结果: {predicted_emotion}")
    print(f"置信度: {confidence_score:.2f}%")
    print("-"*50)
    print("\n各情感概率分布:")
    for emotion, prob in zip(config.EMOTIONS, probabilities[0]):
        bar_length = int(prob.item() * 30)
        bar = '█' * bar_length + '░' * (30 - bar_length)
        print(f"{emotion:10s}: {bar} {prob.item()*100:.2f}%")
    print()

    return {
        'emotion': predicted_emotion,
        'confidence': confidence_score,
        'probabilities': probabilities[0].cpu().numpy()
    }


def predict_emotion(audio_path, model_path=None):
    processor = AudioProcessor()
    extractor = FeatureExtractor()

    audio = processor.load_audio(audio_path)
    audio = processor.normalize_audio(audio)

    features = extractor.extract_all_features(audio)

    mfcc = features['mfcc']
    mel_spec = features['mel_spectrogram']
    chroma = features['chroma']

    mfcc_tensor = torch.FloatTensor(mfcc).unsqueeze(0).unsqueeze(0)
    mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0).unsqueeze(0)
    chroma_tensor = torch.FloatTensor(chroma).unsqueeze(0).unsqueeze(0)

    model = get_model('hybrid')

    if model_path and os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=config.DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型: {model_path}")

    model.eval()

    with torch.no_grad():
        mfcc_input = mfcc_tensor.to(config.DEVICE)
        mel_input = mel_tensor.to(config.DEVICE)
        chroma_input = chroma_tensor.to(config.DEVICE)

        outputs = model(mfcc_input, mel_input, chroma_input)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    predicted_emotion = config.EMOTIONS[predicted.item()]
    confidence_score = confidence.item() * 100

    print("\n" + "="*50)
    print("情感识别结果")
    print("="*50)
    print(f"音频文件: {audio_path}")
    print(f"识别结果: {predicted_emotion}")
    print(f"置信度: {confidence_score:.2f}%")
    print("-"*50)
    print("\n各情感概率分布:")
    for emotion, prob in zip(config.EMOTIONS, probabilities[0]):
        bar_length = int(prob.item() * 30)
        bar = '█' * bar_length + '░' * (30 - bar_length)
        print(f"{emotion:10s}: {bar} {prob.item()*100:.2f}%")

    return {
        'emotion': predicted_emotion,
        'confidence': confidence_score,
        'probabilities': probabilities[0].cpu().numpy(),
        'features': {
            'mfcc': mfcc,
            'mel_spectrogram': mel_spec,
            'chroma': chroma
        }
    }


def visualize_single_audio(audio_path, save_dir='./logs/'):
    os.makedirs(save_dir, exist_ok=True)

    processor = AudioProcessor()
    extractor = FeatureExtractor()

    audio = processor.load_audio(audio_path)

    print("\n正在生成可视化图表...")

    plot_audio_waveform(audio, save_path=os.path.join(save_dir, 'input_waveform.png'),
                       title=f'Input Audio: {os.path.basename(audio_path)}')

    mfcc = extractor.extract_mfcc(audio)
    delta_mfcc = extractor.extract_delta_features(extractor.extract_mfcc(audio))
    plot_mfcc(delta_mfcc, save_path=os.path.join(save_dir, 'input_mfcc.png'),
             title='MFCC Features (with Delta)')

    mel_spec = extractor.extract_mel_spectrogram(audio)
    plot_mel_spectrogram(mel_spec,
                        save_path=os.path.join(save_dir, 'input_mel_spec.png'))

    all_features = extractor.extract_all_features(audio)
    from visualize import plot_feature_comparison
    plot_feature_comparison(all_features,
                           save_path=os.path.join(save_dir, 'all_features.png'))

    print(f"\n可视化完成！图表保存在: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description='语音情感识别系统')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')

    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                             help=f'训练轮数 (默认: {config.EPOCHS})')
    train_parser.add_argument('--batch-size', type=int, default=config.BATCH_SIZE,
                             help=f'批次大小 (默认: {config.BATCH_SIZE})')
    train_parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                             help=f'学习率 (默认: {config.LEARNING_RATE})')
    train_parser.add_argument('--model-type', type=str, default='hybrid',
                             choices=['cnn', 'lstm', 'hybrid', 'wav2vec2'],
                             help='模型类型 (默认: hybrid)')

    predict_parser = subparsers.add_parser('predict', help='预测单个音频文件的情感')
    predict_parser.add_argument('audio_path', help='音频文件路径')
    predict_parser.add_argument('--model-path', default=None,
                               help='模型权重路径')

    vis_parser = subparsers.add_parser('visualize', help='可视化音频特征')
    vis_parser.add_argument('audio_path', help='音频文件路径')

    eval_parser = subparsers.add_parser('evaluate', help='评估模型性能')
    eval_parser.add_argument('--model-path', default=None,
                            help='模型权重路径')

    ensemble_parser = subparsers.add_parser('ensemble', help='模型集成评估')

    demo_parser = subparsers.add_parser('demo', help='运行演示示例')

    args = parser.parse_args()

    if args.command == 'train':
        if args.epochs:
            config.EPOCHS = args.epochs
        if args.batch_size:
            config.BATCH_SIZE = args.batch_size
        if args.lr:
            config.LEARNING_RATE = args.lr

        if args.model_type == 'wav2vec2':
            print("加载数据集 (wav2vec2)...")
            try:
                train_loader, val_loader, test_loader = get_wav2vec2_dataloaders(
                    batch_size=config.WAV2VEC2_BATCH_SIZE
                )
            except Exception as e:
                print(f"\n❌ 数据加载失败: {e}")
                import traceback
                traceback.print_exc()
                return

            trainer = Wav2Vec2Trainer()
            history = trainer.train(train_loader, val_loader, epochs=config.EPOCHS)

            best_model_path = config.MODEL_SAVE_PATH.replace('.pth', '_wav2vec2_best.pth')
            trainer.load_model(best_model_path)
            results = evaluate_wav2vec2_model(trainer.model, test_loader)
        else:
            print("加载数据集...")
            try:
                train_loader, val_loader, test_loader = get_dataloaders(
                    batch_size=config.BATCH_SIZE
                )
            except Exception as e:
                print(f"\n❌ 数据加载失败: {e}")
                print("\n请确保数据目录结构如下:")
                print("./data/")
                print("  ├── train/")
                print("  │   ├── angry/")
                print("  │   ├── disgust/")
                print("  │   ├── fear/")
                print("  │   ├── happy/")
                print("  │   ├── neutral/")
                print("  │   └── sad/")
                print("  ├── val/")
                print("  └── test/")
                return

            trainer = Trainer(model_type=args.model_type)
            history = trainer.train(train_loader, val_loader, epochs=config.EPOCHS)

            best_model_path = config.MODEL_SAVE_PATH.replace('.pth', '_best.pth')
            trainer.load_model(best_model_path)
            results = evaluate_model(trainer.model, test_loader)

        generate_complete_report(results, history, save_dir='./logs/')

    elif args.command == 'predict':
        if not os.path.exists(args.audio_path):
            print(f"❌ 文件不存在: {args.audio_path}")
            return

        result = predict_emotion_wav2vec2(args.audio_path, args.model_path)

        visualize_single_audio(args.audio_path)

    elif args.command == 'visualize':
        if not os.path.exists(args.audio_path):
            print(f"❌ 文件不存在: {args.audio_path}")
            return

        visualize_single_audio(args.audio_path)

    elif args.command == 'evaluate':
        _, _, test_loader = get_dataloaders()
        model = get_model('hybrid')

        model_path = args.model_path or config.MODEL_SAVE_PATH.replace('.pth', '_best.pth')
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=config.DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])

        results = evaluate_model(model, test_loader)

        plot_confusion_matrix(
            results['labels'],
            results['predictions'],
            save_path='./logs/confusion_matrix.png'
        )

    elif args.command == 'ensemble':
        print("\n" + "="*60)
        print("模型集成评估 (Wav2Vec2 + CNN-LSTM)")
        print("="*60)

        _, _, test_loader_raw = get_wav2vec2_dataloaders()
        _, _, test_loader_feat = get_dataloaders()

        w2v_model = get_model('wav2vec2')
        w2v_path = config.MODEL_SAVE_PATH.replace('.pth', '_wav2vec2_best.pth')
        w2v_loaded = False
        if os.path.exists(w2v_path):
            ckpt = torch.load(w2v_path, map_location=config.DEVICE)
            model_dict = w2v_model.state_dict()
            pretrained_dict = {k: v for k, v in ckpt['model_state_dict'].items()
                              if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(pretrained_dict)
            w2v_model.load_state_dict(model_dict, strict=False)
            print(f"已加载 Wav2Vec2 模型 ({len(pretrained_dict)}/{len(model_dict)} 层匹配, val_acc={ckpt['best_val_acc']:.2f}%)")
            w2v_loaded = True
        if not w2v_loaded:
            print("⚠ 未找到兼容的 Wav2Vec2 模型，跳过")
        w2v_model.eval()

        cnn_model = get_model('hybrid')
        cnn_path = config.MODEL_SAVE_PATH.replace('.pth', '_best.pth')
        cnn_loaded = False
        if os.path.exists(cnn_path):
            ckpt2 = torch.load(cnn_path, map_location=config.DEVICE)
            cnn_model.load_state_dict(ckpt2['model_state_dict'])
            print(f"已加载 CNN-LSTM 模型 (val_acc={ckpt2['best_val_acc']:.2f}%)")
            cnn_loaded = True
        if not cnn_loaded:
            print("⚠ 未找到 CNN-LSTM 模型，跳过")
        cnn_model.eval()

        if not w2v_loaded and not cnn_loaded:
            print("❌ 无可用模型")
            return

        import numpy as np
        from sklearn.metrics import classification_report, confusion_matrix
        all_labels = []

        print("\n正在评估 Wav2Vec2 模型...")
        w2v_preds = []
        for batch in test_loader_raw:
            audio = batch['audio'].to(config.DEVICE)
            labels = batch['label'].to(config.DEVICE)
            with torch.no_grad():
                p = torch.softmax(w2v_model(audio), dim=1)
                w2v_preds.extend(p.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        ensemble_preds = np.array(w2v_preds)

        if cnn_loaded:
            print("正在评估 CNN-LSTM 模型...")
            cnn_preds = []
            for batch in test_loader_feat:
                audio = batch['mfcc'].to(config.DEVICE)
                mel = batch['mel_spectrogram'].to(config.DEVICE)
                chroma = batch['chroma'].to(config.DEVICE)
                with torch.no_grad():
                    p = torch.softmax(cnn_model(audio, mel, chroma), dim=1)
                    cnn_preds.extend(p.argmax(dim=1).cpu().numpy())
            cnn_preds = np.array(cnn_preds)
            ensemble_preds = np.where(
                np.random.random(len(ensemble_preds)) < 0.6,
                ensemble_preds,
                cnn_preds[:len(ensemble_preds)]
            )

        acc = 100. * np.mean(ensemble_preds == np.array(all_labels))
        print(f"\n集成准确率: {acc:.2f}%")
        print("\n" + classification_report(all_labels, ensemble_preds, target_names=config.EMOTIONS, digits=4))
        plot_confusion_matrix(all_labels, ensemble_preds, save_path='./logs/ensemble_confusion.png')

    elif args.command == 'demo':
        print("\n" + "="*60)
        print("🎵 语音情感识别系统 - 演示模式")
        print("="*60)

        print("\n系统功能:")
        print("  ✓ 基于深度学习的语音情感识别")
        print("  ✓ 支持6种基本情感分类:")
        for i, emotion in enumerate(config.EMOTIONS, 1):
            print(f"      {i}. {emotion}")
        print("\n核心技术:")
        print("  • Wav2Vec2 预训练模型微调 (方案4)")
        print("  • MFCC特征提取 + 一阶二阶差分")
        print("  • 梅尔频谱图特征")
        print("  • 色度特征")
        print("  • CNN+LSTM混合架构")
        print("  • 注意力机制")
        print("\n使用方法:")
        print("  1. 训练传统模型: python main.py train --model-type hybrid")
        print("  2. 训练wav2vec2: python main.py train --model-type wav2vec2")
        print("  3. 预测情感: python main.py predict <音频文件>")
        print("  4. 可视化: python main.py visualize <音频文件>")
        print("  5. 评估模型: python main.py evaluate")
        print("\n" + "="*60)

    else:
        parser.print_help()


if __name__ == '__main__':
    main()