import os
import subprocess
import sys

def create_data_structure():
    data_dir = './data'
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    splits = ['train', 'val', 'test']

    print("创建数据目录结构...")

    for split in splits:
        for emotion in emotions:
            dir_path = os.path.join(data_dir, split, emotion)
            os.makedirs(dir_path, exist_ok=True)
            print(f"  ✓ {dir_path}")

    print("\n目录结构创建完成!")
    print(f"\n请将音频文件放入对应目录:")
    for split in splits:
        print(f"\n{split}/")
        for emotion in emotions:
            print(f"  └── {emotion}/  (放置{emotion}情感的.wav文件)")


def download_ravdess():
    print("\n" + "="*60)
    print("RAVDESS 数据集下载说明")
    print("="*60)

    print("""
数据集信息:
- 名称: RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- 内容: 24名演员(12男,12女)的语音情感数据
- 情感类别: neutral, calm, happy, sad, angry, fearful, disgust, surprised
- 格式: 16kHz, 16bit, WAV文件
- 数量: 约1440个音频文件

下载地址:
https://zenodo.org/record/1188976/files/Ravdess_speech_files_audio_audio_01.zip

使用方法:
1. 下载数据集
2. 解压到 ./data/raw/ 目录
3. 运行 python prepare_ravdess.py 进行数据预处理
""")

    if input("\n是否打开下载链接? (y/n): ").lower() == 'y':
        import webbrowser
        webbrowser.open('https://zenodo.org/record/1188976')


def prepare_cremad(raw_dir='./data/crema-d/AudioWAV', output_dir='./data'):
    import numpy as np
    import soundfile as sf
    import random as rnd

    emotion_map = {
        'ANG': 'angry',
        'DIS': 'disgust',
        'FEA': 'fear',
        'HAP': 'happy',
        'NEU': 'neutral',
        'SAD': 'sad',
    }

    print(f"\n{'='*60}")
    print("CREMA-D 数据集预处理")
    print(f"{'='*60}")
    print(f"源目录: {raw_dir}")
    print(f"输出目录: {output_dir}")

    if not os.path.exists(raw_dir):
        print(f"\n错误: 未找到 CREMA-D 数据目录 {raw_dir}")
        print("\n请先下载数据集:")
        print("  git clone https://github.com/CheyneyComputerScience/CREMA-D.git ./data/crema-d")
        print("或")
        print("  下载并解压到 ./data/crema-d/ 目录")
        return

    wav_files = [f for f in os.listdir(raw_dir) if f.endswith('.wav')]
    print(f"找到 {len(wav_files)} 个音频文件")

    actor_files = {}
    for f in wav_files:
        try:
            parts = f.replace('.wav', '').split('_')
            actor_id = parts[0]
            emotion_code = parts[2]

            if emotion_code not in emotion_map:
                continue

            if actor_id not in actor_files:
                actor_files[actor_id] = []
            actor_files[actor_id].append((f, emotion_map[emotion_code]))
        except:
            continue

    actor_ids = sorted(actor_files.keys())
    rnd.seed(42)
    rnd.shuffle(actor_ids)

    n_actors = len(actor_ids)
    train_cut = int(n_actors * 0.7)
    val_cut = int(n_actors * 0.85)

    train_actors = actor_ids[:train_cut]
    val_actors = actor_ids[train_cut:val_cut]
    test_actors = actor_ids[val_cut:]

    print(f"总演员数: {n_actors}")
    print(f"  - 训练集演员: {len(train_actors)} ({', '.join(train_actors[:3])}...)")
    print(f"  - 验证集演员: {len(val_actors)} ({', '.join(val_actors[:3])}...)")
    print(f"  - 测试集演员: {len(test_actors)} ({', '.join(test_actors[:3])}...)")

    actor_split = {}
    for aid in train_actors:
        actor_split[aid] = 'train'
    for aid in val_actors:
        actor_split[aid] = 'val'
    for aid in test_actors:
        actor_split[aid] = 'test'

    count = {'train': 0, 'val': 0, 'test': 0}

    for actor_id, file_list in actor_files.items():
        split = actor_split[actor_id]
        for filename, emotion in file_list:
            dst_dir = os.path.join(output_dir, split, emotion)
            os.makedirs(dst_dir, exist_ok=True)

            src = os.path.join(raw_dir, filename)
            dst = os.path.join(dst_dir, filename)
            import shutil
            shutil.copy2(src, dst)
            count[split] += 1

    print(f"\n数据分配完成:")
    for s, c in count.items():
        print(f"  - {s}: {c} 个样本")
    print(f"  - 总计: {sum(count.values())} 个样本")
    print(f"\n现在可以运行训练:")
    print(f"  python main.py train --epochs 50")


def download_savee():
    print("\n" + "="*60)
    print("SAVEE 数据集下载说明")
    print("="*60)

    print("""
数据集信息:
- 名称: Surrey Audio-Visual Expressed Emotion (SAVEE)
- 内容: 4名男性演员的语音情感数据
- 情感类别: anger, disgust, fear, happy, sad, surprise, neutral
- 格式: WAV文件 (16-bit)
- 数量: 约480个音频文件

下载地址:
http://personal.ee.surrey.ac.uk/Personal/P.Jackson/SAVEE/

使用方法:
1. 从网站下载数据
2. 解压到 ./data/raw/ 目录
3. 运行 python prepare_savee.py 进行数据预处理
""")


def generate_sample_data():
    import numpy as np
    import soundfile as sf

    print("\n生成示例数据用于测试...")
    print("(这将生成一些随机噪声作为占位符)")

    sample_rate = config.SAMPLE_RATE if 'config' in sys.modules else 16000
    duration = 3.0
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad']
    splits = ['train', 'val', 'test']

    samples_per_split = {
        'train': 10,
        'val': 5,
        'test': 5
    }

    for split in splits:
        for emotion in emotions:
            dir_path = os.path.join('./data', split, emotion)
            os.makedirs(dir_path, exist_ok=True)

            for i in range(samples_per_split[split]):
                np.random.seed((i + abs(hash(emotion))) % (2**31 - 1))
                audio = np.random.randn(int(sample_rate * duration)) * 0.1

                filename = f'sample_{emotion}_{i:03d}.wav'
                filepath = os.path.join(dir_path, filename)
                sf.write(filepath, audio, sample_rate)

    print(f"\n✓ 已生成示例数据:")
    for split in splits:
        count = samples_per_split[split] * len(emotions)
        print(f"  - {split}: {count} 个样本")

    print("\n注意: 这些是随机生成的测试数据，仅用于验证代码流程。")
    print("实际训练请使用真实情感语音数据集（如RAVDESS、SAVEE等）。")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='数据准备工具')
    subparsers = parser.add_subparsers(dest='command')

    subparsers.add_parser('init', help='创建数据目录结构')
    subparsers.add_parser('prepare-cremad', help='预处理CREMA-D数据集')
    subparsers.add_parser('download-ravdess', help='显示RAVDESS下载信息')
    subparsers.add_parser('download-savee', help='显示SAVEE下载信息')
    subparsers.add_parser('generate-samples', help='生成示例数据用于测试')

    args = parser.parse_args()

    try:
        import config
    except:
        pass

    if args.command == 'init':
        create_data_structure()
    elif args.command == 'prepare-cremad':
        prepare_cremad()
    elif args.command == 'download-ravdess':
        download_ravdess()
    elif args.command == 'download-savee':
        download_savee()
    elif args.command == 'generate-samples':
        generate_sample_data()
    else:
        parser.print_help()
        print("\n常用命令:")
        print("  python data_utils.py init              # 创建目录结构")
        print("  python data_utils.py download-ravdess   # 下载RAVDESS数据集")
        print("  python data_utils.py generate-samples   # 生成测试数据")