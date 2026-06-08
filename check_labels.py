"""检查数据集的labels"""
import os
import sys

# 添加项目目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_processor import EmotionDataset
import config

def check_labels():
    print("=" * 60)
    print("检查数据集标签")
    print("=" * 60)

    # 创建数据集
    train_dataset = EmotionDataset(data_dir=config.DATA_DIR, mode='train', transform=False)

    print(f"\n总共加载了 {len(train_dataset)} 个样本")
    print(f"\n情感类别映射 (config.EMOTIONS): {config.EMOTIONS}")
    print(f"类别数量: {config.NUM_CLASSES}")

    # 统计每个类别的样本数
    label_counts = {}
    for i in range(len(train_dataset)):
        label = train_dataset.labels[i]
        label_counts[label] = label_counts.get(label, 0) + 1

    print("\n" + "-" * 60)
    print("各类别样本数量统计:")
    print("-" * 60)
    for idx, emotion in enumerate(config.EMOTIONS):
        count = label_counts.get(idx, 0)
        print(f"  标签 {idx} ({emotion:8s}): {count} 个样本")

    # 打印前20个样本的详细信息
    print("\n" + "-" * 60)
    print("前20个样本的详细信息:")
    print("-" * 60)
    for i in range(min(20, len(train_dataset))):
        file_path = train_dataset.audio_files[i]
        label = train_dataset.labels[i]
        emotion = config.EMOTIONS[label]
        print(f"  样本 {i+1:3d}: label={label}, emotion={emotion:8s}, file={os.path.basename(file_path)}")

    # 随机抽取10个样本检查
    print("\n" + "-" * 60)
    print("随机抽取10个样本检查:")
    print("-" * 60)
    import random
    random.seed(42)
    indices = random.sample(range(len(train_dataset)), min(10, len(train_dataset)))
    for i in indices:
        file_path = train_dataset.audio_files[i]
        label = train_dataset.labels[i]
        emotion = config.EMOTIONS[label]
        print(f"  索引 {i:4d}: label={label}, emotion={emotion:8s}")

    # 检查验证集和测试集
    print("\n" + "=" * 60)
    print("检查验证集和测试集")
    print("=" * 60)
    val_dataset = EmotionDataset(data_dir=config.DATA_DIR, mode='val', transform=False)
    test_dataset = EmotionDataset(data_dir=config.DATA_DIR, mode='test', transform=False)

    print(f"训练集: {len(train_dataset)} 个样本")
    print(f"验证集: {len(val_dataset)} 个样本")
    print(f"测试集: {len(test_dataset)} 个样本")

    # 检查标签范围
    all_labels = set(train_dataset.labels + val_dataset.labels + test_dataset.labels)
    print(f"\n所有标签值: {sorted(all_labels)}")
    print(f"标签范围: 0 到 {max(all_labels)} (应该是 0 到 5)")

    if all_labels != set(range(6)):
        print("\n⚠️ 警告: 标签不完整，缺少某些类别！")
    else:
        print("\n✓ 标签完整，包含所有6个类别")

if __name__ == '__main__':
    check_labels()
