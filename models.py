import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from transformers import Wav2Vec2Model


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SpeechEmotionModel(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES, num_lstm_layers=2):
        super(SpeechEmotionModel, self).__init__()

        self.mfcc_proj = nn.Conv2d(1, 32, 1)
        self.mel_proj = nn.Conv2d(1, 32, 1)
        self.chroma_proj = nn.Conv2d(1, 32, 1)

        self.stem = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, None))

        lstm_input = 256

        self.lstm = nn.LSTM(
            input_size=lstm_input,
            hidden_size=256,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=0.4 if num_lstm_layers > 1 else 0,
            bidirectional=True
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256, num_classes)
        )

        self._init_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, mfcc, mel_spec=None, chroma=None):
        batch_size = mfcc.size(0)

        x1 = F.interpolate(mfcc, size=(128, 94), mode='bilinear', align_corners=False)
        x1 = self.mfcc_proj(x1)

        if mel_spec is not None:
            x2 = F.interpolate(mel_spec, size=(128, 94), mode='bilinear', align_corners=False)
            x2 = self.mel_proj(x2)
        else:
            x2 = torch.zeros(batch_size, 32, 128, 94, device=mfcc.device)

        if chroma is not None:
            x3 = F.interpolate(chroma, size=(128, 94), mode='bilinear', align_corners=False)
            x3 = self.chroma_proj(x3)
        else:
            x3 = torch.zeros(batch_size, 32, 128, 94, device=mfcc.device)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.squeeze(2)
        x = x.transpose(1, 2)

        lstm_out, _ = self.lstm(x)
        x = torch.cat([lstm_out[:, -1, :256], lstm_out[:, 0, 256:]], dim=1)

        out = self.classifier(x)
        return out


class EmotionCNN(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(EmotionCNN, self).__init__()

        self.mfcc_proj = nn.Conv2d(1, 32, 1)
        self.mel_proj = nn.Conv2d(1, 32, 1)
        self.chroma_proj = nn.Conv2d(1, 32, 1)

        self.stem = nn.Sequential(
            nn.Conv2d(96, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 1)),
            nn.Flatten(),
            nn.Linear(256 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = [ResidualBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_c, out_c, 1))
        return nn.Sequential(*layers)

    def forward(self, mfcc, mel_spec=None, chroma=None):
        b = mfcc.size(0)
        x1 = F.interpolate(mfcc, size=(128, 94), mode='bilinear', align_corners=False)
        x1 = self.mfcc_proj(x1)
        if mel_spec is not None:
            x2 = F.interpolate(mel_spec, size=(128, 94), mode='bilinear', align_corners=False)
            x2 = self.mel_proj(x2)
        else:
            x2 = torch.zeros(b, 32, 128, 94, device=mfcc.device)
        if chroma is not None:
            x3 = F.interpolate(chroma, size=(128, 94), mode='bilinear', align_corners=False)
            x3 = self.chroma_proj(x3)
        else:
            x3 = torch.zeros(b, 32, 128, 94, device=mfcc.device)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.classifier(x)


class EmotionLSTM(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(EmotionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=248,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(1)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        x = torch.cat([lstm_out[:, -1, :256], lstm_out[:, 0, 256:]], dim=1)
        return self.classifier(x)


HybridCNNLSTM = SpeechEmotionModel


class Wav2Vec2EmotionModel(nn.Module):
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(Wav2Vec2EmotionModel, self).__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(config.WAV2VEC2_MODEL_NAME)

        hidden_size = self.wav2vec2.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )

        self._init_classifier()

    def _init_classifier(self):
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, audio):
        outputs = self.wav2vec2(audio)
        x = outputs.last_hidden_state.mean(dim=1)
        return self.classifier(x)


def get_model(model_type='hybrid', num_classes=config.NUM_CLASSES):
    models_map = {
        'cnn': EmotionCNN(num_classes=num_classes),
        'lstm': EmotionLSTM(num_classes=num_classes),
        'hybrid': SpeechEmotionModel(num_classes=num_classes),
        'wav2vec2': Wav2Vec2EmotionModel(num_classes=num_classes),
    }

    if model_type not in models_map:
        raise ValueError(f"Unknown model type: {model_type}")

    return models_map[model_type].to(config.DEVICE)


if __name__ == '__main__':
    model = get_model('hybrid')
    print(f"Model architecture:\n{model}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")
