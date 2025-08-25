# -*- coding: utf-8 -*-
"""
封装 1D-CNN, WDCNN, ResNet18/50-1D, TransformerLSTM, WaveletKernelNet
（SVM+小波在 compare.py 里单独实现）
"""
import math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torchvision.models as tvm
import pywt


# ---------- 1. 1D-CNN ----------
class CNN1D(nn.Module):
    def __init__(self, num_cls=10):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2))
        self.fc = nn.Linear(32*256, num_cls)

    def forward(self, x):
        x = self.backbone(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# ---------- 2. WDCNN ----------
class WDCNN(nn.Module):
    def __init__(self, num_cls=10, input_len=1024):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=64, stride=16, padding=24)
        self.pool  = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(16, 32, 3, padding=1)
        
        # 动态计算特征维度
        with torch.no_grad():
            x = torch.randn(1, 1, input_len)
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            fc_input_dim = x.view(1, -1).size(1)
        
        self.fc = nn.Linear(fc_input_dim, num_cls)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# ---------- 3. ResNet1D ----------
def ResNet1D(model_type='18', num_cls=10):
    backbone = tvm.resnet18() if model_type == '18' else tvm.resnet50()
    backbone.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    backbone.fc = nn.Linear(backbone.fc.in_features, num_cls)
    return backbone


# ---------- 4. Transformer + LSTM ----------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2)*-(math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerLSTM(nn.Module):
    def __init__(self, num_cls=10, seq_len=1024, d_model=64, nhead=4):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos   = PositionalEncoding(d_model, seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*2)
        self.encoder = nn.TransformerEncoder(encoder_layer, 2)
        self.lstm = nn.LSTM(d_model, 128, bidirectional=True, batch_first=True)
        self.fc   = nn.Linear(256, num_cls)

    def forward(self, x):
        x = x.transpose(1,2)            # [B, L, 1]
        x = self.embed(x)               # [B, L, d]
        x = self.pos(x)
        x = self.encoder(x)
        _, (h, _) = self.lstm(x)
        feat = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(feat)


# ---------- 5. WaveletKernelNet ----------
class WaveletKernelConv(nn.Module):
    def __init__(self, wavelet='morl', scales=np.arange(1, 17)):
        super().__init__()
        wav = pywt.ContinuousWavelet(wavelet)
        kernels = []
        for s in scales:
            # 修复：使用正确的API调用方式
            t, psi = wav.wavefun(length=1024)  # 或者使用固定长度
            # 对小波进行缩放处理
            t = t / s  # 手动缩放时间轴
            # 截取中心部分并归一化
            psi = np.expand_dims(psi, axis=0)    # [1, len]
            psi = psi / np.linalg.norm(psi)
            kernels.append(torch.tensor(psi, dtype=torch.float32))
        max_len = max(k.shape[-1] for k in kernels)
        # pad 到相同长度
        kernels = [F.pad(k, (0, max_len-k.size(-1))) for k in kernels]
        weight  = torch.stack(kernels, dim=0)     # [M,1,L]
        self.conv = nn.Conv1d(1, len(scales), max_len, bias=False)
        self.conv.weight = nn.Parameter(weight, requires_grad=False)
        self.fc = nn.Linear(len(scales)*(1024-max_len+1), 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


class BasicBlock1D(nn.Module):
    expansion = 1
    
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(self.expansion * planes)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet1D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet1D, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def ResNet18_1D(num_cls=10):
    return ResNet1D(BasicBlock1D, [2, 2, 2, 2], num_cls)

def ResNet50_1D(num_cls=10):
    return ResNet1D(BasicBlock1D, [3, 4, 6, 3], num_cls)

# ---------- InceptionTime ----------
class _Incep1DBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.b1 = nn.Conv1d(in_ch, out_ch // 4, 1)
        self.b3 = nn.Conv1d(in_ch, out_ch // 4, 3, padding=1)
        self.b5 = nn.Conv1d(in_ch, out_ch // 4, 5, padding=2)
        self.bp = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, out_ch // 4, 1))
    def forward(self, x):
        x = torch.cat([self.b1(x), self.b3(x), self.b5(x), self.bp(x)], dim=1)
        return F.relu(x)

class InceptionTime1D(nn.Module):
    def __init__(self, num_cls=10):
        super().__init__()
        self.stem = nn.Conv1d(1, 32, 3, padding=1)
        self.inc1 = _Incep1DBlock(32, 64)
        self.inc2 = _Incep1DBlock(64, 128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Linear(128, num_cls)
    def forward(self, x):
        x = F.relu(self.stem(x))
        x = self.inc1(x); x = self.inc2(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# ---------- CNN‑LSTM ----------
class CNN_LSTM(nn.Module):
    def __init__(self, num_cls=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU())
        self.lstm = nn.LSTM(64, 128, bidirectional=True, batch_first=True)
        self.fc   = nn.Linear(256, num_cls)
    def forward(self, x):
        x = self.conv(x)           # [B, 64, 512]
        x = x.permute(0, 2, 1)     # [B, 512, 64]
        _, (h, _) = self.lstm(x)   # h: [2, B, 128]
        feat = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc(feat)

# ---------- VMD + CNN ----------
try:
    import vmdpy
    class VMD_CNN(nn.Module):
        def __init__(self, num_cls=10, K=4):
            super().__init__()
            self.K = K
            self.cnn = nn.Sequential(
                nn.Conv1d(K, 64, 5, padding=2), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1))
            self.fc = nn.Linear(128, num_cls)

        def vmd_batch(self, x):
            out = []
            for sig in x.squeeze(1).cpu().numpy():
                u, _, _ = vmdpy.VMD(sig, alpha=2000, tau=0.0, K=self.K, DC=0, init=1, tol=1e-7)
                out.append(u)
            u = torch.tensor(np.stack(out), dtype=torch.float32, device=x.device)
            return u

        def forward(self, x):
            x = self.vmd_batch(x)
            x = self.cnn(x).squeeze(-1)
            return self.fc(x)
except ImportError:
    print("Warning: vmdpy not installed, VMD_CNN will not be available")
    VMD_CNN = None