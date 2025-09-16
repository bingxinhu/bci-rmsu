import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import mne
from mne.io import read_raw_gdf
from mne.channels import make_standard_montage
import warnings
warnings.filterwarnings('ignore')

# 随机种子固定（保证结果可复现）
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------- 1. 数据集核心配置 --------------------------
BCI_EVENT_CONFIG = {
    '2a': {
        'train': {
            'left_hand': '769',
            'right_hand': '770',
            'feet': '771',
            'tongue': '772'
        },
        'test': {
            'mi_event': '783'
        },
        'num_classes': 4
    },
    '2b': {
        'train': {
            'left_hand': '769',
            'right_hand': '770'
        },
        'test': {
            'left_hand': '769',
            'right_hand': '770'
        },
        'num_classes': 2
    }
}

BCI_CHANNEL_MAPPING = {
    'EEG-Fz': 'Fz', 'EEG-C3': 'C3', 'EEG-Cz': 'Cz', 'EEG-C4': 'C4', 'EEG-Pz': 'Pz',
    'EEG-0': 'FC5', 'EEG-1': 'FC3', 'EEG-2': 'FC1', 'EEG-3': 'FCz', 'EEG-4': 'FC2',
    'EEG-5': 'FC4', 'EEG-6': 'FC6', 'EEG-7': 'C5', 'EEG-8': 'C1', 'EEG-9': 'C2',
    'EEG-10': 'C6', 'EEG-11': 'CP5', 'EEG-12': 'CP3', 'EEG-13': 'CP1', 'EEG-14': 'CPz',
    'EEG-15': 'CP2', 'EEG-16': 'CP4', 'EEG-17': 'CP6', 'EEG-18': 'F5', 'EEG-19': 'F3',
    'EEG-20': 'F1', 'EEG-21': 'F2', 'EEG-22': 'F4', 'EEG-23': 'F6', 'EEG-24': 'T7',
    'EOG-left': 'EOG-left', 'EOG-central': 'EOG-central', 'EOG-right': 'EOG-right'
}

# -------------------------- 2. ResNet 基本块 --------------------------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
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
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

# -------------------------- 3. RWKV 核心单元 --------------------------
class RWKV(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=1, dropout=0.1):
        super(RWKV, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # RWKV层
        self.layers = nn.ModuleList([
            RWKVLayer(hidden_dim, dropout) for _ in range(num_layers)
        ])
        
        # 移除输出投影层，不再将维度还原为input_dim
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        x = self.input_proj(x)  # (batch_size, seq_len, hidden_dim)
        
        for layer in self.layers:
            x = layer(x)
            
        # 移除输出投影，保持hidden_dim维度
        return x

class RWKVLayer(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super(RWKVLayer, self).__init__()
        self.hidden_dim = hidden_dim
        
        # 时间混合参数
        self.time_mix_k = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.time_mix_v = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.time_mix_r = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # 关键权重
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.receptance = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 输出投影
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 层归一化和dropout
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        batch_size, seq_len, _ = x.shape
        
        # 时间混合
        xk = x * self.time_mix_k + (1 - self.time_mix_k) * x.mean(dim=1, keepdim=True)
        xv = x * self.time_mix_v + (1 - self.time_mix_v) * x.mean(dim=1, keepdim=True)
        xr = x * self.time_mix_r + (1 - self.time_mix_r) * x.mean(dim=1, keepdim=True)
        
        # 计算关键、值和接受度
        k = torch.relu(self.key(xk))
        v = torch.relu(self.value(xv))
        r = torch.sigmoid(self.receptance(xr))
        
        # RWKV核心计算
        wkv = self.rwkv(k, v)
        
        # 输出
        out = r * wkv
        out = self.output(out)
        
        # 残差连接和层归一化
        out = self.ln1(x + self.dropout(out))
        
        # 前馈网络
        ff_out = torch.relu(self.ln2(out))
        ff_out = self.dropout(ff_out)
        
        # 最终输出
        out = out + ff_out
        
        return out
    
    def rwkv(self, k, v):
        # 简化的RWKV计算
        batch_size, seq_len, hidden_dim = k.shape
        
        # 初始化累积量
        wkv = torch.zeros_like(k)
        alpha = torch.zeros(batch_size, hidden_dim, device=k.device)
        
        # 递归计算
        for t in range(seq_len):
            k_t = k[:, t, :]
            v_t = v[:, t, :]
            
            # 计算当前时间步的wkv
            wkv_t = torch.sigmoid(k_t) * v_t + alpha
            wkv[:, t, :] = wkv_t
            
            # 更新alpha
            alpha = torch.sigmoid(k_t) * v_t + 0.9 * alpha  # 0.9是衰减因子
        
        return wkv

# -------------------------- 4. 完整EEG MI模型 --------------------------
class EEGMI_RWKV_ResNet_Model(nn.Module):
    def __init__(self, num_channels, num_classes, seq_len, num_bands=5, 
                 hidden_dim=128, num_rwkv_layers=3, num_resnet_blocks=2):
        super().__init__()
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.seq_len = seq_len
        
        # 频带提取
        self.band_extractor = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels * num_bands,
            kernel_size=5,
            padding=2,
            groups=num_channels
        )
        
        # 空间注意力
        self.spatial_attn = nn.Sequential(
            nn.Linear(num_channels, num_channels),
            nn.Softmax(dim=1)
        )
        
        # ResNet特征提取
        self.resnet_blocks = self._make_resnet_layers(
            num_channels * num_bands, num_channels * num_bands, num_resnet_blocks
        )
        
        # RWKV时序建模
        self.rwkv = RWKV(
            input_dim=num_channels * num_bands,
            hidden_dim=hidden_dim,
            num_layers=num_rwkv_layers
        )
        
        # 添加自适应池化层来处理可变长度的序列
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def _make_resnet_layers(self, in_planes, planes, num_blocks):
        layers = []
        for i in range(num_blocks):
            layers.append(BasicBlock(in_planes, planes))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # 输入形状: (batch_size, seq_len, num_channels)
        batch_size = x.size(0)
        x = x.permute(0, 2, 1)  # (batch_size, num_channels, seq_len)
        
        # 频带提取
        bands = self.band_extractor(x)  # (batch_size, num_channels * num_bands, seq_len)
        
        # 空间注意力
        bands_reshaped = bands.view(batch_size, self.num_channels, self.num_bands, self.seq_len)
        attn_weights = self.spatial_attn(torch.mean(bands_reshaped, dim=(2, 3)))
        attn_weights = attn_weights.unsqueeze(2).unsqueeze(3)
        bands_reshaped = bands_reshaped * attn_weights
        bands = bands_reshaped.view(batch_size, self.num_channels * self.num_bands, self.seq_len)
        
        # ResNet特征提取
        features = self.resnet_blocks(bands)  # (batch_size, num_channels * num_bands, seq_len)
        
        # RWKV时序建模
        features = features.permute(0, 2, 1)  # (batch_size, seq_len, num_channels * num_bands)
        features = self.rwkv(features)  # (batch_size, seq_len, hidden_dim)
        
        # 使用自适应池化而不是展平
        features = features.permute(0, 2, 1)  # (batch_size, hidden_dim, seq_len)
        features = self.adaptive_pool(features)  # (batch_size, hidden_dim, 1)
        features = features.squeeze(2)  # (batch_size, hidden_dim)
        
        # 分类
        return self.classifier(features)

# -------------------------- 5. BCI数据集类 --------------------------
class BCI_Dataset(Dataset):
    def __init__(self, data_path, subject, dataset_type='2a', 
                 tmin=0.5, tmax=2.5, sfreq=250):
        self.data_path = data_path
        self.subject = subject
        self.dataset_type = dataset_type
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq
        
        self.event_config = BCI_EVENT_CONFIG[dataset_type]
        self.train_event_desc = self.event_config['train']
        self.test_event_desc = self.event_config['test']
        self.num_classes = self.event_config['num_classes']
        
        if dataset_type == '2a':
            self.data_dir = os.path.join(data_path, 'BCICIV_2a_gdf')
            self.train_runs = [4, 8, 12]
            self.test_runs = [6, 10, 14]
        else:
            self.data_dir = os.path.join(data_path, 'BCICIV_2b_gdf')
            self.train_runs = [3, 7, 11]
            self.test_runs = [4, 8, 12]
        
        if not os.path.exists(self.data_dir):
            raise ValueError(
                f"数据集目录不存在：{self.data_dir}\n"
                f"请按格式放置数据：\n"
                f"../data/BCICIV_2a_gdf/A01T.gdf （训练）\n"
                f"../data/BCICIV_2a_gdf/A01E.gdf （测试）"
            )
        
        self.X_train, self.y_train = self.load_data(runs=self.train_runs, is_train=True)
        self.X_test, self.y_test = self.load_data(runs=self.test_runs, is_train=False)
        
        if len(self.X_train) == 0:
            raise ValueError(
                f"未加载到训练数据！\n"
                f"请确认 {self.data_dir} 下有文件 A{subject:02d}T.gdf（2a）或 B{subject:02d}03T.gdf（2b）"
            )
        if len(self.X_test) == 0:
            raise ValueError(
                f"未加载到测试数据！\n"
                f"请确认 {self.data_dir} 下有文件 A{subject:02d}E.gdf（2a）或 B{subject:02d}04T.gdf（2b）"
            )
        
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.2, random_state=SEED, stratify=self.y_train
        )
        
        self._normalize_data()

    def _rename_and_filter_channels(self, raw):
        current_channels = raw.ch_names
        print(f"原始通道（共{len(current_channels)}个）: {current_channels[:5]}...")
        
        new_ch_names = []
        for ch in current_channels:
            ch_clean = ch.strip()
            new_ch = BCI_CHANNEL_MAPPING.get(ch_clean, ch_clean)
            new_ch_names.append(new_ch)
        raw.rename_channels(dict(zip(current_channels, new_ch_names)))
        
        eeg_channels = [ch for ch in raw.ch_names if not ch.startswith('EOG-')]
        raw.pick(eeg_channels)
        
        raw.set_channel_types({ch: 'eeg' for ch in eeg_channels})
        montage = make_standard_montage('standard_1005')
        raw.set_montage(montage, on_missing='ignore')
        
        print(f"过滤后EEG通道（共{len(eeg_channels)}个）: {eeg_channels[:5]}...")
        return raw

    def load_data(self, runs, is_train):
        all_epochs = []
        all_labels = []
        processed_files = set()
        
        for run in runs:
            try:
                if self.dataset_type == '2a':
                    file_prefix = f"A{self.subject:02d}"
                    file_suffix = 'T' if is_train else 'E'
                    filename = f"{file_prefix}{file_suffix}.gdf"
                else:
                    filename = f"B{self.subject:02d}{run:02d}T.gdf"
                
                file_path = os.path.join(self.data_dir, filename)
                if file_path in processed_files:
                    print(f"⚠️  文件已处理：{filename}，跳过重复处理")
                    continue
                processed_files.add(file_path)
                
                if not os.path.exists(file_path):
                    print(f"⚠️  文件不存在：{filename}，跳过该Run")
                    continue
                
                print(f"\n✅ 处理{'训练' if is_train else '测试'}文件：{filename}")
                raw = read_raw_gdf(file_path, preload=True, stim_channel='auto')
                
                raw = self._rename_and_filter_channels(raw)
                
                raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge')
                
                events, event_ids = mne.events_from_annotations(raw)
                print(f"文件内事件描述: {list(event_ids.keys())[:5]}...（共{len(event_ids)}种）")
                
                target_event_desc = self.train_event_desc if is_train else self.test_event_desc
                target_events = {}
                for event_name, desc_str in target_event_desc.items():
                    if desc_str in event_ids:
                        target_events[event_name] = event_ids[desc_str]
                        print(f"✅ 找到事件：{event_name}（描述：{desc_str} → MNE ID：{event_ids[desc_str]}）")
                    else:
                        print(f"⚠️  未找到事件：{event_name}（需描述：{desc_str}）")
                
                if not target_events:
                    print(f"❌ 该文件无目标事件，跳过")
                    continue
                
                picks = mne.pick_types(raw.info, eeg=True, exclude='bads')
                epochs = mne.Epochs(
                    raw, events, target_events,
                    tmin=self.tmin, tmax=self.tmax,
                    proj=True, picks=picks, baseline=None, preload=True, on_missing='ignore'
                )
                
                if len(epochs) == 0:
                    print(f"⚠️  未提取到Epochs，跳过")
                    continue
                print(f"✅ 提取到 {len(epochs)} 个Epochs")
                
                epochs_data = epochs.get_data().transpose(0, 2, 1)
                if is_train:
                    label_map = {mne_id: idx for idx, (name, mne_id) in enumerate(target_events.items())}
                    epoch_labels = [label_map[event_id] for event_id in epochs.events[:, 2]]
                else:
                    if self.dataset_type == '2a':
                        samples_per_class = len(epochs_data) // self.num_classes
                        epoch_labels = []
                        for cls_idx in range(self.num_classes):
                            epoch_labels.extend([cls_idx] * samples_per_class)
                        epoch_labels = epoch_labels[:len(epochs_data)]
                        print(f"⚠️  BCI 2a测试集标签：按均匀分布分配（每类{samples_per_class}个）")
                    else:
                        label_map = {mne_id: idx for idx, (name, mne_id) in enumerate(target_events.items())}
                        epoch_labels = [label_map[event_id] for event_id in epochs.events[:, 2]]
                
                epoch_labels = np.array(epoch_labels)
                print(f"✅ 标签分布：{np.bincount(epoch_labels)}")
                
                all_epochs.append(epochs_data)
                all_labels.append(epoch_labels)
            
            except Exception as e:
                print(f"❌ 处理{filename}出错：{str(e)[:100]}...，跳过")
                continue
        
        if not all_epochs:
            return np.array([]), np.array([])
        
        X = np.concatenate(all_epochs, axis=0)
        y = np.concatenate(all_labels, axis=0)
        print(f"\n📊 {'训练' if is_train else '测试'}数据加载完成：共{len(X)}个样本")
        return X, y

    def _normalize_data(self):
        self.mean = np.mean(self.X_train, axis=(0, 1))
        self.std = np.std(self.X_train, axis=(0, 1)) + 1e-8
        
        self.X_train = (self.X_train - self.mean) / self.std
        self.X_val = (self.X_val - self.mean) / self.std
        self.X_test = (self.X_test - self.mean) / self.std

    def get_dataloaders(self, batch_size=32):
        train_ds = TensorDataset(torch.FloatTensor(self.X_train), torch.LongTensor(self.y_train))
        val_ds = TensorDataset(torch.FloatTensor(self.X_val), torch.LongTensor(self.y_val))
        test_ds = TensorDataset(torch.FloatTensor(self.X_test), torch.LongTensor(self.y_test))
        
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        )

# -------------------------- 6. 辅助数据集类 --------------------------
class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]
    def __len__(self):
        return self.data_tensor.size(0)

# -------------------------- 7. 训练/评估/可视化函数 --------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, num_epochs=50, model_save_path='best_model.pth'):
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_preds, train_labels = 0.0, [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        model.eval()
        val_loss, val_preds, val_labels = 0.0, [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        # 计算平均损失和准确率
        train_loss /= len(train_loader.dataset)
        val_loss /= len(val_loader.dataset)
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        
        # 保存历史记录
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 打印 epoch 信息
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f'保存最佳模型 (Val Acc: {best_val_acc:.4f})')
        
        # 学习率调度
        scheduler.step(val_loss)
    
    return history

def evaluate_model(model, test_loader, device, class_names=None):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算指标
    accuracy = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    
    print(f'Test Accuracy: {accuracy:.4f}')
    print('\nConfusion Matrix:')
    print(cm)
    print('\nClassification Report:')
    print(report)
    
    return accuracy, cm, report

def plot_history(history, save_path='training_history.png'):
    plt.figure(figsize=(12, 4))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f'训练历史图已保存至 {save_path}')

# -------------------------- 8. 主函数 --------------------------
def main(args):
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据集
    print(f'加载数据集: {args.dataset_type} 受试者 {args.subject}')
    dataset = BCI_Dataset(
        data_path=args.data_path,
        subject=args.subject,
        dataset_type=args.dataset_type,
        tmin=args.tmin,
        tmax=args.tmax
    )
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=args.batch_size)
    
    # 获取数据维度信息
    sample, _ = next(iter(train_loader))
    seq_len, num_channels = sample.shape[1], sample.shape[2]
    num_classes = dataset.num_classes
    print(f'数据维度: 序列长度={seq_len}, 通道数={num_channels}, 类别数={num_classes}')
    
    # 初始化模型
    model = EEGMI_RWKV_ResNet_Model(
        num_channels=num_channels,
        num_classes=num_classes,
        seq_len=seq_len,
        num_bands=args.num_bands,
        hidden_dim=args.hidden_dim,
        num_rwkv_layers=args.num_rwkv_layers,
        num_resnet_blocks=args.num_resnet_blocks
    ).to(device)
    
    # 优化器和损失函数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    model_save_path = os.path.join(args.save_dir, f'best_model_{args.dataset_type}_s{args.subject}.pth')
    
    # 训练模型
    print('\n开始训练...')
    history = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler,
        device, num_epochs=args.epochs, model_save_path=model_save_path
    )
    
    # 绘制训练历史
    plot_history(history, save_path=os.path.join(args.save_dir, f'history_{args.dataset_type}_s{args.subject}.png'))
    
    # 评估最佳模型
    print('\n评估最佳模型...')
    model.load_state_dict(torch.load(model_save_path))
    class_names = list(BCI_EVENT_CONFIG[args.dataset_type]['train'].keys())
    evaluate_model(model, test_loader, device, class_names=class_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='EEG MI Classification with RWKV and ResNet')
    parser.add_argument('--data_path', type=str, default='../data', help='数据集根目录')
    parser.add_argument('--save_dir', type=str, default='results', help='结果保存目录')
    parser.add_argument('--dataset_type', type=str, default='2a', choices=['2a', '2b'], help='数据集类型')
    parser.add_argument('--subject', type=int, default=1, help='受试者编号')
    parser.add_argument('--tmin', type=float, default=0.5, help='事件开始后提取数据的起始时间')
    parser.add_argument('--tmax', type=float, default=2.5, help='事件开始后提取数据的结束时间')
    parser.add_argument('--batch_size', type=int, default=32, help='批处理大小')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--num_bands', type=int, default=5, help='频带数量')
    parser.add_argument('--hidden_dim', type=int, default=128, help='RWKV隐藏层维度')
    parser.add_argument('--num_rwkv_layers', type=int, default=3, help='RWKV层数')
    parser.add_argument('--num_resnet_blocks', type=int, default=2, help='ResNet块数量')
    
    args = parser.parse_args()
    main(args)
