import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pywt
import mne
from mne.io import read_raw_gdf
from mne.channels import make_standard_montage
import warnings
warnings.filterwarnings('ignore')

# 随机种子固定
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

# -------------------------- 1. 数据集核心配置 --------------------------
BCI_EVENT_CONFIG = {
    '2a': {
        'train': {'left_hand': '769', 'right_hand': '770', 'feet': '771', 'tongue': '772'},
        'test': {'mi_event': '783'},
        'num_classes': 4,
        'target_classes': ['right_hand', 'feet'],
        'target_labels': [1, 2]
    },
    '2b': {
        'train': {'left_hand': '769', 'right_hand': '770'},
        'test': {'left_hand': '769', 'right_hand': '770'},
        'num_classes': 2,
        'target_classes': ['right_hand'],
        'target_labels': [1]
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

# -------------------------- 2. 工具函数（小波变换） --------------------------
def eeg_to_time_freq(eeg_data, wavelet='db4', level=4):
    seq_len, channels = eeg_data.shape
    time_freq_features = []
    for ch in range(channels):
        coeffs = pywt.wavedec(eeg_data[:, ch], wavelet, level=level)
        detail_coeffs = np.stack([np.pad(c, (0, seq_len - len(c)), mode='edge') for c in coeffs[1:]], axis=1)
        time_freq_features.append(detail_coeffs)
    return np.stack(time_freq_features, axis=0)

# -------------------------- 3. RMSU核心单元 --------------------------
class RMSU(nn.Module):
    def __init__(self, input_dim, state_dim=64, alpha=0.1, gamma=0.05, window_size=100):
        super().__init__()
        self.input_dim = input_dim  # 保存输入维度用于校验
        self.state_dim = state_dim
        self.alpha = alpha
        self.gamma = gamma
        self.window_size = window_size
        
        self.W_g = nn.Linear(input_dim, state_dim)
        self.U_g = nn.Linear(state_dim, state_dim)
        self.b_g = nn.Parameter(torch.zeros(state_dim))
        
        self.W_q = nn.Linear(input_dim, state_dim)
        self.W_v = nn.Linear(input_dim, state_dim)
        self.A = nn.Parameter(torch.randn(state_dim))
        self.B = nn.Linear(input_dim, state_dim)
        self.W_o = nn.Linear(state_dim, input_dim)
        
        # 权重初始化
        nn.init.xavier_uniform_(self.W_g.weight)
        nn.init.xavier_uniform_(self.U_g.weight)
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.B.weight)
        nn.init.xavier_uniform_(self.W_o.weight)
    
    def forward(self, x, prev_state=None, prev_attn=None, prev_mu=None):
        batch_size = x.size(0)
        if prev_state is None:
            prev_state = torch.zeros(batch_size, self.state_dim, device=x.device)
        if prev_attn is None:
            prev_attn = torch.zeros(batch_size, self.state_dim, device=x.device)
        if prev_mu is None:
            prev_mu = torch.zeros(batch_size, self.input_dim, device=x.device)
        
        curr_mu = (prev_mu * (self.window_size - 1) + x) / self.window_size
        delta_mu = curr_mu - prev_mu
        
        g = torch.sigmoid(self.W_g(x) + self.U_g(prev_state) + self.b_g)
        Q = self.W_q(x)
        V = self.W_v(x)
        current_info = Q * V
        
        decayed_history = torch.exp(-self.alpha * g.unsqueeze(1)) * prev_attn.unsqueeze(1)
        curr_attn = (current_info.unsqueeze(1) + decayed_history).squeeze(1)
        
        diag_matrix = torch.diag_embed(torch.tanh(self.A))
        batch_diag = diag_matrix.unsqueeze(0).repeat(batch_size, 1, 1)
        state_transfer = batch_diag @ prev_state.unsqueeze(2)
        state_transfer = state_transfer.squeeze(2)
        
        global_input = self.B(x) * (1 - g)
        drift_compensation = self.gamma * delta_mu @ self.B.weight.t()
        
        curr_state = state_transfer + global_input + drift_compensation + curr_attn
        output = torch.sigmoid(self.W_o(curr_state)) + x
        
        return output, curr_state, curr_attn, curr_mu

# -------------------------- 4. RMSU块（添加批归一化） --------------------------
class RMSU_Block(nn.Module):
    def __init__(self, input_dim, state_dim=64, num_layers=2, **kwargs):
        super().__init__()
        # 关键：使用传入的input_dim初始化所有层
        self.layers = nn.ModuleList([RMSU(input_dim, state_dim, **kwargs) for _ in range(num_layers)])
        self.norm_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.BatchNorm1d(input_dim)  # 添加批归一化
            ) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(0.6)
    
    def forward(self, x_seq):
        batch_size, seq_len, input_dim = x_seq.size()
        device = x_seq.device
        
        # 维度校验：确保输入维度与RMSU层预期一致
        if input_dim != self.layers[0].input_dim:
            raise ValueError(
                f"RMSU块输入维度不匹配！预期{self.layers[0].input_dim}, 实际{input_dim}。"
                f"请检查num_channels({input_dim//(self.layers[0].input_dim//input_dim)})、num_bands或wavelet_level参数。"
            )
        
        states = [torch.zeros(batch_size, layer.state_dim, device=device) for layer in self.layers]
        attns = [torch.zeros(batch_size, layer.state_dim, device=device) for layer in self.layers]
        mus = [torch.zeros(batch_size, input_dim, device=device) for layer in self.layers]
        
        output_seq = []
        for t in range(seq_len):
            x = x_seq[:, t, :]
            for i, (layer, norm) in enumerate(zip(self.layers, self.norm_layers)):
                x = norm(x)  # 此时norm维度与x的最后一维严格匹配
                x, states[i], attns[i], mus[i] = layer(x, states[i], attns[i], mus[i])
            x = self.dropout(x)
            output_seq.append(x.unsqueeze(1))
        
        return torch.cat(output_seq, dim=1)

# -------------------------- 5. 时空注意力模块 --------------------------
class SpatialAttention(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(num_channels, num_channels//2),
            nn.ReLU(),
            nn.Linear(num_channels//2, num_channels),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        avg_seq = torch.mean(x, dim=(2, 3))
        attn_weights = self.attn(avg_seq).unsqueeze(2).unsqueeze(3)
        return x * attn_weights

class TemporalAttention(nn.Module):
    def __init__(self, seq_len):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(seq_len, seq_len//2),
            nn.ReLU(),
            nn.Linear(seq_len//2, seq_len),
            nn.Softmax(dim=1)  # 修复：对seq_len维度（dim=1）计算softmax
        )
    
    def forward(self, x):
        avg_ch = torch.mean(x, dim=(1, 3))  # 形状：(batch, seq_len)
        attn_weights = self.attn(avg_ch).unsqueeze(1).unsqueeze(3)
        return x * attn_weights

# -------------------------- 6. 完整EEG MI模型（添加维度校验） --------------------------
class EEGMI_RMSU_Model(nn.Module):
    def __init__(self, num_channels, num_classes, seq_len, num_bands=5, 
                 state_dim=64, num_rmsu_layers=2, wavelet_level=4, **rmsu_kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.seq_len = seq_len
        self.wavelet_level = wavelet_level
        
        # 重新计算RMSU输入维度（考虑卷积层的输出）
        # 卷积层输出通道数为 num_channels * num_bands
        # 小波变换后的特征维度为 wavelet_level
        # 但卷积核大小为(3,2)且padding=(1,1)会改变特征图的宽度
        self.input_dim = num_channels * num_bands * (wavelet_level + 1)  # 修正后的计算
        print(f"📐 模型维度配置：通道数={num_channels}, 频段数={num_bands}, 小波层数={wavelet_level} → 输入维度={self.input_dim}")
        
        self.freq_adjust = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels * num_bands,
            kernel_size=(3, 2),
            padding=(1, 1),
            groups=num_channels
        )
        
        self.spatial_attn = SpatialAttention(num_channels * num_bands)
        self.temporal_attn = TemporalAttention(seq_len)
        
        self.rmsu_block = RMSU_Block(
            input_dim=self.input_dim,  # 传递明确计算的维度
            state_dim=state_dim,
            num_layers=num_rmsu_layers,
            **rmsu_kwargs
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(self.input_dim * seq_len, 128),  # 使用统一维度
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.size(0)
        
        # 时频特征提取
        time_freq_list = []
        for i in range(batch_size):
            eeg_seq = x[i].cpu().numpy()
            tf_feat = eeg_to_time_freq(eeg_seq, level=self.wavelet_level)
            time_freq_list.append(torch.FloatTensor(tf_feat).to(x.device))
        x_tf = torch.stack(time_freq_list, dim=0)  # (batch, channels, seq_len, wavelet_level)
        
        # 频带调整与时空注意力
        x_tf = self.freq_adjust(x_tf)  # (batch, channels×num_bands, seq_len, wavelet_level)
        x_tf = self.spatial_attn(x_tf)
        x_tf = self.temporal_attn(x_tf)
        
        # 转换为RMSU输入格式并校验维度
        x_feat = x_tf.permute(0, 2, 1, 3).contiguous()  # (batch, seq_len, channels×num_bands, wavelet_level)
        x_feat = x_feat.view(batch_size, self.seq_len, -1)  # (batch, seq_len, input_dim)
        
        # 关键校验：确保展平后的维度与预期一致
        if x_feat.size(-1) != self.input_dim:
            raise ValueError(
                f"特征维度不匹配！计算值={self.input_dim}, 实际值={x_feat.size(-1)}。"
                f"请检查wavelet_level（当前{self.wavelet_level}）或num_bands（当前{self.num_bands}）参数。"
            )
        
        # RMSU特征编码与分类
        x_feat = self.rmsu_block(x_feat)
        x_feat = x_feat.view(batch_size, -1)
        return self.classifier(x_feat)

# -------------------------- 7. 带类别针对性增强的数据集 --------------------------
class EnhancedTensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, target_labels=[1,2], prob=0.5):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.target_labels = target_labels
        self.prob = prob
    
    def __getitem__(self, index):
        x = self.data_tensor[index].numpy()
        y = self.target_tensor[index]
        
        if y in self.target_labels:
            x = self._apply_enhanced_aug(x)
        else:
            x = self._apply_basic_aug(x)
        
        return torch.FloatTensor(x.copy()), y  # 修复负步长问题
    
    def __len__(self):
        return self.data_tensor.size(0)
    
    def _apply_basic_aug(self, x):
        if np.random.random() < self.prob:
            noise_std = np.std(x) * 0.02
            x += np.random.normal(0, noise_std, x.shape)
        if np.random.random() < self.prob:
            shift = np.random.randint(-3, 4)
            if shift != 0:
                x = np.roll(x, shift, axis=0)
                x[:shift, :] = x[shift:shift+1, :] if shift>0 else x[shift-1:shift, :]
        if np.random.random() < self.prob:
            x *= np.random.uniform(0.9, 1.1)
        return x
    
    def _apply_enhanced_aug(self, x):
        if np.random.random() < self.prob + 0.2:
            noise_std = np.std(x) * 0.05
            x += np.random.normal(0, noise_std, x.shape)
        if np.random.random() < self.prob + 0.2:
            shift = np.random.randint(-5, 6)
            if shift != 0:
                x = np.roll(x, shift, axis=0)
                x[:shift, :] = x[shift:shift+1, :] if shift>0 else x[shift-1:shift, :]
        if np.random.random() < self.prob + 0.2:
            x *= np.random.uniform(0.8, 1.2)
        if np.random.random() < self.prob:
            from scipy.signal import filtfilt, butter
            b, a = butter(2, [0.1, 0.3], btype='bandpass', fs=250)
            x = filtfilt(b, a, x, axis=0)
        return x

class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]
    def __len__(self):
        return self.data_tensor.size(0)

# -------------------------- 8. BCI数据集类 --------------------------
class BCI_Dataset:
    def __init__(self, data_path, subject, dataset_type='2a', 
                 tmin=0.5, tmax=2.5, sfreq=250):
        self.data_path = data_path
        self.subject = subject
        self.dataset_type = dataset_type
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq
        self.event_config = BCI_EVENT_CONFIG[dataset_type]
        self.target_labels = self.event_config['target_labels']
        
        if dataset_type == '2a':
            self.data_dir = os.path.join(data_path, 'BCICIV_2a_gdf')
            self.train_runs = [4, 8, 12]
            self.test_runs = [6, 10, 14]
        else:
            self.data_dir = os.path.join(data_path, 'BCICIV_2b_gdf')
            self.train_runs = [3, 7, 11]
            self.test_runs = [4, 8, 12]
        
        if not os.path.exists(self.data_dir):
            raise ValueError(f"数据集目录不存在：{self.data_dir}")
        
        self.X_train, self.y_train = self.load_data(runs=self.train_runs, is_train=True)
        self.X_test, self.y_test = self.load_data(runs=self.test_runs, is_train=False)
        self._check_data_validity()
        
        self.X_train, self.y_train = self._balance_target_classes(self.X_train, self.y_train)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, test_size=0.3, random_state=SEED, stratify=self.y_train
        )
        self._normalize_data()
    
    def _check_data_validity(self):
        if len(self.X_train) == 0:
            raise ValueError(f"未加载到训练数据！")
        if len(self.X_test) == 0:
            raise ValueError(f"未加载到测试数据！")
        print(f"初始训练集标签分布：{np.bincount(self.y_train)}")
    
    def _balance_target_classes(self, X, y):
        X_flat = X.reshape(X.shape[0], -1)
        smote = SMOTE(random_state=SEED, sampling_strategy={lab: X[y==lab].shape[0]*2 for lab in self.target_labels})
        X_flat_balanced, y_balanced = smote.fit_resample(X_flat, y)
        X_balanced = X_flat_balanced.reshape(X_flat_balanced.shape[0], X.shape[1], X.shape[2])
        print(f"过采样后训练集标签分布：{np.bincount(y_balanced)}")
        return X_balanced, y_balanced
    
    def _rename_and_filter_channels(self, raw):
        current_channels = raw.ch_names
        new_ch_names = [BCI_CHANNEL_MAPPING.get(ch.strip(), ch.strip()) for ch in current_channels]
        raw.rename_channels(dict(zip(current_channels, new_ch_names)))
        
        eeg_channels = [ch for ch in raw.ch_names if not ch.startswith('EOG-')]
        raw.pick(eeg_channels)
        raw.set_channel_types({ch: 'eeg' for ch in eeg_channels})
        raw.set_montage(make_standard_montage('standard_1005'), on_missing='ignore')
        
        print(f"过滤后EEG通道（{len(eeg_channels)}个）: {eeg_channels[:5]}...")
        return raw
    
    def load_data(self, runs, is_train):
        all_epochs = []
        all_labels = []
        processed_files = set()
        
        for run in runs:
            if self.dataset_type == '2a':
                file_prefix = f"A{self.subject:02d}"
                file_suffix = 'T' if is_train else 'E'
                filename = f"{file_prefix}{file_suffix}.gdf"
            else:
                filename = f"B{self.subject:02d}{run:02d}T.gdf"
            
            file_path = os.path.join(self.data_dir, filename)
            if file_path in processed_files or not os.path.exists(file_path):
                print(f"⚠️  跳过文件：{filename}")
                continue
            processed_files.add(file_path)
            
            try:
                raw = read_raw_gdf(file_path, preload=True, stim_channel='auto')
                raw = self._rename_and_filter_channels(raw)
                raw.filter(8., 30., fir_design='firwin', skip_by_annotation='edge')
                
                events, event_ids = mne.events_from_annotations(raw)
                target_event_desc = self.event_config['train'] if is_train else self.event_config['test']
                target_events = {name: event_ids[desc] for name, desc in target_event_desc.items() if desc in event_ids}
                
                if not target_events:
                    print(f"❌ {filename}无目标事件，跳过")
                    continue
                
                epochs = mne.Epochs(
                    raw, events, target_events, tmin=self.tmin, tmax=self.tmax,
                    proj=True, picks='eeg', baseline=None, preload=True, on_missing='ignore'
                )
                
                epochs_data = epochs.get_data().transpose(0, 2, 1)
                if is_train:
                    label_map = {event_ids[desc]: idx for idx, desc in enumerate(target_event_desc.values()) if desc in event_ids}
                    epoch_labels = [label_map[ev] for ev in epochs.events[:, 2]]
                else:
                    if self.dataset_type == '2a':
                        samples_per_class = len(epochs_data) // self.event_config['num_classes']
                        epoch_labels = np.repeat(range(self.event_config['num_classes']), samples_per_class)[:len(epochs_data)]
                    else:
                        label_map = {event_ids[desc]: idx for idx, desc in enumerate(target_event_desc.values()) if desc in event_ids}
                        epoch_labels = [label_map[ev] for ev in epochs.events[:, 2]]
                
                all_epochs.append(epochs_data)
                all_labels.append(epoch_labels)
                print(f"✅ 处理{filename}：{len(epochs_data)}个Epochs")
            
            except Exception as e:
                print(f"❌ 处理{filename}出错：{str(e)[:100]}...")
                continue
        
        if not all_epochs:
            return np.array([]), np.array([])
        return np.concatenate(all_epochs, axis=0), np.concatenate(all_labels, axis=0)
    
    def _normalize_data(self):
        self.mean = np.mean(self.X_train, axis=(0, 1))
        self.std = np.std(self.X_train, axis=(0, 1)) + 1e-8
        
        self.X_train = (self.X_train - self.mean) / self.std
        self.X_val = (self.X_val - self.mean) / self.std
        self.X_test = (self.X_test - self.mean) / self.std
    
    def get_dataloaders(self, batch_size=32):
        train_ds = EnhancedTensorDataset(
            torch.FloatTensor(self.X_train),
            torch.LongTensor(self.y_train),
            target_labels=self.target_labels
        )
        val_ds = TensorDataset(torch.FloatTensor(self.X_val), torch.LongTensor(self.y_val))
        test_ds = TensorDataset(torch.FloatTensor(self.X_test), torch.LongTensor(self.y_test))
        
        return (
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        )

# -------------------------- 9. 早停机制 --------------------------
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_acc = 0.0
        self.early_stop = False
    
    def __call__(self, current_acc):
        if current_acc > self.best_acc + self.min_delta:
            self.best_acc = current_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# -------------------------- 10. 训练/评估/可视化函数 --------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, 
                device, num_epochs=50, model_save_path='best_model.pth', early_stop_patience=5):
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    early_stopping = EarlyStopping(patience=early_stop_patience)
    
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
        
        epoch_train_loss = train_loss / len(train_loader.dataset)
        epoch_train_acc = accuracy_score(train_labels, train_preds)
        epoch_val_loss = val_loss / len(val_loader.dataset)
        epoch_val_acc = accuracy_score(val_labels, val_preds)
        
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        
        print(f"Epoch {epoch+1:2d}/{num_epochs}")
        print(f"Train | Loss: {epoch_train_loss:.4f} | Acc: {epoch_train_acc:.4f}")
        print(f"Val   | Loss: {epoch_val_loss:.4f} | Acc: {epoch_val_acc:.4f}")
        print(f"早停计数器：{early_stopping.counter}/{early_stopping.patience}")
        print('-' * 50)
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"💾 保存最佳模型（Val Acc: {best_val_acc:.4f}）")
        
        early_stopping(epoch_val_acc)
        if early_stopping.early_stop:
            print(f"⏹️  早停触发")
            break
        
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(epoch_val_loss)
        else:
            scheduler.step()
    
    return history

def evaluate_model(model, test_loader, device, class_names, target_labels=[1,2]):
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    cm = confusion_matrix(test_labels, test_preds)
    
    target_mask = np.isin(test_labels, target_labels)
    target_test_labels = np.array(test_labels)[target_mask]
    target_test_preds = np.array(test_preds)[target_mask]
    target_acc = accuracy_score(target_test_labels, target_test_preds) if len(target_test_labels) > 0 else 0.0
    
    print(f"\n📊 整体测试准确率: {test_acc:.4f}")
    print(f"🎯 目标类别准确率: {target_acc:.4f}")
    print("\n分类报告：")
    print(classification_report(
        test_labels, test_preds, target_names=class_names, digits=4
    ))
    
    return test_acc, target_acc, cm

def plot_history(history, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📈 训练历史图保存至：{save_path}")

def plot_confusion_matrix(cm, class_names, save_path, target_labels=[1,2]):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, label='Number of Samples')
    
    tick_marks = np.arange(len(class_names))
    tick_labels = [f"{name} {'(目标)' if i in target_labels else ''}" for i, name in enumerate(class_names)]
    plt.xticks(tick_marks, tick_labels, rotation=45, ha='right')
    plt.yticks(tick_marks, tick_labels)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 混淆矩阵图保存至：{save_path}")

# -------------------------- 11. 主函数 --------------------------
def main(args):
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"📁 结果保存至：{args.save_dir}")
    
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_cuda else 'cpu')
    print(f"\n🚀 使用设备：{device}")
    
    print(f"\n📥 加载BCI {args.dataset_type}数据集（被试{args.subject}）")
    dataset = BCI_Dataset(
        data_path=args.data_path,
        subject=args.subject,
        dataset_type=args.dataset_type,
        tmin=args.tmin,
        tmax=args.tmax
    )
    train_loader, val_loader, test_loader = dataset.get_dataloaders(batch_size=args.batch_size)
    
    num_channels = dataset.X_train.shape[2]
    seq_len = dataset.X_train.shape[1]
    class_names = list(dataset.event_config['train'].keys())
    target_labels = dataset.event_config['target_labels']
    print(f"\n📋 数据集信息：")
    print(f"  - 类别：{dataset.event_config['num_classes']}类 → {class_names}")
    print(f"  - 目标类别：{[class_names[lab] for lab in target_labels]}")
    print(f"  - 维度：{num_channels}通道 × {seq_len}时间步")
    
    print(f"\n🔧 初始化RMSU模型")
    model = EEGMI_RMSU_Model(
        num_channels=num_channels,
        num_classes=dataset.event_config['num_classes'],
        seq_len=seq_len,
        num_bands=args.num_bands,
        state_dim=args.state_dim,
        num_rmsu_layers=args.num_rmsu_layers,
        wavelet_level=args.wavelet_level,
        alpha=args.alpha,
        gamma=args.gamma,
        window_size=args.window_size
    ).to(device)
    
    # 添加模型参数打印
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数: {total_params:,}")
    
    # 使用带权重调整的损失函数，处理类别不平衡
    class_counts = np.bincount(dataset.y_train)
    class_weights = 1. / class_counts
    class_weights = torch.FloatTensor(class_weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 使用AdamW优化器，通常比Adam更稳定
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 使用余弦退火学习率调度器
    scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    print(f"\n🔥 开始训练（共{args.epochs}轮）")
    model_save_path = os.path.join(args.save_dir, f'best_model_subj{args.subject}.pth')
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        model_save_path=model_save_path,
        early_stop_patience=args.early_stop_patience
    )
    
    history_save_path = os.path.join(args.save_dir, f'train_history_subj{args.subject}.png')
    plot_history(history, history_save_path)
    
    print(f"\n📊 评估最佳模型")
    model.load_state_dict(torch.load(model_save_path))
    test_acc, target_acc, cm = evaluate_model(model, test_loader, device, class_names, target_labels)
    
    cm_save_path = os.path.join(args.save_dir, f'confusion_matrix_subj{args.subject}.png')
    plot_confusion_matrix(cm, class_names, cm_save_path, target_labels)
    
    print(f"\n🎉 实验完成！目标类别准确率：{target_acc:.4f}")

# -------------------------- 12. 命令行参数解析 --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='EEG MI Recognition with RMSU Model（优化feet/right_hand分类）'
    )
    
    # 数据相关参数
    parser.add_argument('--data_path', type=str, default='../data', help='数据根路径')
    parser.add_argument('--dataset_type', type=str, default='2a', choices=['2a', '2b'], help='数据集类型')
    parser.add_argument('--subject', type=int, default=1, help='被试编号（1-9）')
    parser.add_argument('--tmin', type=float, default=0.5, help='事件后起始时间（秒）')
    parser.add_argument('--tmax', type=float, default=2.5, help='事件后结束时间（秒）')
    
    # 模型相关参数（需确保 num_channels * num_bands * wavelet_level 维度一致）
    parser.add_argument('--num_bands', type=int, default=5, help='频段提取数量')
    parser.add_argument('--state_dim', type=int, default=64, help='RMSU状态维度')
    parser.add_argument('--num_rmsu_layers', type=int, default=2, help='RMSU层数')
    parser.add_argument('--wavelet_level', type=int, default=4, help='小波分解层数（影响输入维度）')
    parser.add_argument('--alpha', type=float, default=0.1, help='RMSU时间衰减系数')
    parser.add_argument('--gamma', type=float, default=0.05, help='漂移补偿系数')
    parser.add_argument('--window_size', type=int, default=100, help='基线窗口大小')
    
    # 训练相关参数
    parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
    parser.add_argument('--epochs', type=int, default=500, help='训练轮数')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减')
    parser.add_argument('--early_stop_patience', type=int, default=50, help='早停耐心轮数')
    parser.add_argument('--use_cuda', action='store_true', default=True, help='使用CUDA')
    
    # 结果保存
    parser.add_argument('--save_dir', type=str, default='./results_final', help='结果保存目录')
    
    args = parser.parse_args()
    main(args)
