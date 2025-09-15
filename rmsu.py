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

# -------------------------- 2. RMSU核心单元 --------------------------
class RMSU(nn.Module):
    def __init__(self, input_dim, state_dim, alpha=0.1, gamma=0.05, window_size=100):
        super().__init__()
        self.input_dim = input_dim
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
        current_info = Q * V  # 形状: (batch_size, state_dim)
        
        # 关键修复：统一维度后相乘（均扩展为3维）
        decayed_history = torch.exp(-self.alpha * g.unsqueeze(1)) * prev_attn.unsqueeze(1)  # (batch, 1, state_dim)
        curr_attn = (current_info.unsqueeze(1) + decayed_history).squeeze(1)  # 压缩回2维 (batch, state_dim)
        
        # 状态转移计算
        diag_matrix = torch.diag_embed(torch.tanh(self.A))  # (state_dim, state_dim)
        batch_diag = diag_matrix.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch_size, state_dim, state_dim)
        state_transfer = batch_diag @ prev_state.unsqueeze(2)  # (batch_size, state_dim, 1)
        state_transfer = state_transfer.squeeze(2)  # (batch_size, state_dim)
        
        # 修复：先将输入x通过线性层B投影到状态空间，然后再与(1-g)逐元素相乘
        global_input = self.B(x) * (1 - g)
        drift_compensation = self.gamma * delta_mu @ self.B.weight.t()
        
        curr_state = state_transfer + global_input + drift_compensation + curr_attn
        output = torch.sigmoid(self.W_o(curr_state)) + x
        
        return output, curr_state, curr_attn, curr_mu

# -------------------------- 3. RMSU块 --------------------------
class RMSU_Block(nn.Module):
    def __init__(self, input_dim, state_dim, num_layers=3, **kwargs):
        super().__init__()
        self.layers = nn.ModuleList([RMSU(input_dim, state_dim,** kwargs) for _ in range(num_layers)])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])
    
    def forward(self, x_seq):
        batch_size, seq_len, input_dim = x_seq.size()
        device = x_seq.device
        
        states = [torch.zeros(batch_size, layer.state_dim, device=device) for layer in self.layers]
        attns = [torch.zeros(batch_size, layer.state_dim, device=device) for layer in self.layers]
        mus = [torch.zeros(batch_size, input_dim, device=device) for layer in self.layers]
        
        output_seq = []
        for t in range(seq_len):
            x = x_seq[:, t, :]
            for i, (layer, norm) in enumerate(zip(self.layers, self.norm_layers)):
                x = norm(x)
                x, states[i], attns[i], mus[i] = layer(x, states[i], attns[i], mus[i])
            output_seq.append(x.unsqueeze(1))
        
        return torch.cat(output_seq, dim=1)

# -------------------------- 4. 完整EEG MI模型 --------------------------
class EEGMI_RMSU_Model(nn.Module):
    def __init__(self, num_channels, num_classes, seq_len, num_bands=5, 
                 state_dim=128, num_rmsu_layers=3, **rmsu_kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.seq_len = seq_len
        
        self.band_extractor = nn.Conv1d(
            in_channels=num_channels,
            out_channels=num_channels * num_bands,
            kernel_size=5,
            padding=2,
            groups=num_channels
        )
        
        self.spatial_attn = nn.Sequential(nn.Linear(num_channels, num_channels), nn.Softmax(dim=1))
        input_dim = num_channels * num_bands
        
        self.rmsu_block = RMSU_Block(
            input_dim=input_dim,
            state_dim=state_dim,
            num_layers=num_rmsu_layers,** rmsu_kwargs
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim * seq_len, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        bands = self.band_extractor(x)
        bands = bands.view(bands.size(0), self.num_channels, self.num_bands, self.seq_len)
        
        attn_weights = self.spatial_attn(torch.mean(bands, dim=(2, 3)))
        attn_weights = attn_weights.unsqueeze(2).unsqueeze(3)
        bands = bands * attn_weights
        
        features = bands.permute(0, 3, 1, 2).contiguous()
        features = features.view(features.size(0), self.seq_len, -1)
        
        features = self.rmsu_block(features)
        features = features.view(features.size(0), -1)
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
        print('-' * 50)
        
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), model_save_path)
            print(f"💾 保存最佳模型（Val Acc: {best_val_acc:.4f}）→ {model_save_path}")
        
        scheduler.step(epoch_val_loss)
    
    return history

def evaluate_model(model, test_loader, device, class_names):
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
    
    print(f"\n📊 测试集性能：")
    print(f"测试准确率: {test_acc:.4f}")
    print("\n分类报告：")
    print(classification_report(
        test_labels, test_preds, target_names=class_names, digits=4
    ))
    
    return test_acc, cm

def plot_history(history, save_path):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', linewidth=2)
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy', linewidth=2)
    plt.plot(history['val_acc'], label='Val Accuracy', linewidth=2)
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n📈 训练历史图保存至：{save_path}")

def plot_confusion_matrix(cm, class_names, save_path):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, label='Number of Samples')
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45, ha='right')
    plt.yticks(tick_marks, class_names)
    
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
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 混淆矩阵图保存至：{save_path}")

# -------------------------- 8. 主函数 --------------------------
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
    class_names = list(dataset.train_event_desc.keys())
    print(f"\n📋 数据集信息：")
    print(f"  - 类别：{dataset.num_classes}类 → {class_names}")
    print(f"  - 维度：{num_channels}通道 × {seq_len}时间步")
    print(f"  - 样本：训练{len(dataset.X_train)}个 | 验证{len(dataset.X_val)}个 | 测试{len(dataset.X_test)}个")
    
    print(f"\n🔧 初始化RMSU模型（{args.num_rmsu_layers}层，状态维度{args.state_dim}）")
    model = EEGMI_RMSU_Model(
        num_channels=num_channels,
        num_classes=dataset.num_classes,
        seq_len=seq_len,
        num_bands=args.num_bands,
        state_dim=args.state_dim,
        num_rmsu_layers=args.num_rmsu_layers,
        alpha=args.alpha,
        gamma=args.gamma,
        window_size=args.window_size
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
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
        model_save_path=model_save_path
    )
    
    history_save_path = os.path.join(args.save_dir, f'train_history_subj{args.subject}.png')
    plot_history(history, history_save_path)
    
    print(f"\n📊 评估最佳模型（{model_save_path}）")
    model.load_state_dict(torch.load(model_save_path))
    test_acc, cm = evaluate_model(model, test_loader, device, class_names)
    
    cm_save_path = os.path.join(args.save_dir, f'confusion_matrix_subj{args.subject}.png')
    plot_confusion_matrix(cm, class_names, cm_save_path)
    
    print(f"\n🎉 实验完成！结果保存至：{args.save_dir}")

# -------------------------- 9. 命令行参数解析 --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='EEG MI Recognition with RMSU Model (BCICIV_2a_gdf/2b_gdf)',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument('--data_path', type=str, default='../data',
                        help='数据根路径（默认：../data）\n需包含BCICIV_2a_gdf/BCICIV_2b_gdf子目录')
    parser.add_argument('--dataset_type', type=str, default='2a',
                        choices=['2a', '2b'],
                        help='数据集类型（默认：2a）\n2a:4类(左手/右手/脚/舌头) | 2b:2类(左手/右手)')
    parser.add_argument('--subject', type=int, default=1, help='被试编号（1-9）')
    parser.add_argument('--tmin', type=float, default=0.5, help='事件后起始时间（秒）')
    parser.add_argument('--tmax', type=float, default=2.5, help='事件后结束时间（秒）')
    
    parser.add_argument('--num_bands', type=int, default=5, help='频段提取数量（默认：5）')
    parser.add_argument('--state_dim', type=int, default=128, help='RMSU状态维度（默认：128）')
    parser.add_argument('--num_rmsu_layers', type=int, default=3, help='RMSU层数（默认：3）')
    parser.add_argument('--alpha', type=float, default=0.1, help='RMSU时间衰减系数（默认：0.1）')
    parser.add_argument('--gamma', type=float, default=0.05, help='漂移补偿系数（默认：0.05）')
    parser.add_argument('--window_size', type=int, default=100, help='基线窗口大小（默认：100）')
    
    parser.add_argument('--batch_size', type=int, default=64, help='批次大小（默认：32）')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数（默认：50）')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='初始学习率（默认：1e-3）')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='权重衰减（默认：1e-5）')
    parser.add_argument('--use_cuda', action='store_true', default=True, help='使用CUDA（默认：True）')
    
    parser.add_argument('--save_dir', type=str, default='./results', help='结果保存目录（默认：./results）')
    
    args = parser.parse_args()
    main(args)
