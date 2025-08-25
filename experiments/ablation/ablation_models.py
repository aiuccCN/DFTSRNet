import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from config import Config

# ---- 基础CNN模块 ----
class BaseCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.resblock = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2)
        )
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        res = self.resblock(x)
        x = F.relu(x + res)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        out = self.classifier(x)
        return out

# ---- 自适应频带选择模块 ----
class AdaptiveFreqBandSelect(nn.Module):
    def __init__(self, in_channels=1, num_modal=6, seq_len=1024):
        super().__init__()
        self.num_modal = num_modal
        kernel_sizes = [3,5,7,9,11,13][:num_modal]
        self.filters = nn.ModuleList([
            nn.Conv1d(in_channels, 1, kernel_size=ks, padding=ks // 2, bias=False)
            for ks in kernel_sizes
        ])
        self.attn_fc = nn.Sequential(
            nn.Linear(num_modal, 64),
            nn.ReLU(),
            nn.Linear(64, num_modal)
        )
        self.temperature = 1.0
    
    def forward(self, x):
        modals = torch.stack([filt(x) for filt in self.filters], dim=1)  # [B, M, 1, L]
        modals = modals.squeeze(2)  # [B, M, L]
        fft_modals = torch.fft.rfft(modals, dim=-1).abs()  # [B, M, F]
        modal_feat = fft_modals.mean(dim=2)  # [B, M]
        attn_logits = self.attn_fc(modal_feat)  # [B, M]
        attn_weights = F.softmax(attn_logits / self.temperature, dim=-1)  # [B, M]
        weights = attn_weights.unsqueeze(-1)  # [B, M, 1]
        fused = (modals * weights).sum(dim=1, keepdim=True)  # [B,1,L]
        return fused, modals, attn_weights

# ---- 模态图推理模块 ----
class ModalGraphReasoning(nn.Module):
    def __init__(self, in_dim=1, hidden=32, out_dim=32, heads=4):
        super().__init__()
        self.node_proj = nn.Linear(in_dim, hidden)
        self.gat1 = GATConv(hidden, out_dim // heads, heads=heads, concat=True, dropout=0.2)
        self.gat2 = GATConv(out_dim, out_dim // heads, heads=heads, concat=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, modals):
        # modals: [B, M, L]
        B, M, L = modals.shape
        modals_agg = modals.mean(dim=-1)  # [B, M]
        x = modals_agg.unsqueeze(-1)  # [B, M, 1]
        x = self.node_proj(x)  # [B, M, hidden]
        x = self.relu(x)
        
        device = modals.device
        idx = torch.arange(M, device=device)
        row = idx.repeat_interleave(M)
        col = idx.repeat(M)
        edge_index = torch.stack([row, col], dim=0)  # complete graph
        
        all_node_embs = []
        all_attn_weights = []
        
        for i in range(B):
            xi = x[i]  # [M, hidden]
            h1, (edge_idx1, attn1) = self.gat1(xi, edge_index, return_attention_weights=True)
            h1 = self.relu(h1)
            h1 = self.dropout(h1)
            h2, (edge_idx2, attn2) = self.gat2(h1, edge_index, return_attention_weights=True)
            h2 = self.relu(h2)  # [M, out_dim]
            all_node_embs.append(h2)
            
            # 构建注意力矩阵
            attn_matrix = torch.zeros((M, M), device=device)
            attn_vals = attn2.mean(dim=1)  # [E]
            for (src, dst), attn_val in zip(edge_idx2.t(), attn_vals):
                attn_matrix[src, dst] = attn_val
            all_attn_weights.append(attn_matrix)
        
        graph_features = torch.stack(all_node_embs, dim=0)  # [B, M, out_dim]
        attn_tensor = torch.stack(all_attn_weights, dim=0)  # [B, M, M]
        return graph_features, attn_tensor

# ---- FiLM融合模块 ----
class FiLMFusion(nn.Module):
    def __init__(self, in_channels, graph_dim):
        super().__init__()
        self.film_gamma = nn.Linear(graph_dim, in_channels)
        self.film_beta = nn.Linear(graph_dim, in_channels)
    
    def forward(self, signal, graph_summary):
        gamma = self.film_gamma(graph_summary).unsqueeze(-1)  # [B, C, 1]
        beta = self.film_beta(graph_summary).unsqueeze(-1)
        modulated = signal * (1 + gamma) + beta
        return modulated

# ---- 消融模型定义 ----
class AblationModel(nn.Module):
    def __init__(self, model_type, in_channels=1, num_classes=10):
        super().__init__()
        self.model_type = model_type
        
        # 基础CNN
        self.base_cnn = BaseCNN(in_channels, num_classes)
        
        # 各个模块
        if 'AFBS' in model_type or model_type == 'Full':
            self.afbs = AdaptiveFreqBandSelect(in_channels, Config.NUM_MODALS, Config.SEQ_LEN)
        
        if 'MGR' in model_type or model_type == 'Full':
            self.mgr = ModalGraphReasoning(1, Config.GNN_HIDDEN, Config.GNN_HIDDEN, 4)
        
        if 'FiLM' in model_type or model_type == 'Full':
            self.film = FiLMFusion(in_channels, Config.GNN_HIDDEN)
        
        # 根据模型类型调整最终分类器
        if model_type != 'Base_CNN':
            self.final_conv = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
            self.final_bn = nn.BatchNorm1d(64)
            self.final_resblock = nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=5, padding=2)
            )
            self.final_conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
            self.final_bn2 = nn.BatchNorm1d(128)
            self.final_pool = nn.AdaptiveAvgPool1d(1)
            self.final_classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(128, num_classes)
            )
    
    def forward(self, x):
        if self.model_type == 'Base_CNN':
            return self.base_cnn(x)
        
        # 处理各种消融变体
        processed_signal = x
        modals = None
        graph_summary = None
        
        # AFBS处理
        if hasattr(self, 'afbs') and 'w/o_AFBS' not in self.model_type:
            processed_signal, modals, _ = self.afbs(x)
        else:
            # 没有AFBS时，创建虚拟模态
            modals = x.repeat(1, Config.NUM_MODALS, 1)  # [B, M, L]
        
        # MGR处理
        if hasattr(self, 'mgr') and 'w/o_MGR' not in self.model_type:
            if '+MGR' == self.model_type:
                # 静态FiLM版本
                graph_feat, _ = self.mgr(modals)
                graph_summary = graph_feat.mean(dim=1)  # [B, hidden]
            else:
                graph_feat, _ = self.mgr(modals)
                graph_summary = graph_feat.mean(dim=1)
        else:
            # 没有MGR时，用均值池化
            graph_summary = modals.mean(dim=(1,2)).unsqueeze(-1).repeat(1, Config.GNN_HIDDEN)
        
        # FiLM处理
        if hasattr(self, 'film') and 'w/o_FiLM' not in self.model_type:
            if '+FiLM' == self.model_type:
                # MGR用均值池化替代
                graph_summary = modals.mean(dim=(1,2)).unsqueeze(-1).repeat(1, Config.GNN_HIDDEN)
            processed_signal = self.film(processed_signal, graph_summary)
        
        # 最终分类
        x = F.relu(self.final_bn(self.final_conv(processed_signal)))
        res = self.final_resblock(x)
        x = F.relu(x + res)
        x = F.relu(self.final_bn2(self.final_conv2(x)))
        x = self.final_pool(x).squeeze(-1)
        out = self.final_classifier(x)
        
        return out

def create_ablation_model(model_name, num_classes=10):
    """根据模型名称创建对应的消融模型"""
    return AblationModel(model_name, in_channels=1, num_classes=num_classes)