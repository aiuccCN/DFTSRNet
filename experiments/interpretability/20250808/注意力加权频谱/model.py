import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from config import Config

# ---- Adaptive Frequency Band Selection ----
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
        for conv in self.filters:
            nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
        for m in self.attn_fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

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

# ---- Modal Graph Reasoning with GAT (learnable edge attention) ----
class ModalGraphReasoning(nn.Module):
    def __init__(self, in_dim=1, hidden=32, out_dim=32, heads=4):
        super().__init__()
        self.node_proj = nn.Linear(in_dim, hidden)
        # GATConv: returns attention if requested
        self.gat1 = GATConv(hidden, out_dim // heads, heads=heads, concat=True, dropout=0.2)
        self.gat2 = GATConv(out_dim, out_dim // heads, heads=heads, concat=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.node_proj.weight)
        if self.node_proj.bias is not None:
            nn.init.constant_(self.node_proj.bias, 0.0)

    def forward(self, modals):
        # modals: [B, M, L] -> treat each modal as a node
        B, M, L = modals.shape
        
        # Create node features: aggregate each modal to a single feature
        node_feat = modals.mean(dim=2)  # [B, M] - average over sequence length
        node_feat = self.node_proj(node_feat.unsqueeze(-1)).squeeze(-1)  # [B, M, hidden]
        
        # Create fully connected graph edges for each batch
        edge_index = torch.combinations(torch.arange(M), r=2, with_replacement=False).t()
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)  # bidirectional
        edge_index = edge_index.to(modals.device)
        
        # Process each sample in batch
        batch_outputs = []
        batch_edge_attns = []
        
        for b in range(B):
            x = node_feat[b]  # [M, hidden]
            
            # First GAT layer
            x1, edge_attn1 = self.gat1(x, edge_index, return_attention_weights=True)
            x1 = self.dropout(F.relu(x1))
            
            # Second GAT layer  
            x2, edge_attn2 = self.gat2(x1, edge_index, return_attention_weights=True)
            
            # Global pooling
            graph_summary = torch.mean(x2, dim=0)  # [out_dim]
            
            batch_outputs.append(graph_summary)
            batch_edge_attns.append(edge_attn2[1])  # attention weights
        
        graph_feat = torch.stack(batch_outputs)  # [B, out_dim]
        edge_attn = torch.stack(batch_edge_attns)  # [B, num_edges]
        
        return graph_feat, edge_attn

# ---- Modal Structural Fusion CNN ----
class ModalStructuralFusionCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        
        # Graph feature integration
        self.graph_proj = nn.Linear(32, 256)  # Project graph features
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(256 + 256, 512),  # CNN + Graph features
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, fused_signal, graph_summary):
        # CNN pathway
        x = F.relu(self.conv1(fused_signal))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)  # [B, 256]
        x = self.dropout(x)
        
        # Graph pathway
        graph_feat = F.relu(self.graph_proj(graph_summary))  # [B, 256]
        
        # Fusion
        combined = torch.cat([x, graph_feat], dim=1)  # [B, 512]
        output = self.classifier(combined)
        
        return output

# ---- Main DFTSRNet Model ----
class DFTSRNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super().__init__()
        self.afbs = AdaptiveFreqBandSelect(in_channels, Config.NUM_MODALS, Config.SEQ_LEN)
        self.mgr = ModalGraphReasoning(in_dim=1, hidden=Config.GNN_HIDDEN, out_dim=Config.GNN_HIDDEN)
        self.msfcnn = ModalStructuralFusionCNN(in_channels, num_classes)

    def forward(self, x, return_attention=False):
        # AFBS: Adaptive frequency band selection
        fused_signal, modals, attn_weights = self.afbs(x)
        
        # MGR: Modal graph reasoning
        graph_feat, edge_attn = self.mgr(modals)
        
        # MSFCNN: Final classification
        output = self.msfcnn(fused_signal, graph_feat)
        
        return output, attn_weights, graph_feat, edge_attn