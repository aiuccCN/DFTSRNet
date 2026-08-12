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
    """Graph reasoning over the AFBS sub-bands.

    node_feat selects the per-sub-band node descriptor:
      'mean'        - temporal mean, i.e. ONE scalar per sub-band. Because the
                      six scalars are nearly identical, the attention logits are
                      nearly identical too and the coupling matrix collapses to
                      the uniform matrix 1/M (kept for the ablation in the paper).
      'band_energy' - log band-energy profile of the sub-band spectrum, which
                      keeps sub-bands with different passbands distinguishable
                      and lets the attention learn a genuinely non-uniform
                      inter-band coupling matrix.
    """
    def __init__(self, in_dim=None, hidden=32, out_dim=32, heads=4,
                 node_feat='band_energy', num_bins=8,
                 couple_from='layer1', residual=True):
        super().__init__()
        self.node_feat = node_feat
        self.num_bins = num_bins
        # couple_from: which attention layer is reported as the coupling matrix C.
        #   'layer1' reflects relations between the original sub-band descriptors.
        #   'layer2' is computed on already-aggregated features, which on a complete
        #            graph are strongly smoothed (over-smoothing) and nearly identical.
        self.couple_from = couple_from
        # residual keeps the nodes distinguishable across layers (anti over-smoothing)
        self.residual = residual
        if in_dim is None:
            in_dim = 1 if node_feat == 'mean' else num_bins
        self.node_proj = nn.Linear(in_dim, hidden)
        # GATConv: returns attention if requested
        self.gat1 = GATConv(hidden, out_dim // heads, heads=heads, concat=True, dropout=0.2)
        self.gat2 = GATConv(out_dim, out_dim // heads, heads=heads, concat=True, dropout=0.2)
        self.dropout = nn.Dropout(0.2)
        self.relu = nn.ReLU()

    def _node_descriptor(self, modals):
        """modals: [B, M, L] -> node features [B, M, in_dim]"""
        if self.node_feat == 'mean':
            return modals.mean(dim=-1, keepdim=True)              # [B, M, 1]
        # log band-energy profile over the one-sided spectrum
        S = torch.fft.rfft(modals, dim=-1).abs()                  # [B, M, F]
        F_bins = S.shape[-1]
        edges = torch.linspace(0, F_bins, self.num_bins + 1).long().tolist()
        bands = torch.stack([S[..., edges[i]:edges[i + 1]].mean(dim=-1)
                             for i in range(self.num_bins)], dim=-1)
        # Vibration amplitudes are O(1e-4), so log1p(x) ~= x would leave the
        # descriptor vanishingly small and the attention logits indistinguishable.
        # Use a log scale with an explicit floor, then standardise per sample so
        # the descriptor has O(1) dynamic range and the sub-bands stay comparable.
        d = torch.log(bands + 1e-8)                               # [B, M, num_bins]
        mu = d.mean(dim=(1, 2), keepdim=True)
        sd = d.std(dim=(1, 2), keepdim=True)
        return (d - mu) / (sd + 1e-6)

    def forward(self, modals):
        # modals: [B, M, L]
        B, M, L = modals.shape
        x = self._node_descriptor(modals)  # [B, M, in_dim]
        x = self.node_proj(x)  # [B, M, hidden]
        x = self.relu(x)

        device = modals.device
        idx = torch.arange(M, device=device)
        row = idx.repeat_interleave(M)
        col = idx.repeat(M)
        edge_index = torch.stack([row, col], dim=0)  # complete graph with self loops

        all_node_embs = []
        all_attn_weights = []  # to collect attention per sample

        for i in range(B):
            xi = x[i]  # [M, hidden]
            # GATConv expects [num_nodes, feat]
            h1, (edge_idx1, attn1) = self.gat1(xi, edge_index, return_attention_weights=True)
            h1 = self.relu(h1)
            if self.residual:
                h1 = h1 + xi  # keep nodes distinguishable (anti over-smoothing)
            h1 = self.dropout(h1)
            h2, (edge_idx2, attn2) = self.gat2(h1, edge_index, return_attention_weights=True)
            h2 = self.relu(h2)  # [M, out_dim]
            if self.residual:
                h2 = h2 + h1
            all_node_embs.append(h2)  # per-sample
            # Reported coupling matrix: attention of the selected layer,
            # averaged over heads. attn* is [E, heads]; edge_idx* is [2, E].
            attn_matrix = torch.zeros((M, M), device=device)
            if self.couple_from == 'layer1':
                edge_idx2, attn2 = edge_idx1, attn1
            elif self.couple_from == 'mean':
                attn2 = (attn1 + attn2) / 2.0
            attn_vals = attn2.mean(dim=1)  # [E]
            for (src, dst), attn_val in zip(edge_idx2.t(), attn_vals):
                attn_matrix[src, dst] = attn_val
            all_attn_weights.append(attn_matrix)  # [M, M]

        graph_features = torch.stack(all_node_embs, dim=0)  # [B, M, out_dim]
        attn_tensor = torch.stack(all_attn_weights, dim=0)  # [B, M, M]
        return graph_features, attn_tensor  # return both embeddings and learned coupling weights

# ---- FiLM-style Fusion + Structural CNN ----
class ModalStructuralFusionCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.film_gamma = nn.Linear(Config.GNN_HIDDEN, in_channels)
        self.film_beta = nn.Linear(Config.GNN_HIDDEN, in_channels)

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

        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        nn.init.xavier_uniform_(self.film_gamma.weight)
        nn.init.xavier_uniform_(self.film_beta.weight)

    def forward(self, fused_signal, graph_summary):
        gamma = self.film_gamma(graph_summary).unsqueeze(-1)  # [B, C, 1]
        beta = self.film_beta(graph_summary).unsqueeze(-1)
        modulated = fused_signal * (1 + gamma) + beta

        x = F.relu(self.bn1(self.conv1(modulated)))
        res = self.resblock(x)
        x = F.relu(x + res)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).squeeze(-1)
        out = self.classifier(x)
        return out

# ---- Overall DFTSRNet ----
class DFTSRNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, graph_node_feat='band_energy'):
        super().__init__()
        self.afbs = AdaptiveFreqBandSelect(in_channels, num_modal=Config.NUM_MODALS, seq_len=Config.SEQ_LEN)
        self.graph_reason = ModalGraphReasoning(hidden=Config.GNN_HIDDEN, out_dim=Config.GNN_HIDDEN,
                                               heads=4, node_feat=graph_node_feat)
        self.fusion = ModalStructuralFusionCNN(in_channels=1, num_classes=num_classes)

    def forward(self, x):
        fused, modals, attn_weights = self.afbs(x)  # [B,1,L], [B,M,L], [B,M]
        graph_feat, attn_matrix = self.graph_reason(modals)  # [B,M,hidden], [B,M,M]
        graph_summary = graph_feat.mean(dim=1)  # [B, hidden]
        out = self.fusion(fused, graph_summary)  # [B, num_classes]
        return out, attn_weights, graph_feat, attn_matrix  # include attention matrix for coupling viz
