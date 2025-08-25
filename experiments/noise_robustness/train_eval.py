import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

def train(model, dataloader, criterion, optimizer, device, epoch, logdir, scheduler=None):
    model.train()
    total_loss = 0
    y_true, y_pred = [], []
    last_attn, last_graph = None, None
    last_edge_attn = None
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out, attn_weights, graph_feat, edge_attn = model(x)
        loss = criterion(out, y)
        loss = loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        y_true.extend(y.cpu().numpy())
        y_pred.extend(out.argmax(dim=1).cpu().numpy())
        last_attn = attn_weights.detach().cpu()
        last_graph = graph_feat.detach().cpu()
        last_edge_attn = edge_attn.detach().cpu()

    acc = accuracy_score(y_true, y_pred)
    return total_loss / len(dataloader), acc, y_true, y_pred, last_attn, last_graph, last_edge_attn

def evaluate(model, dataloader, device):
    model.eval()
    y_true, y_pred = [], []
    all_attn = []
    all_graph = []
    all_edge_attn = []
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            out, attn_weights, graph_feat, edge_attn = model(x)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(out.argmax(dim=1).cpu().numpy())
            all_attn.append(attn_weights.cpu())
            all_graph.append(graph_feat.cpu())
            all_edge_attn.append(edge_attn.cpu())
    
    acc = accuracy_score(y_true, y_pred)
    # Use last batch for visualization
    last_attn = all_attn[-1] if all_attn else None
    last_graph = all_graph[-1] if all_graph else None
    last_edge_attn = all_edge_attn[-1] if all_edge_attn else None
    
    return acc, y_true, y_pred, last_attn, last_graph, last_edge_attn

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_train_curve(train_loss_list, val_acc_list, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(train_loss_list)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax2.plot(val_acc_list)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_attention_weights(attn_weights, save_path):
    if attn_weights is None:
        return
    plt.figure(figsize=(10, 6))
    attn_np = attn_weights.numpy()
    plt.imshow(attn_np, cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title('Attention Weights')
    plt.xlabel('Modal')
    plt.ylabel('Sample')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_gnn_coupling(edge_attn_tensor, save_path):
    if edge_attn_tensor is None:
        return
    plt.figure(figsize=(8, 6))
    edge_attn_np = edge_attn_tensor.numpy()
    plt.imshow(edge_attn_np, cmap='plasma', aspect='auto')
    plt.colorbar()
    plt.title('GNN Edge Attention')
    plt.xlabel('Edge')
    plt.ylabel('Sample')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_classification_report(y_true, y_pred, save_path):
    report = classification_report(y_true, y_pred)
    with open(save_path, 'w') as f:
        f.write(report)