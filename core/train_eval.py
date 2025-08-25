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
    attn_concat = torch.cat(all_attn, dim=0)
    graph_concat = torch.cat(all_graph, dim=0)
    edge_attn_concat = torch.cat(all_edge_attn, dim=0)  # [N, M, M]
    return acc, y_true, y_pred, attn_concat, graph_concat, edge_attn_concat

def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted'); plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def plot_train_curve(train_loss_list, val_acc_list, save_path):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(train_loss_list, label='Train Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Train Loss'); plt.grid(True)
    plt.subplot(1,2,2)
    plt.plot(val_acc_list, label='Val Acc')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Validation Accuracy'); plt.grid(True)
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def plot_attention_weights(attn_weights, save_path):
    avg = attn_weights.mean(dim=0).numpy()
    plt.figure(figsize=(8,4))
    sns.barplot(x=np.arange(len(avg)), y=avg, hue=np.arange(len(avg)), dodge=False, legend=False)
    plt.xlabel('Modal Index'); plt.ylabel('Avg Attention Weight')
    plt.title('Frequency Attention Weights')
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def plot_gnn_coupling(edge_attn_tensor, save_path):
    # edge_attn_tensor: [N, M, M]
    mean_edge = edge_attn_tensor.mean(dim=0).numpy()  # [M, M]
    plt.figure(figsize=(6,5))
    sns.heatmap(mean_edge, annot=True, cmap='coolwarm')
    plt.title('Learned Modal Coupling Strength (GAT Attention)') 
    plt.xlabel('Modal'); plt.ylabel('Modal')
    plt.tight_layout(); plt.savefig(save_path); plt.close()

def save_classification_report(y_true, y_pred, save_path):
    report = classification_report(y_true, y_pred, digits=4)
    with open(save_path, 'w') as f:
        f.write(report)
