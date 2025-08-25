import torch, os, math
from config import Config
from dataset import get_dataloaders
from model import DFTSRNet
from focal_loss import FocalLoss
from train_eval import (
    train, evaluate,
    plot_confusion_matrix, plot_train_curve,
    plot_attention_weights, plot_gnn_coupling,
    save_classification_report
)

# ------------- Setup -------------
device = Config.DEVICE
os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

# Data loaders
data_dir = os.path.join(Config.DATA_ROOT, "Dataset_A")
train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=Config.BATCH_SIZE)

# Model / loss / optimizer
model = DFTSRNet(in_channels=Config.INPUT_CHANNELS, num_classes=Config.NUM_CLASSES).to(device)
criterion = FocalLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)

# Scheduler: warmup + cosine decay
num_batches = len(train_loader)
warmup_steps = Config.WARMUP_EPOCHS * num_batches
total_steps = Config.NUM_EPOCHS * num_batches

def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / max(1, warmup_steps)
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Logging containers
train_losses, val_accs = [], []

# ------------- Training Loop -------------
for epoch in range(Config.NUM_EPOCHS):
    loss, train_acc, y_true_train, y_pred_train, attn_train, graph_feat_train, edge_attn_train = train(
        model, train_loader, criterion, optimizer, device, epoch, Config.OUTPUT_DIR, scheduler=scheduler
    )
    val_acc, y_true_val, y_pred_val, attn_val, graph_feat_val, edge_attn_val = evaluate(model, val_loader, device)
    train_losses.append(loss)
    val_accs.append(val_acc)
    print(f"[Epoch {epoch+1}/{Config.NUM_EPOCHS}] Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

    # periodic saves
    if (epoch + 1) % 10 == 0 or epoch == 0:
        plot_train_curve(train_losses, val_accs, os.path.join(Config.OUTPUT_DIR, 'train_curve.png'))
        plot_confusion_matrix(y_true_val, y_pred_val, os.path.join(Config.OUTPUT_DIR, f'confusion_epoch{epoch+1}.png'))
        plot_attention_weights(attn_val, os.path.join(Config.OUTPUT_DIR, f'attn_epoch{epoch+1}.png'))
        plot_gnn_coupling(edge_attn_val, os.path.join(Config.OUTPUT_DIR, f'gnn_coupling_epoch{epoch+1}.png'))
        save_classification_report(y_true_val, y_pred_val, os.path.join(Config.OUTPUT_DIR, f'class_report_epoch{epoch+1}.txt'))

# ------------- Final Evaluation & Save -------------
val_acc, y_true_val, y_pred_val, attn_val, graph_feat_val, edge_attn_val = evaluate(model, val_loader, device)
plot_confusion_matrix(y_true_val, y_pred_val, os.path.join(Config.OUTPUT_DIR, 'confusion_final.png'))
plot_attention_weights(attn_val, os.path.join(Config.OUTPUT_DIR, 'attention_final.png'))
plot_gnn_coupling(edge_attn_val, os.path.join(Config.OUTPUT_DIR, 'gnn_coupling_final.png'))
save_classification_report(y_true_val, y_pred_val, os.path.join(Config.OUTPUT_DIR, 'classification_report_final.txt'))
plot_train_curve(train_losses, val_accs, os.path.join(Config.OUTPUT_DIR, 'train_curve_final.png'))

# Test set
test_acc, y_true_test, y_pred_test, attn_test, graph_feat_test, edge_attn_test = evaluate(model, test_loader, device)
print(f"Test Accuracy: {test_acc:.4f}")
plot_confusion_matrix(y_true_test, y_pred_test, os.path.join(Config.OUTPUT_DIR, 'confusion_test.png'))
plot_attention_weights(attn_test, os.path.join(Config.OUTPUT_DIR, 'attention_test.png'))
plot_gnn_coupling(edge_attn_test, os.path.join(Config.OUTPUT_DIR, 'gnn_coupling_test.png'))
save_classification_report(y_true_test, y_pred_test, os.path.join(Config.OUTPUT_DIR, 'classification_report_test.txt'))

# Save model checkpoint
torch.save(model.state_dict(), os.path.join(Config.OUTPUT_DIR, 'dftsrnet_final.pth'))
