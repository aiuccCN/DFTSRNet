import torch
import os
import math
import time
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

def train_single_dataset(dataset_name):
    """Train single dataset"""
    print(f"\n{'='*60}")
    print(f"Starting training dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Set paths
    data_dir = os.path.join(Config.BASE_DIR, dataset_name)
    output_dir = os.path.join(Config.OUTPUT_BASE_DIR, f"results_{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if dataset exists
    if not os.path.exists(data_dir):
        print(f"Error: Dataset directory does not exist: {data_dir}")
        return None
    
    # Device and data loaders
    device = Config.DEVICE
    print(f"Using device: {device}")
    
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir, batch_size=Config.BATCH_SIZE
        )
        print(f"Data loading successful - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
    except Exception as e:
        print(f"Data loading failed: {e}")
        return None
    
    # Model, loss function, optimizer
    model = DFTSRNet(in_channels=Config.INPUT_CHANNELS, num_classes=Config.NUM_CLASSES).to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler: warmup + cosine decay
    num_batches = len(train_loader)
    warmup_steps = Config.WARMUP_EPOCHS * num_batches
    total_steps = Config.NUM_EPOCHS * num_batches
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Training records
    train_losses, val_accs = [], []
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    
    # Training loop
    start_time = time.time()
    for epoch in range(Config.NUM_EPOCHS):
        # Training
        loss, train_acc, y_true_train, y_pred_train, attn_train, graph_feat_train, edge_attn_train = train(
            model, train_loader, criterion, optimizer, device, epoch, output_dir, scheduler=scheduler
        )
        
        # Validation
        val_acc, y_true_val, y_pred_val, attn_val, graph_feat_val, edge_attn_val = evaluate(
            model, val_loader, device
        )
        
        train_losses.append(loss)
        val_accs.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
        
        print(f"[Epoch {epoch+1}/{Config.NUM_EPOCHS}] "
              f"Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Acc: {val_acc:.4f}, Best Val Acc: {best_val_acc:.4f}")
        
        # Periodically save visualization results
        if (epoch + 1) % 20 == 0 or epoch == 0:
            plot_train_curve(train_losses, val_accs, os.path.join(output_dir, f'train_curve_epoch{epoch+1}.png'))
            plot_confusion_matrix(y_true_val, y_pred_val, os.path.join(output_dir, f'confusion_epoch{epoch+1}.png'))
            plot_attention_weights(attn_val, os.path.join(output_dir, f'attn_epoch{epoch+1}.png'))
            plot_gnn_coupling(edge_attn_val, os.path.join(output_dir, f'gnn_coupling_epoch{epoch+1}.png'))
            save_classification_report(y_true_val, y_pred_val, os.path.join(output_dir, f'class_report_epoch{epoch+1}.txt'))
    
    training_time = time.time() - start_time
    print(f"Training completed, time taken: {training_time:.2f} seconds")
    
    # Load best model for final evaluation
    model.load_state_dict(torch.load(best_model_path))
    
    # Final validation evaluation
    val_acc, y_true_val, y_pred_val, attn_val, graph_feat_val, edge_attn_val = evaluate(model, val_loader, device)
    plot_confusion_matrix(y_true_val, y_pred_val, os.path.join(output_dir, 'confusion_final.png'))
    plot_attention_weights(attn_val, os.path.join(output_dir, 'attention_final.png'))
    plot_gnn_coupling(edge_attn_val, os.path.join(output_dir, 'gnn_coupling_final.png'))
    save_classification_report(y_true_val, y_pred_val, os.path.join(output_dir, 'classification_report_final.txt'))
    plot_train_curve(train_losses, val_accs, os.path.join(output_dir, 'train_curve_final.png'))
    
    # Test set evaluation
    test_acc, y_true_test, y_pred_test, attn_test, graph_feat_test, edge_attn_test = evaluate(model, test_loader, device)
    print(f"Final test accuracy: {test_acc:.4f}")
    
    plot_confusion_matrix(y_true_test, y_pred_test, os.path.join(output_dir, 'confusion_test.png'))
    plot_attention_weights(attn_test, os.path.join(output_dir, 'attention_test.png'))
    plot_gnn_coupling(edge_attn_test, os.path.join(output_dir, 'gnn_coupling_test.png'))
    save_classification_report(y_true_test, y_pred_test, os.path.join(output_dir, 'classification_report_test.txt'))
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(output_dir, 'dftsrnet_final.pth'))
    
    # Save training summary
    summary = {
        'dataset': dataset_name,
        'best_val_acc': best_val_acc,
        'final_test_acc': test_acc,
        'training_time': training_time,
        'num_epochs': Config.NUM_EPOCHS
    }
    
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    
    return summary

def main():
    """Main function: train all datasets"""
    print("Starting batch training of all noise datasets...")
    print(f"Total {len(Config.DATASETS)} datasets")
    
    all_results = []
    total_start_time = time.time()
    
    for i, dataset_name in enumerate(Config.DATASETS):
        print(f"\nProgress: {i+1}/{len(Config.DATASETS)}")
        
        try:
            result = train_single_dataset(dataset_name)
            if result:
                all_results.append(result)
                print(f"✅ {dataset_name} training completed")
                print(f"   Best validation accuracy: {result['best_val_acc']:.4f}")
                print(f"   Final test accuracy: {result['final_test_acc']:.4f}")
            else:
                print(f"❌ {dataset_name} training failed")
        except Exception as e:
            print(f"❌ {dataset_name} training error: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start_time
    
    # Save overall results summary
    summary_path = os.path.join(Config.OUTPUT_BASE_DIR, 'all_results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("All Datasets Training Results Summary\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total training time: {total_time:.2f} seconds\n")
        f.write(f"Successfully trained datasets: {len(all_results)}/{len(Config.DATASETS)}\n\n")
        
        for result in all_results:
            f.write(f"Dataset: {result['dataset']}\n")
            f.write(f"  Best validation accuracy: {result['best_val_acc']:.4f}\n")
            f.write(f"  Final test accuracy: {result['final_test_acc']:.4f}\n")
            f.write(f"  Training time: {result['training_time']:.2f} seconds\n")
            f.write("-" * 30 + "\n")
    
    print(f"\n{'='*60}")
    print("All datasets training completed!")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Successfully trained: {len(all_results)}/{len(Config.DATASETS)} datasets")
    print(f"Results summary saved in: {summary_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()