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
    """训练单个数据集"""
    print(f"\n{'='*60}")
    print(f"开始训练数据集: {dataset_name}")
    print(f"{'='*60}")
    
    # 设置路径
    data_dir = os.path.join(Config.BASE_DIR, dataset_name)
    output_dir = os.path.join(Config.OUTPUT_BASE_DIR, f"results_{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    # 检查数据集是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 数据集目录不存在: {data_dir}")
        return None
    
    # 设备和数据加载器
    device = Config.DEVICE
    print(f"使用设备: {device}")
    
    try:
        train_loader, val_loader, test_loader = get_dataloaders(
            data_dir, batch_size=Config.BATCH_SIZE
        )
        print(f"数据加载成功 - 训练集: {len(train_loader)} batches, 验证集: {len(val_loader)} batches, 测试集: {len(test_loader)} batches")
    except Exception as e:
        print(f"数据加载失败: {e}")
        return None
    
    # 模型、损失函数、优化器
    model = DFTSRNet(in_channels=Config.INPUT_CHANNELS, num_classes=Config.NUM_CLASSES).to(device)
    criterion = FocalLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # 学习率调度器: warmup + cosine decay
    num_batches = len(train_loader)
    warmup_steps = Config.WARMUP_EPOCHS * num_batches
    total_steps = Config.NUM_EPOCHS * num_batches
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练记录
    train_losses, val_accs = [], []
    best_val_acc = 0.0
    best_model_path = os.path.join(output_dir, 'best_model.pth')
    
    # 训练循环
    start_time = time.time()
    for epoch in range(Config.NUM_EPOCHS):
        # 训练
        loss, train_acc, y_true_train, y_pred_train, attn_train, graph_feat_train, edge_attn_train = train(
            model, train_loader, criterion, optimizer, device, epoch, output_dir, scheduler=scheduler
        )
        
        # 验证
        val_acc, y_true_val, y_pred_val, attn_val, graph_feat_val, edge_attn_val = evaluate(
            model, val_loader, device
        )
        
        train_losses.append(loss)
        val_accs.append(val_acc)
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
        
        print(f"[Epoch {epoch+1}/{Config.NUM_EPOCHS}] "
              f"Train Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Acc: {val_acc:.4f}, Best Val Acc: {best_val_acc:.4f}")
        
        # 定期保存可视化结果
        if (epoch + 1) % 20 == 0 or epoch == 0:
            plot_train_curve(train_losses, val_accs, os.path.join(output_dir, f'train_curve_epoch{epoch+1}.png'))
            plot_confusion_matrix(y_true_val, y_pred_val, os.path.join(output_dir, f'confusion_epoch{epoch+1}.png'))
            plot_attention_weights(attn_val, os.path.join(output_dir, f'attn_epoch{epoch+1}.png'))
            plot_gnn_coupling(edge_attn_val, os.path.join(output_dir, f'gnn_coupling_epoch{epoch+1}.png'))
            save_classification_report(y_true_val, y_pred_val, os.path.join(output_dir, f'class_report_epoch{epoch+1}.txt'))
    
    training_time = time.time() - start_time
    print(f"训练完成，用时: {training_time:.2f}秒")
    
    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load(best_model_path))
    
    # 最终验证评估
    val_acc, y_true_val, y_pred_val, attn_val, graph_feat_val, edge_attn_val = evaluate(model, val_loader, device)
    plot_confusion_matrix(y_true_val, y_pred_val, os.path.join(output_dir, 'confusion_final.png'))
    plot_attention_weights(attn_val, os.path.join(output_dir, 'attention_final.png'))
    plot_gnn_coupling(edge_attn_val, os.path.join(output_dir, 'gnn_coupling_final.png'))
    save_classification_report(y_true_val, y_pred_val, os.path.join(output_dir, 'classification_report_final.txt'))
    plot_train_curve(train_losses, val_accs, os.path.join(output_dir, 'train_curve_final.png'))
    
    # 测试集评估
    test_acc, y_true_test, y_pred_test, attn_test, graph_feat_test, edge_attn_test = evaluate(model, test_loader, device)
    print(f"最终测试准确率: {test_acc:.4f}")
    
    plot_confusion_matrix(y_true_test, y_pred_test, os.path.join(output_dir, 'confusion_test.png'))
    plot_attention_weights(attn_test, os.path.join(output_dir, 'attention_test.png'))
    plot_gnn_coupling(edge_attn_test, os.path.join(output_dir, 'gnn_coupling_test.png'))
    save_classification_report(y_true_test, y_pred_test, os.path.join(output_dir, 'classification_report_test.txt'))
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(output_dir, 'dftsrnet_final.pth'))
    
    # 保存训练摘要
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
    """主函数：训练所有数据集"""
    print("开始批量训练所有噪声数据集...")
    print(f"总共 {len(Config.DATASETS)} 个数据集")
    
    all_results = []
    total_start_time = time.time()
    
    for i, dataset_name in enumerate(Config.DATASETS):
        print(f"\n进度: {i+1}/{len(Config.DATASETS)}")
        
        try:
            result = train_single_dataset(dataset_name)
            if result:
                all_results.append(result)
                print(f"✅ {dataset_name} 训练完成")
                print(f"   最佳验证准确率: {result['best_val_acc']:.4f}")
                print(f"   最终测试准确率: {result['final_test_acc']:.4f}")
            else:
                print(f"❌ {dataset_name} 训练失败")
        except Exception as e:
            print(f"❌ {dataset_name} 训练出错: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = time.time() - total_start_time
    
    # 保存总体结果摘要
    summary_path = os.path.join(Config.OUTPUT_BASE_DIR, 'all_results_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("所有数据集训练结果摘要\n")
        f.write("=" * 50 + "\n")
        f.write(f"总训练时间: {total_time:.2f}秒\n")
        f.write(f"成功训练数据集数量: {len(all_results)}/{len(Config.DATASETS)}\n\n")
        
        for result in all_results:
            f.write(f"数据集: {result['dataset']}\n")
            f.write(f"  最佳验证准确率: {result['best_val_acc']:.4f}\n")
            f.write(f"  最终测试准确率: {result['final_test_acc']:.4f}\n")
            f.write(f"  训练时间: {result['training_time']:.2f}秒\n")
            f.write("-" * 30 + "\n")
    
    print(f"\n{'='*60}")
    print("所有数据集训练完成！")
    print(f"总用时: {total_time:.2f}秒")
    print(f"成功训练: {len(all_results)}/{len(Config.DATASETS)} 个数据集")
    print(f"结果摘要保存在: {summary_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()