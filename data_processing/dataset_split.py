import os
import shutil
import matplotlib.pyplot as plt

class CWRUDataSorter:
    """西储大学轴承数据集分类器（调试版，带详细日志）"""
    
    def __init__(self, source_dir, target_dir=None):
        self.source_dir = os.path.abspath(source_dir)
        self.target_dir = os.path.abspath(target_dir) if target_dir else \
                          os.path.join(os.path.dirname(self.source_dir), "SortedData")
        
        # 强制创建目标目录
        os.makedirs(self.target_dir, exist_ok=True)
        print(f"=== 初始化信息 ===")
        print(f"源目录: {self.source_dir}")
        print(f"目标目录: {self.target_dir}")
        print(f"源目录是否存在: {os.path.exists(self.source_dir)}")
        print(f"源目录是否可访问: {os.access(self.source_dir, os.R_OK)}")
        
        # 核心文件映射表（与之前一致）
        self.file_info_map = {
            # 正常状态
            "97.mat": {"fault_type": "Normal", "fault_diameter": "0", "load": 0, "rpm": 1797, "dataset": "Dataset_A"},
            "98.mat": {"fault_type": "Normal", "fault_diameter": "0", "load": 1, "rpm": 1772, "dataset": "Dataset_B"},
            "99.mat": {"fault_type": "Normal", "fault_diameter": "0", "load": 2, "rpm": 1750, "dataset": "Dataset_C"},
            
            # 滚珠故障（Ball）
            "118.mat": {"fault_type": "Ball", "fault_diameter": "0.007", "load": 0, "rpm": 1797, "dataset": "Dataset_A"},
            "119.mat": {"fault_type": "Ball", "fault_diameter": "0.007", "load": 1, "rpm": 1772, "dataset": "Dataset_B"},
            "120.mat": {"fault_type": "Ball", "fault_diameter": "0.007", "load": 2, "rpm": 1750, "dataset": "Dataset_C"},
            "185.mat": {"fault_type": "Ball", "fault_diameter": "0.014", "load": 0, "rpm": 1797, "dataset": "Dataset_A"},
            "186.mat": {"fault_type": "Ball", "fault_diameter": "0.014", "load": 1, "rpm": 1772, "dataset": "Dataset_B"},
            "187.mat": {"fault_type": "Ball", "fault_diameter": "0.014", "load": 2, "rpm": 1750, "dataset": "Dataset_C"},
            "222.mat": {"fault_type": "Ball", "fault_diameter": "0.021", "load": 0, "rpm": 1797, "dataset": "Dataset_A"},
            "223.mat": {"fault_type": "Ball", "fault_diameter": "0.021", "load": 1, "rpm": 1772, "dataset": "Dataset_B"},
            "224.mat": {"fault_type": "Ball", "fault_diameter": "0.021", "load": 2, "rpm": 1750, "dataset": "Dataset_C"},
            
            # 内圈故障（InnerRing）
            "105.mat": {"fault_type": "InnerRing", "fault_diameter": "0.007", "load": 0, "rpm": 1797, "dataset": "Dataset_A"},
            "106.mat": {"fault_type": "InnerRing", "fault_diameter": "0.007", "load": 1, "rpm": 1772, "dataset": "Dataset_B"},
            "107.mat": {"fault_type": "InnerRing", "fault_diameter": "0.007", "load": 2, "rpm": 1750, "dataset": "Dataset_C"},
            "169.mat": {"fault_type": "InnerRing", "fault_diameter": "0.014", "load": 0, "rpm": 1797, "dataset": "Dataset_A"},
            "170.mat": {"fault_type": "InnerRing", "fault_diameter": "0.014", "load": 1, "rpm": 1772, "dataset": "Dataset_B"},
            "171.mat": {"fault_type": "InnerRing", "fault_diameter": "0.014", "load": 2, "rpm": 1750, "dataset": "Dataset_C"},
            "209.mat": {"fault_type": "InnerRing", "fault_diameter": "0.021", "load": 0, "rpm": 1797, "dataset": "Dataset_A"},
            "210.mat": {"fault_type": "InnerRing", "fault_diameter": "0.021", "load": 1, "rpm": 1772, "dataset": "Dataset_B"},
            "211.mat": {"fault_type": "InnerRing", "fault_diameter": "0.021", "load": 2, "rpm": 1750, "dataset": "Dataset_C"},
            
            # 外圈故障（6点方向）
            "130.mat": {"fault_type": "OuterRing", "fault_diameter": "0.007", "load": 0, "rpm": 1797, "dataset": "Dataset_A"},
            "131.mat": {"fault_type": "OuterRing", "fault_diameter": "0.007", "load": 1, "rpm": 1772, "dataset": "Dataset_B"},
            "132.mat": {"fault_type": "OuterRing", "fault_diameter": "0.007", "load": 2, "rpm": 1750, "dataset": "Dataset_C"},
            "197.mat": {"fault_type": "OuterRing", "fault_diameter": "0.014", "load": 0, "rpm": 1797, "dataset": "Dataset_A"},
            "198.mat": {"fault_type": "OuterRing", "fault_diameter": "0.014", "load": 1, "rpm": 1772, "dataset": "Dataset_B"},
            "199.mat": {"fault_type": "OuterRing", "fault_diameter": "0.014", "load": 2, "rpm": 1750, "dataset": "Dataset_C"},
            "234.mat": {"fault_type": "OuterRing", "fault_diameter": "0.021", "load": 0, "rpm": 1797, "dataset": "Dataset_A"},
            "235.mat": {"fault_type": "OuterRing", "fault_diameter": "0.021", "load": 1, "rpm": 1772, "dataset": "Dataset_B"},
            "236.mat": {"fault_type": "OuterRing", "fault_diameter": "0.021", "load": 2, "rpm": 1750, "dataset": "Dataset_C"},
        }
        
        self.stats = {
            "Normal": 0, "Ball": 0, "InnerRing": 0, "OuterRing": 0, "skipped": 0
        }

    def sort_data(self, copy_files=True):
        print("\n=== 开始处理数据 ===")
        print(f"需要处理的文件总数: {len(self.file_info_map)}")
        
        # 先检查所有需要处理的文件是否存在
        missing_files = []
        for filename in self.file_info_map.keys():
            # 搜索文件在源目录中的位置
            found = False
            for root, _, files in os.walk(self.source_dir):
                if filename in files:
                    found = True
                    break
            if not found:
                missing_files.append(filename)
        
        if missing_files:
            print(f"\n警告: 以下文件在源目录中未找到:")
            for f in missing_files:
                print(f"  - {f}")
        else:
            print("\n所有需要处理的文件均已找到，开始复制...")
        
        # 处理文件
        processed_files = 0
        for root, _, files in os.walk(self.source_dir):
            print(f"\n正在扫描目录: {root}")
            dir_files = [f for f in files if f.endswith('.mat')]
            print(f"该目录下的MAT文件数量: {len(dir_files)}")
            
            # 仅处理目标目录
            if "12k Drive End Bearing Fault Data" in root or "Normal Baseline Data" in root:
                for file in dir_files:
                    if file in self.file_info_map:
                        print(f"找到目标文件: {file}")
                        file_info = self.file_info_map[file]
                        
                        target_subdir = os.path.join(
                            self.target_dir,
                            file_info["fault_type"],
                            f"Load_{file_info['load']}_RPM_{file_info['rpm']}",
                            f"FaultDiameter_{file_info['fault_diameter']}",
                            file_info["dataset"]
                        )
                        os.makedirs(target_subdir, exist_ok=True)
                        
                        source_path = os.path.join(root, file)
                        target_path = os.path.join(target_subdir, file)
                        
                        try:
                            if copy_files:
                                shutil.copy2(source_path, target_path)
                            else:
                                shutil.move(source_path, target_path)
                            
                            self.stats[file_info["fault_type"]] += 1
                            processed_files += 1
                            print(f"已处理: {file} -> 目标目录")
                        except Exception as e:
                            print(f"处理失败 {file}: {str(e)}")
                    else:
                        self.stats["skipped"] += 1
        
        self._print_stats()
        self.visualize_dataset_distribution()

    def _print_stats(self):
        print("\n=== 分类统计结果 ===")
        total_processed = sum(self.stats.values()) - self.stats["skipped"]
        print(f"正常状态: {self.stats['Normal']} 个")
        print(f"滚珠故障: {self.stats['Ball']} 个")
        print(f"内圈故障: {self.stats['InnerRing']} 个")
        print(f"外圈故障: {self.stats['OuterRing']} 个")
        print(f"跳过的文件: {self.stats['skipped']} 个")
        print(f"总处理文件: {total_processed} 个")

    def visualize_dataset_distribution(self):
        if sum(self.stats.values()) == 0:
            print("没有数据可可视化")
            return
        
        plot_data = {k: v for k, v in self.stats.items() if k != "skipped"}
        plt.figure(figsize=(10, 6))
        plt.bar(plot_data.keys(), plot_data.values(), color=['green', 'red', 'blue', 'orange'])
        plt.title('数据集分布')
        plt.xlabel('故障类型')
        plt.ylabel('文件数量')
        plt.xticks(rotation=0)
        plt.tight_layout()
        
        plot_path = os.path.join(self.target_dir, "dataset_distribution.png")
        plt.savefig(plot_path)
        print(f"分布图已保存至: {plot_path}")
        plt.close()  # 不显示图表，避免阻塞脚本

if __name__ == "__main__":
    source_dir = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\西储大学数据（官方版）"
    target_dir = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学数据集"
    
    try:
        sorter = CWRUDataSorter(source_dir, target_dir)
        sorter.sort_data(copy_files=True)
        print("\n=== 处理完成 ===")
    except Exception as e:
        print(f"执行失败: {str(e)}")