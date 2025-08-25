import os
import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False


class CWRUDatasetSplitter:
    def __init__(self, root_dir, save_dir=None):
        self.root_dir = os.path.abspath(root_dir)
        self.save_dir = os.path.abspath(save_dir) if save_dir else os.path.join(self.root_dir, "SplitDataset")
        os.makedirs(self.save_dir, exist_ok=True)
        
        print(f"=== 初始化数据集划分器 ===")
        print(f"源数据集目录: {self.root_dir}")
        print(f"划分结果保存目录: {self.save_dir}")
        print(f"源目录是否存在: {os.path.exists(self.root_dir)}")
        
        self.label_map = {
            ("Normal", "0"): 0,
            ("Ball", "0.007"): 1, ("Ball", "0.014"): 2, ("Ball", "0.021"): 3,
            ("InnerRing", "0.007"): 4, ("InnerRing", "0.014"): 5, ("InnerRing", "0.021"): 6,
            ("OuterRing", "0.007"): 7, ("OuterRing", "0.014"): 8, ("OuterRing", "0.021"): 9,
        }
        
        self.dataset_info = {
            "Dataset_A": {"load": 0, "rpm": 1797},
            "Dataset_B": {"load": 1, "rpm": 1772},
            "Dataset_C": {"load": 2, "rpm": 1750},
        }
        
        self.sample_counts = {"train": 50, "val": 25, "test": 25}
        self.tfr_length = 1024  # TFR长度
        
        # 西储大学数据集可能的信号键名列表（按优先级排序），这里重点修正构造键名的逻辑
        self.possible_signal_keys = [
            # 驱动端振动信号（Drive End），这里动态构造带前导0的键名
            lambda fname: f"X{fname.zfill(3)}_DE_time",  # 补全前导0，让97变成097，构造X097_DE_time
            "DE_time",  # 通用格式
            
            # 其他可能的键名可继续补充...
        ]

    def load_mat_signal(self, mat_path):
        """智能加载.mat文件中的振动信号，自动探测可能的信号键，重点修正键名构造"""
        try:
            print(f"  正在加载：{mat_path}")
            mat_data = sio.loadmat(mat_path)
            file_name = os.path.splitext(os.path.basename(mat_path))[0]  # 获取文件名，如97、105
            
            # 打印.mat文件中的所有键（用于调试）
            all_keys = list(mat_data.keys())
            print(f"  .mat文件中的可用键: {all_keys}")
            
            # 尝试所有可能的信号键
            for key_pattern in self.possible_signal_keys:
                if callable(key_pattern):  # 如果是函数，传入文件名生成键名（会补全前导0）
                    signal_key = key_pattern(file_name)
                else:  # 直接使用字符串键名
                    signal_key = key_pattern
                
                if signal_key in mat_data:
                    signal = mat_data[signal_key].flatten()
                    print(f"  成功找到信号键: {signal_key}，信号长度: {len(signal)}")
                    return signal
            
            # 如果都没找到，尝试找到包含"time"的最长键（启发式方法）
            time_keys = [k for k in all_keys if "time" in k.lower()]
            if time_keys:
                # 选择最长的键（通常包含更多信息）
                signal_key = max(time_keys, key=len)
                signal = mat_data[signal_key].flatten()
                print(f"  使用启发式方法找到信号键: {signal_key}，信号长度: {len(signal)}")
                return signal
            
            # 实在找不到，报错并提供建议
            raise KeyError(f"无法在.mat文件中找到振动信号键。"
                          f"可用键: {all_keys}。"
                          f"尝试过的键模式: {[key(file_name) if callable(key) else key for key in self.possible_signal_keys]}")
        
        except Exception as e:
            print(f"  加载失败: {str(e)}")
            raise

    def split_tfr_from_signal(self, signal):
        """从信号中分割非重叠的TFR（1024点）"""
        tfr_list = []
        num_points = len(signal)
        num_tfr = num_points // self.tfr_length
        print(f"  可分割TFR数量: {num_tfr}（每个1024点）")
        
        if num_tfr < 100:  # 至少需要100个TFR（100+100+100）
            print(f"  警告: TFR数量不足（{num_tfr} < 100），可能导致后续错误")
        
        for i in range(num_tfr):
            tfr = signal[i * self.tfr_length: (i + 1) * self.tfr_length]
            tfr_list.append(tfr)
        
        return np.array(tfr_list)

    def split_dataset(self):
        """划分A/B/C三个数据集"""
        for dataset_name, info in self.dataset_info.items():
            load = info["load"]
            rpm = info["rpm"]
            print(f"\n===== 开始划分 {dataset_name}（负载{load}，转速{rpm}） =====")
            
            all_data = {"train": [], "val": [], "test": []}
            all_labels = {"train": [], "val": [], "test": []}
            
            for (fault_type, diameter), label in self.label_map.items():
                print(f"\n--- 处理类别：{fault_type}（直径{diameter}英寸），标签{label} ---")
                
                source_dir = os.path.join(
                    self.root_dir, fault_type,
                    f"Load_{load}_RPM_{rpm}",
                    f"FaultDiameter_{diameter}",
                    dataset_name
                )
                print(f"  源目录：{source_dir}")
                
                if not os.path.exists(source_dir):
                    raise FileNotFoundError(f"源目录不存在：{source_dir}")
                
                mat_files = [f for f in os.listdir(source_dir) if f.endswith(".mat")]
                if len(mat_files) != 1:
                    raise ValueError(f"目录下应只有1个.mat文件，实际找到{len(mat_files)}个")
                mat_path = os.path.join(source_dir, mat_files[0])
                
                # 加载信号并分割TFR
                signal = self.load_mat_signal(mat_path)
                tfrs = self.split_tfr_from_signal(signal)
                
                # 检查TFR数量并进行重复采样
                required = self.sample_counts["train"] + self.sample_counts["val"] + self.sample_counts["test"]
                if len(tfrs) < required:
                    print(f"  TFR数量不足（{len(tfrs)} < {required}），进行重复采样")
                    # 重复采样到足够数量
                    repeat_times = (required // len(tfrs)) + 1
                    tfrs_repeated = np.tile(tfrs, (repeat_times, 1))
                    tfrs = tfrs_repeated[:required]  # 截取到所需数量
                    print(f"  重复采样后TFR数量：{len(tfrs)}")
                
                # 划分数据集 - 严格按照指定数量
                np.random.seed(42)  # 确保可重复性
                indices = np.random.permutation(len(tfrs))
                
                train_indices = indices[:self.sample_counts["train"]]
                val_indices = indices[self.sample_counts["train"]:self.sample_counts["train"]+self.sample_counts["val"]]
                test_indices = indices[self.sample_counts["train"]+self.sample_counts["val"]:self.sample_counts["train"]+self.sample_counts["val"]+self.sample_counts["test"]]
                
                train_tfrs = tfrs[train_indices]
                val_tfrs = tfrs[val_indices]
                test_tfrs = tfrs[test_indices]
                
                # 加入总数据
                all_data["train"].extend(train_tfrs)
                all_labels["train"].extend([label] * len(train_tfrs))
                
                all_data["val"].extend(val_tfrs)
                all_labels["val"].extend([label] * len(val_tfrs))
                
                all_data["test"].extend(test_tfrs)
                all_labels["test"].extend([label] * len(test_tfrs))
                
                print(f"  类别处理完成：{len(train_tfrs)}+{len(val_tfrs)}+{len(test_tfrs)}")
            
            # 保存数据集
            save_dir = os.path.join(self.save_dir, dataset_name)
            os.makedirs(save_dir, exist_ok=True)
            
            for split in ["train", "val", "test"]:
                all_data[split] = np.array(all_data[split])
                all_labels[split] = np.array(all_labels[split])
                np.save(os.path.join(save_dir, f"{split}_data.npy"), all_data[split])
                np.save(os.path.join(save_dir, f"{split}_labels.npy"), all_labels[split])
                print(f"  已保存：{save_dir}/{split}_data.npy ({len(all_data[split])}个样本)")
        
        print("\n===== 所有数据集划分完成！ =====")


if __name__ == "__main__":
    root_dir = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学数据集"
    save_dir = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\西储大学实验数据集"
    
    try:
        print("开始运行数据集划分程序...")
        splitter = CWRUDatasetSplitter(root_dir, save_dir)
        splitter.split_dataset()
    except Exception as e:
        print(f"\n程序终止，错误原因：{str(e)}")