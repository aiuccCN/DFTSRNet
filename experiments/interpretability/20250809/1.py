import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import fft, fftfreq
from scipy.signal import stft
from scipy.linalg import toeplitz, lstsq

# 设置中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False

# 文献核心参数
GAUSSIAN_EPS_NEGENTROPY = 1.1525  # 高斯SE负熵（Table 1）
GAUSSIAN_ES_NEGENTROPY = 0.4228  # 高斯SES负熵（Table 1）

# 西储大学轴承参数（类别6：内环故障，0.021英寸，1797RPM）
BEARING_PARAMS = {
    "d_i": 25.0,   # 内圈直径 (mm)
    "d_o": 52.0,   # 外圈直径 (mm)
    "d_b": 7.94,   # 滚珠直径 (mm)
    "n_b": 9,      # 滚珠数
    "theta": 0,    # 接触角 (°)
    "rpm": 1797    # 转速 (RPM)
}

# 计算内环故障频率BPFI
def compute_bpfi(params):
    rpm = params["rpm"]
    n_b = params["n_b"]
    d_i = params["d_i"]
    d_b = params["d_b"]
    theta = np.radians(params["theta"])  # 转换为弧度
    bpfi = (n_b / 2) * (rpm / 60) * (1 + (d_b / d_i) * np.cos(theta))
    return bpfi

BPFI = compute_bpfi(BEARING_PARAMS)
print(f"内环故障频率BPFI: {BPFI:.1f}Hz")  # 约177.6Hz


def prewhiten(signal_data, order=20):
    """预白化（Section 4.5，阶数适配机械信号）"""
    r = np.correlate(signal_data, signal_data, mode='full')[len(signal_data)-1:len(signal_data)+order]
    R = toeplitz(r[:order])
    r = r[1:order+1]
    a, _, _, _ = lstsq(R, r, cond=None)  # 兼容旧版SciPy
    whitened = signal.lfilter([1] + [-a_i for a_i in a], 1, signal_data)
    return whitened

def synchronous_averaging(signal_data, fs, period):
    """修复残差长度不匹配问题：强制扩展平均信号到原长度"""
    period_samples = int(period * fs)
    n_periods = len(signal_data) // period_samples
    # 仅对完整周期部分计算平均
    signal_truncated = signal_data[:n_periods * period_samples]
    averaged = np.zeros(period_samples)
    for i in range(n_periods):
        start = i * period_samples
        end = start + period_samples
        averaged += signal_truncated[start:end] / n_periods
    
    # 扩展平均信号到原信号长度（填充最后一个周期的部分）
    averaged_extended = np.tile(averaged, n_periods + 1)[:len(signal_data)]
    residual = signal_data - averaged_extended
    return residual

def get_stft_resolutions(fs, signal_length, max_level=6):
    """确保窗口长度≤信号长度，避免STFT报错"""
    levels = [0, 1, 2, 2.585, 3, 3.585, 4, 4.585, 5][:max_level+1]
    delta_f = [fs / (2 ** i) for i in levels]
    window_lengths = [int(fs / df) for df in delta_f]
    # 窗口长度≤信号长度（保证至少1个窗口）
    window_lengths = [min(wl, signal_length) for wl in window_lengths]
    window_lengths = [wl if wl % 2 == 0 else wl + 1 for wl in window_lengths]
    return delta_f, window_lengths

def compute_squared_envelope(signal_data, fs, window_length):
    """平方包络计算（STFT，文献Section 2.2.1）"""
    nperseg = window_length
    noverlap = int(nperseg * 0.75)  # 75%重叠
    nfft = nperseg * 2 if nperseg * 2 <= len(signal_data) else len(signal_data)
    f, t, Zxx = stft(
        signal_data, fs=fs, window='hann', nperseg=nperseg,
        noverlap=noverlap, nfft=nfft, return_onesided=True
    )
    return f, t, np.abs(Zxx)**2  # 平方包络

def compute_se_negentropy(squared_env):
    """SE负熵（文献公式11，增强稳定性）"""
    eps_sq = squared_env ** 2
    eps_sq_mean = np.mean(eps_sq, axis=1, keepdims=True)
    eps_norm = eps_sq / (eps_sq_mean + 1e-10)
    eps_norm = np.clip(eps_norm, 1e-10, None)  # 避免log(0)
    negentropy = np.mean(eps_norm * np.log(eps_norm), axis=1)
    return negentropy

def compute_ses_negentropy(squared_env):
    """SES负熵（文献公式12，增强稳定性）"""
    ses = np.abs(fft(squared_env, axis=1)) ** 2
    freq_ses = fftfreq(squared_env.shape[1], d=1/squared_env.shape[1])
    zero_freq_idx = np.argmin(np.abs(freq_ses))
    ses = np.delete(ses, zero_freq_idx, axis=1)
    ses_mean = np.mean(ses, axis=1, keepdims=True)
    ses_norm = ses / (ses_mean + 1e-10)
    ses_norm = np.clip(ses_norm, 1e-10, None)  # 避免log(0)
    negentropy = np.mean(ses_norm * np.log(ses_norm), axis=1)
    return negentropy

def compute_infograms(signal_data, fs, max_level=6):
    """完整infogram计算流程（含同步平均）"""
    # 1. 预处理
    whitened = prewhiten(signal_data)
    # 2. 同步平均（周期=60/RPM）
    period = 60 / BEARING_PARAMS["rpm"]  # 旋转周期（秒）
    whitened = synchronous_averaging(whitened, fs, period)
    # 3. 多分辨率STFT
    delta_f_list, window_lengths = get_stft_resolutions(fs, len(signal_data), max_level)
    all_f, se_neg_list, ses_neg_list = [], [], []
    
    for wl, df in zip(window_lengths, delta_f_list):
        f, t, squared_env = compute_squared_envelope(whitened, fs, wl)
        all_f.append(f)
        # 负熵计算+归一化
        se_neg = compute_se_negentropy(squared_env) - GAUSSIAN_EPS_NEGENTROPY
        ses_neg = compute_ses_negentropy(squared_env) - GAUSSIAN_ES_NEGENTROPY
        se_neg_list.append(se_neg)
        ses_neg_list.append(ses_neg)
    
    ave_neg_list = [(se + ses) / 2 for se, ses in zip(se_neg_list, ses_neg_list)]
    return all_f, delta_f_list, se_neg_list, ses_neg_list, ave_neg_list

def plot_infograms(all_f, delta_f, se_neg, ses_neg, ave_neg, fs, signal_data, bpfi):
    """可视化（对齐文献+标注故障频率）"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('文献方法实现的Infogram分析（类别6：内环故障，BPFI≈{:.1f}Hz）'.format(bpfi), fontsize=16)
    
    # 原始信号（前1秒，突出冲击周期性）
    t = np.arange(len(signal_data)) / fs
    axes[0, 0].plot(t, signal_data)
    axes[0, 0].set_xlim(0, 1)  # 强制展示1秒
    axes[0, 0].set_title('原始振动信号（前1秒）')
    axes[0, 0].set_xlabel('时间 (s)')
    axes[0, 0].set_ylabel('振幅')
    
    # 故障频率标记（BPFI及其谐波）
    freq_marks = [bpfi * i for i in range(1, 6)]  # 1~5倍频
    
    def plot_single(ax, data_list, title):
        for i, (f, data) in enumerate(zip(all_f, data_list)):
            # 截取有效频率范围（0~1000Hz，覆盖BPFI谐波）
            mask = (f >= 0) & (f <= 1000)
            f_slice = f[mask]
            data_slice = data[mask]
            
            # 分辨率区间
            y_low = delta_f[i] * 0.9
            y_high = delta_f[i] * 1.1
            y = [y_low, y_high]
            
            # 绘制热力图
            C = np.expand_dims(data_slice[:-1], axis=0)
            mesh = ax.pcolormesh(f_slice, y, C, cmap='jet', alpha=0.8, shading='flat')
            
            # 标记故障频率
            for fm in freq_marks:
                if fm < 1000:
                    ax.axvline(fm, color='white', linestyle='--', alpha=0.8)
        
        ax.set_xlim(0, 1000)
        ax.set_yscale('log')
        ax.set_ylim(min(delta_f)*0.8, max(delta_f)*1.2)
        ax.set_xlabel('频率 (Hz)')
        ax.set_ylabel('频率分辨率Δf (Hz)')
        ax.set_title(title)
        plt.colorbar(mesh, ax=ax, label='负熵 (nats)')
    
    plot_single(axes[0, 1], se_neg, 'SE Infogram（平方包络负熵）')
    plot_single(axes[1, 0], ses_neg, 'SES Infogram（平方包络谱负熵）')
    plot_single(axes[1, 1], ave_neg, 'AVE Infogram（平均负熵）')
    
    plt.tight_layout()
    return fig

def main():
    # 加载自定义的类别6时序信号（class6.npy）
    data_path = r"E:\PA_SAD_VKCNN_UADP_AFN\论文全流程解决方案\AMSN框架\数据集\class6.npy"
    signal_data = np.load(data_path)
    
    # 西储大学数据集采样频率（DE通道为12kHz）
    fs = 12000  
    
    # 截取前1秒（12000点）保证统计稳定性
    signal_length = len(signal_data)
    if signal_length >= 12000:
        signal_data = signal_data[:12000]
        print(f"截取后信号长度: {len(signal_data)}（1秒时长，符合文献统计要求）")
    else:
        raise ValueError(f"信号长度不足1秒！当前长度: {signal_length}")
    
    # 计算infogram（max_level=6，复现文献多分辨率分析）
    all_f, delta_f, se_neg, ses_neg, ave_neg = compute_infograms(signal_data, fs, max_level=6)
    
    # 可视化并保存结果
    fig = plot_infograms(all_f, delta_f, se_neg, ses_neg, ave_neg, fs, signal_data, BPFI)
    fig.savefig('类别6_infogram结果.png', dpi=300, bbox_inches='tight')
    print("Infogram结果已保存为：类别6_infogram结果.png")
    plt.show()

if __name__ == "__main__":
    main()