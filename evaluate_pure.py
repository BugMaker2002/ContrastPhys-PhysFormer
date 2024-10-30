import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, resample
from utils_sig import hr_fft, butter_bandpass, normalize
import os

def MyEval(HR_pr, HR_rel):
    HR_pr = np.array(HR_pr).reshape(-1)
    HR_rel = np.array(HR_rel).reshape(-1)
    temp = HR_pr - HR_rel
    me = np.mean(temp)
    std = np.std(temp)
    mae = np.sum(np.abs(temp))/len(temp)
    rmse = np.sqrt(np.sum(np.power(temp, 2))/len(temp))
    mer = np.mean(np.abs(temp) / HR_rel)
    p = np.sum((HR_pr - np.mean(HR_pr))*(HR_rel - np.mean(HR_rel))) / (
                0.01 + np.linalg.norm(HR_pr - np.mean(HR_pr), ord=2) * np.linalg.norm(HR_rel - np.mean(HR_rel), ord=2))
    print('| me: %.4f' % me,
          '| std: %.4f' % std,
          '| mae: %.4f' % mae,
          '| rmse: %.4f' % rmse,
          '| mer: %.4f' % mer,
          '| p: %.4f' % p
          )
    return me, std, mae, rmse, mer, p

root_dir = "./results_cnn_pure/2/1"
for file in os.listdir(root_dir):
    
    if not file.endswith(".npy"):
        continue
    print(file)
    data = np.load(os.path.join(root_dir, file), allow_pickle=True)
    rppg_list = data.item().get('rppg_list')
    bvp_list = data.item().get('bvp_list')
    
    # 展平信号
    rppg = np.concatenate(rppg_list)
    # print(rppg.shape) # (1500,)
    bvp = np.concatenate(bvp_list)

    # 标准化信号
    rppg_norm = normalize(rppg)
    bvp_norm = normalize(bvp)
    

    # 计算信号长度和采样率
    signal_length = len(rppg_norm)
    fs = 30  # 假设采样率为 30Hz

    # 对标准化信号进行滤波
    rppg_filtered = butter_bandpass(rppg_norm, lowcut=0.6, highcut=4, fs=fs)
    bvp_filtered = butter_bandpass(bvp_norm, lowcut=0.6, highcut=4, fs=fs)

    # 计算预测心率和地面真值心率
    hr_rppg, rppg_psd, rppg_hr = hr_fft(rppg_filtered, fs=fs)
    hr_bvp, bvp_psd, bvp_hr = hr_fft(bvp_filtered, fs=fs)

    # 计算评估指标
    mae = np.mean(np.abs(rppg_filtered - bvp_filtered))
    mse = np.mean((rppg_filtered - bvp_filtered) ** 2)
    rmse = np.sqrt(mse)
    
    r = np.corrcoef(rppg_psd, bvp_psd)[0, 1]

    # 绘制时域图和频谱图
    fig, (ax1, ax2) = plt.subplots(2, figsize=(20, 10))

    # 时域图
    time = np.arange(signal_length) / fs
    ax1.plot(time, rppg_filtered, label='Predicted Signal')
    ax1.plot(time, bvp_filtered, label='Ground Truth Signal')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Predicted Signal vs Ground Truth Signal')
    ax1.legend()
    ax1.grid('on')

    # 频谱图
    ax2.plot(rppg_hr, rppg_psd, label='Predicted Signal')
    ax2.plot(bvp_hr, bvp_psd, label='Ground Truth Signal')
    ax2.set_xlabel('Heart Rate (bpm)')
    ax2.set_ylabel('Power Spectral Density')
    ax2.set_xlim([40, 200])
    ax2.set_title('Power Spectral Density of Predicted Signal vs Ground Truth Signal')
    ax2.legend()
    ax2.grid('on')

    # 显示评估指标和预测心率
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"Pearson's Correlation Coefficient (r): {r:.4f}")
    print(f"Predicted Heart Rate: {hr_rppg:.2f} bpm")
    print(f"Ground Truth Heart Rate: {hr_bvp:.2f} bpm")

    plt.show()