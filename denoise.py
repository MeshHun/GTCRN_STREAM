"""
author: Mesh_Hun
created: 2025-10-21
description: Use GTCRN onnx model in stream audio
license: MIT
"""
import os
import time
import numpy as np
import sounddevice as sd
import sys
import librosa
import onnxruntime
from librosa import istft
import soundfile as sf

# 配置参数
sr = 16000  # 采样率
n_fft = 512
hop_length = 256  # 帧移
win_length = 512

# 关键修复：使用满足COLA条件的窗函数
# 对于50%重叠，汉宁窗满足完美重构条件
window = np.hanning(win_length + 1)[:-1]  # 标准的汉宁窗

# 验证COLA条件
overlap_ratio = (win_length - hop_length) / win_length
print(f"重叠比例: {overlap_ratio:.1%}")

# 检查COLA兼容性
cola_sum = np.zeros(win_length)
for i in range(0, win_length, hop_length):
    cola_sum += window ** 2
if np.allclose(cola_sum[hop_length:-hop_length], 1.0, atol=0.01):
    print("窗函数满足COLA条件")
else:
    print("警告: 窗函数可能不满足完美重构条件")

# 加载ONNX模型
model_path = "./gtcrn/stream/onnx_models/gtcrn_simple.onnx"
session = onnxruntime.InferenceSession(
    model_path, None, providers=['CPUExecutionProvider'])

# 初始化缓存变量
input_cache = np.zeros(win_length - hop_length, dtype=np.float32)  # 输入缓存
output_cache = np.zeros(win_length - hop_length, dtype=np.float32)  # 输出缓存
conv_cache = np.zeros([2, 1, 16, 16, 33], dtype="float32")
tra_cache = np.zeros([2, 3, 1, 1, 16], dtype="float32")
inter_cache = np.zeros([2, 1, 33, 16], dtype="float32")
audiobyte = []

def denoise_audio(input_audio):
    global conv_cache, tra_cache, inter_cache, input_cache, output_cache
    
    assert len(input_audio) == hop_length, f"输入帧长度应为{hop_length}，实际为{len(input_audio)}"
    
    # 1. 构建当前分析帧（输入缓存 + 新输入）
    analysis_frame = np.concatenate([input_cache, input_audio])
    
    # 2. 应用窗函数
    windowed_frame = analysis_frame * window
    
    # 3. 计算STFT
    stft = librosa.stft(
        windowed_frame,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length, 
        window=np.ones(win_length),  # 已手动加窗
        center=False
    )
    
    stft_input = np.stack([stft.real, stft.imag], axis=-1)[np.newaxis, ...]
    
    # 4. 模型推理（暂时注释）
    output, conv_cache, tra_cache, inter_cache = session.run(
        [], 
        {
            'mix': stft_input.astype(np.float32),
            'conv_cache': conv_cache,
            'tra_cache': tra_cache,
            'inter_cache': inter_cache
        }
    )

    # 测试无模型影响下 stft————>istft音频是否有噪音
    # output = stft_input    

    # 5. ISTFT重构
    reconstructed_frame = librosa.istft(
        output[0, ..., 0] + 1j * output[0, ..., 1],
        hop_length=hop_length,
        win_length=win_length,
        window=np.ones(win_length),  # 已手动加窗
        center=False
    )
    
    # 重叠-相加处理
    # 当前帧的重构信号 = 输出缓存（前帧后半部分） + 当前帧前半部分
    current_output = np.zeros(hop_length, dtype=np.float32)
    
    # 前半部分与输出缓存重叠相加
    overlap_len = len(output_cache)
    if overlap_len > 0:
        current_output[:overlap_len] = output_cache + reconstructed_frame[:overlap_len]
    
    # 后半部分直接取自重构信号
    if len(reconstructed_frame) > overlap_len:
        remaining_len = min(hop_length - overlap_len, len(reconstructed_frame) - overlap_len)
        current_output[overlap_len:overlap_len + remaining_len] = reconstructed_frame[overlap_len:overlap_len + remaining_len]
    
    # 7. 更新缓存
    input_cache = input_audio.copy()  # 更新输入缓存为当前输入
    output_cache = reconstructed_frame[hop_length:hop_length + overlap_len].copy()  # 保存当前帧的重叠部分
    
    return current_output

def audio_callback(indata, outdata, frames, time, status):
    global audiobyte
    if status:
        print(status, file=sys.stderr)

    if len(indata[:, 0]) == hop_length:
        denoised_audio = denoise_audio(indata[:, 0].flatten())
        outdata[:, 0] = denoised_audio
        audiobyte.append(denoised_audio.copy())
    else:
        outdata[:, 0] = indata[:, 0]

if __name__ == "__main__":
    try:
        with sd.Stream(
            samplerate=sr,
            blocksize=hop_length,
            dtype='float32',
            channels=1,
            callback=audio_callback
        ):
            print("正在进行实时音频降噪（按Ctrl+C停止）...")
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n程序已停止")
        if audiobyte:
            sf.write("enhanced_output.wav", np.concatenate(audiobyte), sr)
    except Exception as e:
        print(f"错误: {e}")