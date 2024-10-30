import numpy as np
import os
import json
from scipy import interpolate, io
import pandas as pd
import re
import cv2 as cv
import torch

# def read_video(data_path):
#     """读取视频 T x H x W x C, C = 3"""
#     vid = cv.VideoCapture(data_path + os.sep + "video.avi")
#     vid.set(cv.CAP_PROP_POS_MSEC, 0)  # 设置从 0 开始读取
#     ret, frame = vid.read()
#     frames = list()
#     while ret:
#         frame = cv.cvtColor(np.array(frame), cv.COLOR_BGR2RGB)
#         frame = np.asarray(frame)
#         frame[np.isnan(frame)] = 0
#         frames.append(frame)
#         ret, frame = vid.read()

#     frames = np.asarray(frames)
    
#     f_time = np.loadtxt(data_path + os.sep + "time.txt")
    
#     waves = pd.read_csv(data_path + os.sep + "wave.csv")
#     fun = interpolate.CubicSpline(range(len(waves)), waves)
#     x_new = np.linspace(0, len(waves) - 1, num=len(frames))  # linspace 为闭区间
#     gts = fun(x_new) 
#     print(x_new)
    
#     print(f"视频的帧数是{len(frames)}")
#     print(f"time.txt文件当中记录的总帧数是{len(f_time)}")
#     print(f"time.txt文件当中记录的视频的长度是{f_time[-1]}")
#     print(f"waves的长度是{len(waves)}")
#     print(f"更改之后gts的长度为{len(gts)}")
#     print(f"fps:{len(frames) * 1000 / f_time[-1]:.2f}")

# read_video("/share2/data/zhouwenqing/VIPL-HR/data/p1/v2/source1")

def get_frame_count(video_path):
    # 打开视频文件
    cap = cv.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video file.")
        return

    # 获取视频的帧数
    frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    
    # 释放视频文件
    cap.release()

    return frame_count

raw_data_path = "/share2/data/zhouwenqing/PURE"
for subject in os.listdir(raw_data_path):
    raw_data_path_subject = os.path.join(raw_data_path, subject, subject)
    len1 = len(os.listdir(raw_data_path_subject)) # 包含图片的数量
    len2 = get_frame_count(os.path.join(raw_data_path, subject, "vid.avi")) # 视频的帧数
    with open(os.path.join(raw_data_path, subject, f"{subject}.json"), 'r') as f:
        info = json.load(f)
        len3 = len(info["/Image"]) # json当中包含的图片的数量
    print(len1, len2, len3)
    

