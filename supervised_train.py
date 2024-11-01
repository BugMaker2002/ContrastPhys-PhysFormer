import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import h5py
import torch
from PhysNetModel import PhysNet
from loss import ContrastLoss
from IrrelevantPowerRatio import IrrelevantPowerRatio
from utils_data import *
from utils_sig import *
from torch import optim
from torch.utils.data import DataLoader
from sacred import Experiment
from sacred.observers import FileStorageObserver
from torch import nn
from torch.cuda.amp import autocast, GradScaler
import random
from UniFormer import Uniformer

ex = Experiment('model_train', save_git_info=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
else:
    device = torch.device('cpu')

@ex.config
def my_config():
    # here are some hyperparameters in our method

    # hyperparams for model training
    total_epoch = 30 # total number of epochs for training the model
    lr = 1e-5 # learning rate
    in_ch = 3 # TODO: number of input video channels, in_ch=3 for RGB videos, in_ch=1 for NIR videos.

    # hyperparams for ST-rPPG block
    fs = 30 # video frame rate, TODO: modify it if your video frame rate is not 30 fps.
    T = fs * 10 # temporal dimension of ST-rPPG block, default is 10 seconds.
    S = 2 # spatial dimenion of ST-rPPG block, default is 2x2.

    # hyperparams for rPPG spatiotemporal sampling
    delta_t = int(T/2) # time length of each rPPG sample
    K = 4 # the number of rPPG samples at each spatial position

    result_dir = './results_supervised' # store checkpoints and training recording
    ex.observers.append(FileStorageObserver(result_dir))

@ex.automain
def my_main(_run, total_epoch, T, S, lr, result_dir, fs, delta_t, K, in_ch):

    exp_dir = result_dir + '/%d'%(int(_run._id)) # store experiment recording to the path

    # get the training and test file path list by spliting the dataset
    train_list, test_list = UBFC_LU_split() # TODO: you should define your function to split your dataset for training and testing
    # 将数组 train_list 保存为 NumPy 的 .npy 文件格式
    np.save(exp_dir+'/train_list.npy', train_list)
    np.save(exp_dir+'/test_list.npy', test_list)

    # define the dataloader
    dataset = H5Dataset(train_list, T) # please read the code about H5Dataset when preparing your dataset
    dataloader = DataLoader(dataset, batch_size=2, # two videos for contrastive learning
                            shuffle=True, num_workers=4, pin_memory=True, drop_last=True) # TODO: If you run the code on Windows, please remove num_workers=4.
    
    # define the model and loss
    
    # model = PhysNet(S, in_ch=in_ch).to(device).train()
    model = Uniformer(
        num_classes = 1000,                 # number of output classes
        dims = (64, 64, 64, 64),
        depths = (2, 2, 2, 2),
        mhsa_types = ('l', 'l', 'g', 'g')   # aggregation type at each stage, 'l' stands for local, 'g' stands for global
    ).to(device).train()
    
    loss_func = ContrastLoss(delta_t, K, fs, high_pass=40, low_pass=250)

    # define irrelevant power ratio
    IPR = IrrelevantPowerRatio(Fs=fs, high_pass=40, low_pass=250)

    # define the optimizer
    opt = optim.AdamW(model.parameters(), lr=lr)

    for e in range(total_epoch):
        for it in range(np.round(60/(T/fs)).astype('int')): # TODO: 60 means the video length of each video is 60s. If each video's length in your dataset is other value (e.g, 30s), you should use that value.
            for imgs in dataloader: # dataloader randomly samples a video clip with length T
                imgs = imgs.to(device)
                # model forward propagation
                model_output = model(imgs) # torch.Size([2, 5, 300])
                # print("model_output.shape: ", model_output.shape)
                rppg = model_output[:,-1] # get rppg # torch.Size([2, 300])
                # print("rppg.shape:", rppg.shape)
                
                # define the loss functions
                loss, p_loss, n_loss = loss_func(model_output)

                # optimize
                opt.zero_grad()
                loss.backward()
                opt.step()

                # evaluate irrelevant power ratio during training
                ipr = torch.mean(IPR(rppg.clone().detach()))

                # save loss values and IPR
                ex.log_scalar("loss", loss.item())
                ex.log_scalar("p_loss", p_loss.item())
                ex.log_scalar("n_loss", n_loss.item())
                ex.log_scalar("ipr", ipr.item())

        # save model checkpoints
        torch.save(model.state_dict(), exp_dir+'/epoch%d.pt'%e)
