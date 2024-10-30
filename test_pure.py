import numpy as np
import h5py
from torch import nn
import torch
from PhysNetModel import PhysNet
from utils_data import *
from utils_sig import *
from sacred import Experiment
from sacred.observers import FileStorageObserver
import json
from scipy.stats import pearsonr
from UniFormer import Uniformer
from scipy import interpolate
import json

ex = Experiment('model_pred', save_git_info=False)

@ex.config
def my_config():
    e = 29 # the model checkpoint at epoch e
    train_exp_num = 2 # the training experiment number
    train_exp_dir = './results_cnn_pure/%d'%train_exp_num # training experiment directory
    
    # 这里为了适应transformer架构，我将时间从30s改成了10s
    time_interval = 30 # get rppg for 30s video clips, too long clips might cause out of memory

    ex.observers.append(FileStorageObserver(train_exp_dir))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    else:
        device = torch.device('cpu')

@ex.automain
def my_main(_run, e, train_exp_dir, device, time_interval):

    # load test file paths
    test_list = list(np.load(train_exp_dir + '/test_list.npy'))
    pred_exp_dir = train_exp_dir + '/%d'%(int(_run._id)) # prediction experiment directory

    with open(train_exp_dir+'/config.json') as f:
        config_train = json.load(f)

    model = PhysNet(config_train['S'], config_train['in_ch']).to(device).eval()
    
    # model = Uniformer(
    # num_classes = 1000,                 # number of output classes
    # dims = (64, 64, 64, 64),         # feature dimensions per stage (4 stages)
    # depths = (2, 2, 2, 2),           # depth at each stage
    # mhsa_types = ('l', 'l', 'g', 'g')   # aggregation type at each stage, 'l' stands for local, 'g' stands for global
    # ).to(device).eval()
    
    
    model.load_state_dict(torch.load(train_exp_dir+'/epoch%d.pt'%(e), map_location=device)) # load weights to the model
    # model.load_state_dict(torch.load("./demo/model_weights.pt", map_location=device))

    @torch.no_grad()
    def dl_model(imgs_clip):
        # model inference
        img_batch = imgs_clip
        img_batch = img_batch.transpose((3,0,1,2))
        # 在img_batch前面新增了一个批量大小的维度（批量大小为1）
        img_batch = img_batch[np.newaxis].astype('float32')
        img_batch = torch.tensor(img_batch).to(device)

        rppg = model(img_batch)[:,-1, :] # (1, 5, T) -> (1, T)
        rppg = rppg[0].detach().cpu().numpy()
        return rppg
    
    @torch.no_grad()
    def read_wave(info, target_length):
        bvp = []
        hr = []
        for signal in info["/FullPackage"]:
            bvp.append(signal["Value"]["waveform"])
            hr.append(signal["Value"]["pulseRate"])
        
        T = len(bvp)
        gts = list()
        x_new = np.linspace(0, T - 1, num=target_length)  
        bvp_func = interpolate.CubicSpline(range(T), bvp)
        gts.append(bvp_func(x_new))
        hr_func = interpolate.CubicSpline(range(T), hr)
        gts.append(hr_func(x_new))

        return np.asarray(gts)

    
    for h5_path in test_list:
        h5_path = str(h5_path)
        # print(h5_path)
        with h5py.File(h5_path, 'r') as f:
            imgs = f['imgs']
            subject_name = os.path.basename(h5_path)[:-3]
            json_path = f"/share2/data/zhouwenqing/PURE/{subject_name}/{subject_name}.json"
            with open(json_path) as f:
                info = json.load(f)
                gts = read_wave(info, imgs.shape[0])
                # print(gts.shape)
            bvp = gts[0].reshape(-1, 1)
            # bvppeak = f['bvp_peak']
            fs = config_train['fs']
            
            print(subject_name, imgs.shape, bvp.shape)
            
            # duration表示秒数，fs表示frame per seccond 
            duration = np.min([imgs.shape[0], bvp.shape[0]]) / fs
            num_blocks = int(duration // time_interval)
            # 从整个视频当中截取出num_blocks个视频片段，这些片段之间是连续的（指从原视频当中截取的方式）
            rppg_list = []
            bvp_list = []
            # bvppeak_list = []
            for b in range(num_blocks):
                rppg_clip = dl_model(imgs[b*time_interval*fs:(b+1)*time_interval*fs])
                rppg_list.append(rppg_clip)

                bvp_list.append(bvp[b*time_interval*fs:(b+1)*time_interval*fs])
                # bvppeak_list.append(bvppeak[b*time_interval*fs:(b+1)*time_interval*fs])

            rppg_list = np.array(rppg_list)
            bvp_list = np.array(bvp_list)
            bvp_list = bvp_list.reshape(num_blocks, -1)
            print(num_blocks, rppg_list.shape, bvp_list.shape)
            # bvppeak_list = np.array(bvppeak_list)
            # results = {'rppg_list': rppg_list, 'bvp_list': bvp_list, 'bvppeak_list':bvppeak_list}
            results = {'rppg_list': rppg_list, 'bvp_list': bvp_list}
            np.save(pred_exp_dir+'/'+h5_path.split('/')[-1][:-3], results)
            