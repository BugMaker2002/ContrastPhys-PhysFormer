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
from Physformer import ViT_ST_ST_Compact3_TDC_gra_sharp

ex = Experiment('model_pred', save_git_info=False)

@ex.config
def my_config():
    e = 29 # the model checkpoint at epoch e
    train_exp_num = 1 # the training experiment number
    train_exp_dir = './results_physformer/%d'%train_exp_num # training experiment directory
    time_interval = 160 / 30
    T = 160 

    ex.observers.append(FileStorageObserver(train_exp_dir))

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
    
    else:
        device = torch.device('cpu')

@ex.automain
def my_main(_run, e, train_exp_dir, device, time_interval):
    
    T = 160 

    # load test file paths
    test_list = list(np.load(train_exp_dir + '/test_list.npy'))
    pred_exp_dir = train_exp_dir + '/%d'%(int(_run._id)) # prediction experiment directory

    with open(train_exp_dir+'/config.json') as f:
        config_train = json.load(f)

    model = ViT_ST_ST_Compact3_TDC_gra_sharp(S=4, dim=96, ff_dim=144, num_heads=4, num_layers=12, 
                                             image_size=(160, 128, 128), patches=(4, 4, 4), theta=0.7, 
                                             dropout_rate=0.1).to(device).eval()
    
    model.load_state_dict(torch.load("/share1/home/zhouwenqing/rPPG-Toolbox/final_model_release/UBFC-rPPG_PhysFormer_DiffNormalized.pth", map_location=device))
    # model.load_state_dict(torch.load(train_exp_dir+'/epoch%d.pt'%(e), map_location=device)) # load weights to the model

    @torch.no_grad()
    def dl_model(imgs_clip):
        # model inference
        img_batch = imgs_clip
        img_batch = img_batch.transpose((3,0,1,2))
        # 在img_batch前面新增了一个批量大小的维度（批量大小为1）
        img_batch = img_batch[np.newaxis].astype('float32')
        img_batch = torch.tensor(img_batch).to(device)

        rppg = model(img_batch, 2.0)[:,-1, :] # (1, 5, T) -> (1, T)
        rppg = rppg[0].detach().cpu().numpy()
        return rppg

    
    for h5_path in test_list:
        h5_path = str(h5_path)

        with h5py.File(h5_path, 'r') as f:
            imgs = f['imgs']
            subject_name = os.path.basename(h5_path)[:-3]
            bvp_path = f"/share2/data/zhouwenqing/UBFC_rPPG/dataset2/{subject_name}/ground_truth.txt"
            bvp = np.loadtxt(bvp_path).reshape((-1, 1))
            num_blocks = int(np.min([imgs.shape[0], bvp.shape[0]]) // T)
            # 从整个视频当中截取出num_blocks个视频片段，这些片段之间是连续的（指从原视频当中截取的方式）
            rppg_list = []
            bvp_list = []
            # bvppeak_list = []
            for b in range(num_blocks):
                rppg_clip = dl_model(imgs[b*T:(b+1)*T])
                rppg_list.append(rppg_clip)

                bvp_list.append(bvp[b*T:(b+1)*T])

            rppg_list = np.array(rppg_list)
            bvp_list = np.array(bvp_list)

            results = {'rppg_list': rppg_list, 'bvp_list': bvp_list}
            np.save(pred_exp_dir+'/'+h5_path.split('/')[-1][:-3], results)
            