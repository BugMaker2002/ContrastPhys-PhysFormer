import numpy as np
import os
import h5py
from torch.utils.data import Dataset
from scipy.fft import fft
from scipy import signal
from scipy.signal import butter, filtfilt

def UBFC_LU_split():
    # split UBFC dataset into training and testing parts
    # the function returns the file paths for the training set and test set.
    # TODO: if you want to train on another dataset, you should define new train-test split function.
    
    h5_dir = ''
    train_list = []
    val_list = []

    val_subject = [49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38]

    for subject in range(1,50):
        if os.path.isfile(h5_dir+'/subject%d.h5'%(subject)):
            if subject in val_subject:
                val_list.append(h5_dir+'/subject%d.h5'%(subject))
            else:
                train_list.append(h5_dir+'/subject%d.h5'%(subject))

    return train_list, val_list    

def PURE_LU_split():
    # split UBFC dataset into training and testing parts
    # the function returns the file paths for the training set and test set.
    # TODO: if you want to train on another dataset, you should define new train-test split function.
    
    h5_dir = '/share2/data/zhouwenqing/PURE_h5'
    train_list = []
    val_list = []

    val_subject = ["10-01.h5", "10-02.h5", "10-03.h5", "10-04.h5", "10-05.h5", "10-06.h5",
                   "09-01.h5", "09-02.h5", "09-03.h5", "09-04.h5", "09-05.h5", "09-06.h5",
                   "08-01.h5", "08-02.h5", "08-03.h5", "08-04.h5", "08-05.h5", "08-06.h5",]
    for file in os.listdir(h5_dir):
        if file in val_subject:
            val_list.append(os.path.join(h5_dir, file))
        else:
            train_list.append(os.path.join(h5_dir, file))
        

    return train_list, val_list    
    
class H5Dataset(Dataset):

    def __init__(self, train_list, T):
        self.train_list = train_list # list of .h5 file paths for training
        self.T = T # video clip length

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, idx):
        with h5py.File(self.train_list[idx], 'r') as f:
            img_length = f['imgs'].shape[0]

            idx_start = np.random.choice(img_length-self.T)

            idx_end = idx_start+self.T

            img_seq = f['imgs'][idx_start:idx_end]
            img_seq = np.transpose(img_seq, (3, 0, 1, 2)).astype('float32')
        return img_seq
