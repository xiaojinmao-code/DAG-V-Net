import os
from tensorflow.python.client import device_lib
from Vnet2d.vnet_model import DSVnet2dModule
from Vnet2d.vnet_model import Vnet2dModule
from Vnet2d.vnet_attention import vnet2d_attentionModule
from Vnet2d.vnet_attention import Dsvnet2d_attentionModule
import numpy as np
import pandas as pd
print(device_lib.list_local_devices())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def train():
    """
    Preprocessing for dataset
    """
    # Read  data set (Train data from CSV file)
    csvmaskdata = pd.read_csv('C:/Users/spark/Desktop/HC18/datdaprocess/train_mask_aug.csv')
    csvimagedata = pd.read_csv('C:/Users/spark/Desktop/HC18/datdaprocess/train_src_aug.csv')
    maskdata = csvmaskdata.iloc[:, :].values
    imagedata = csvimagedata.iloc[:, :].values
    # shuffle imagedata and maskdata together
    perm = np.arange(len(csvimagedata))
    np.random.shuffle(perm)
    imagedata = imagedata[perm]
    maskdata = maskdata[perm]

    unet2d = Dsvnet2d_attentionModule(512, 768, channels=1, costname="dice coefficient")
    unet2d.train(imagedata, maskdata, "AdsVnet2d.pd", "log\\test\\", 0.0005, 0.5, 25, 2)


train()
