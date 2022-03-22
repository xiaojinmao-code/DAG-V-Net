import os
from tensorflow.python.client import device_lib
from Vnet2d.vnet_model import DSVnet2dModule
from Vnet2d.vnet_model import Vnet2dModule
from Vnet2d.vnet_attention import Dsvnet2d_attentionModule
import numpy as np
import pandas as pd
import cv2
print(device_lib.list_local_devices())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def calcu_dice(Y_pred, Y_gt, K=255):
    """
    calculate two input dice value
    :param Y_pred:
    :param Y_gt:
    :param K:
    :return:
    """
    intersection = 2 * np.sum(Y_pred[Y_gt == K])
    denominator = np.sum(Y_pred) + np.sum(Y_gt)
    loss = (intersection / denominator)
    return loss


def predict_test():
    Vnet2d = Dsvnet2d_attentionModule(512, 768, channels=1, costname="dice coefficient", inference=True,
                                      model_path="C:/Users/spark/Desktop/HC18/log/best/model/AdsVnet2d.pd-229376")

    test_set_csv = pd.read_csv("C:/Users/spark/Desktop/HC18/data/test_set_pixel_size.csv")
    src_test_set = "C:/Users/spark/Desktop/HC18/data/test_set/"
    mask_test_set = "C:/Users/spark/Desktop/HC18/data/predict/"
    imagedata = test_set_csv.iloc[:, 0].values
    for i in range(len(imagedata)):
        src_image = cv2.imread(src_test_set + imagedata[i], cv2.IMREAD_GRAYSCALE)
        resize_image = cv2.resize(src_image, (768, 512))
        mask_image = Vnet2d.prediction(resize_image)
        new_mask_image = cv2.resize(mask_image, (src_image.shape[1], src_image.shape[0]))
        cv2.imwrite(mask_test_set + imagedata[i], new_mask_image)


predict_test()
