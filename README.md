## DAG V-Net
The code will be constantly updated(!!!2020-11-16!!!)

This repository provides the code for "Fetal ultrasound image segmentation for automatic head circumference biometry using deeply-supervised attention-gated V-Net". Our work now is Accepted by the Journal of Digital Imaging.

![img_net](./pictures/model.jpg)
Fig. 1. Structure of DAG V-Net.



### Requirementss
Some important required packages include:
* tensorflow version >=1.12.0.
* opencv-python >=3.3.0
* pandas >=0.20.1
* python >= 3.6 
* Some basic python packages such as SimpleITK.



## Usages
### Data
First, you can download the dataset at [HC-18][data_link]. 
* Download the HC-18 training set that consists of 999 2D ultrasound images and their annotations. 
* Download the HC-18 test set that consists of 335 2D ultrasound images.  

![img_src](./pictures/HC18.png)
Fig. 2. The official data sample.

[data_link]:https://hc18.grand-challenge.org/


### Preprocessing
* The annotation of this dataset are contours. We need to convert them into binary masks for segmentation.
```
python get_ground_truth.py
```

* data augmentation.
```
python augtest.py
```
### Train
To train DAG-Net in HC-18.
```
python main.py
```

### Test
To evaluate the trained model in ISIC 2018 (we added a test data in folder0, testing the 0th-fold validation for example), run:
```
python validation.py
```

## Result
![uncertainty](./pictures/comparison.jpg)
Fig. 2. Fetal head segmentation.


Our experimental results are shown in the table:
![refinement](./pictures/skin_segmentation_results_table.png)

4. You can save the attention weight map in the middle step of the network to '/result' folder. Visualizing the attention weight above the original images, run:
```
python show_fused_heatmap.py
```

## Acknowledgement
Part of the code is revised from [Attention-Gate-Networks][AG].

[AG]:https://github.com/ozan-oktay/Attention-Gated-Networks

## Contact
* email:799633204@qq.com
* wechat:18752726918
* QQ:799633204
* [CSDN][web_link].


[web_link]:https://hc18.grand-challenge.org/









