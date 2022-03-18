from datdaprocess.Augmentation.ImageAugmentation import DataAug

aug = DataAug(rotation=20, width_shift=0.01, height_shift=0.01, rescale=1.1)
aug.DataAugmentation('Train_X.csv', 'Train_Y.csv', 30,
                     path="C:/Users/spark/Desktop/HC18/datdaprocess/data/Aug/")
