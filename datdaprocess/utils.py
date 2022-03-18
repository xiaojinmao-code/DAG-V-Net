import os

paths = ['C:/Users/spark/Desktop/HC18/data/train_image/mask/']
f = open('C:/Users/spark/Desktop/HC18/datdaprocess/train_Y.csv', 'w')

for path in paths:
    p = os.path.abspath(path)
    filenames = os.listdir(p)
    for filename in filenames:
        im_path = p+'/'+filename
        print(im_path)
        f.write(im_path+'\n')
f.close()
