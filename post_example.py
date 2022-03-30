import cv2 as cv
import numpy as np
import os

def canny_demo(image):
    t = 80
    canny_output = cv.Canny(image, t, t * 2)
    return canny_output

# 设置图片路径
DATADIR = "C:/HC18/test_set/98.17/"
DATADIR1 = "C:/HC18/test_set/test_set//"


# 使用os.path模块的join方法生成路径
path = os.path.join(DATADIR)
path1 = os.path.join(DATADIR1)

# 使用os.listdir(path)函数，返回path路径下所有文件的名字，以及文件夹的名字
img_list = os.listdir(path)
img_list1 = os.listdir(path1)

ind = 0

for i in img_list:

    src = cv.imread(os.path.join(path,i))
    src1 = cv.imread(os.path.join(path1,i))

    # 矩形，3*3，中心点
    se = cv.getStructuringElement(cv.MORPH_RECT, (3, 3), (-1, -1))
    # 开运算
    src = cv.morphologyEx(src, cv.MORPH_OPEN, se)
    # canny算子，80,160
    binary = cv.Canny(src, 80, 80 * 2)
    # 3*3全为1的掩膜
    k = np.ones((3,3), dtype=np.uint8)
    # 膨胀
    binary = cv.morphologyEx(binary, cv.MORPH_DILATE, k)
    # 轮廓发现:(图像,提取最外层轮廓,轮廓的近似办法)
    contours, hierarchy = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for c in range(len(contours)):

        # 轮廓描绘
        rect = cv.minAreaRect(contours[c])
        cx2, cy2 = rect[0]
        box = cv.boxPoints(rect)
        box = np.int0(box)
        # cv.drawContours(src1, [box], 0, (0, 255, 0), 2)

        #cv.circle(src1, (np.int32(cx2), np.int32(cy2)), 5, (255, 0, 0), 3, 8, 0)
        #cv.drawContours(src1, contours, c, (0, 255, 0), 2, 8)

        # 椭圆拟合
        (cx, cy), (b, a), angle = cv.fitEllipse(contours[c])
        # 绘制椭圆
        cv.ellipse(src1, (np.int32(cx), np.int32(cy)),
                       (np.int32((b+4) / 2), np.int32((a+4) / 2)), angle, 0, 360, (0, 0, 255), 3, 8, 0)

        cv.ellipse(src1, (np.int32(cx), np.int32(cy)),
                   (np.int32((b-4) / 2), np.int32((a-4) / 2)), angle, 0, 360, (255, 0, 0), 3, 8, 0)
        # 计算
        predalc = 2 * (a - b) + np.pi * b
        predarea = np.pi * (a/2) * (b/2)
        cv.putText(src1, "predalc:" + str(predalc), (30, 40), cv.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 1)
        cv.putText(src1, "predarea:" + str(predarea), (30, 80), cv.FONT_HERSHEY_SIMPLEX, .6, (0, 0, 255), 1)
        print(i,cx,cy,(a)/2,(b)/2,angle)
        print(i,predalc)
        # 显示
        cv.imwrite('C:/HC18/test_set/dsvnet/'+str(i)+'.png', src1)

        cv.waitKey(0)
        cv.destroyAllWindows()



