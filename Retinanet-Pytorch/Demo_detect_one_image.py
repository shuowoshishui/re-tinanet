# -*- coding: utf-8 -*-
# @Author  : LG
from Model import RetainNet
from Configs import _C as cfg
from PIL import Image
import matplotlib.pyplot as plt
# 实例化模型
net = RetainNet(cfg)
# 使用cpu或gpu
net.to('cuda')
# 模型从权重文件中加载权重
net.load_pretrained_weight('/home/chenshuo/Retinanet-Pytorch/Weights/trained1/model_20000.pkl')
# 打开图片
image = Image.open("/home/chenshuo/RTTS/JPEGImages/BD_Baidu_145.png")
# 进行检测, 分别返回 绘制了检测框的图片数据/回归框/标签/分数.
drawn_image, boxes, labels, scores = net.Detect_single_img(image=image,score_threshold=0.5)

plt.imsave('XXX_det.png',drawn_image)
plt.imshow(drawn_image)
plt.show()