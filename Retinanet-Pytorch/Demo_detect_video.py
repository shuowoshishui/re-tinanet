# -*- coding: utf-8 -*-
# @Author  : LG
from Model import RetainNet
from Configs import _C as cfg

# 实例化模型
net = RetainNet(cfg)
# 使用cpu或gpu
net.to('cuda')
# 模型从权重文件中加载权重
net.load_pretrained_weight(cfg.FILE.MODEL_SAVE_ROOT+'/'+'model_8000.pkl')

video_path = '/home/chenshuo/Retinanet-Pytorch/test.mp4'

# 进行检测,
# if save_video_path不为None,则不保存视频,如需保存视频save_video_path=XXX.mp4 ,
# show=True,实时显示检测结果
net.Detect_video(video_path=video_path, score_threshold=0.02, save_video_path='/home/chenshuo/Retinanet-Pytorch/test1.mp4', show=False)
