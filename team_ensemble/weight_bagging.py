import numpy as np
import pandas as pd
import os
# 读取不同模型的概率文件

proba_conv2d_withlabelsmoothing= np.loadtxt('stacking/conv2d_withlabelsmoothing.txt')
sub_conv2d_withlabelsmoothing = pd.read_csv('data/submit_template.csv')
sub_conv2d_withlabelsmoothing['behavior_id'] = np.argmax(proba_conv2d_withlabelsmoothing, axis=1)

proba_conv2d_withoutlabelsmoothing= np.loadtxt('stacking/conv2d_withoutlabelsmoothing.txt')
sub_conv2d_withoutlabelsmoothing = pd.read_csv('data/submit_template.csv')
sub_conv2d_withoutlabelsmoothing['behavior_id'] = np.argmax(proba_conv2d_withoutlabelsmoothing, axis=1)

proba_conv1d_lstm = np.loadtxt('stacking/conv1d_lstm.txt')
sub_conv1d_lstm = pd.read_csv('data/submit_template.csv')
sub_conv1d_lstm['behavior_id'] = np.argmax(proba_conv1d_lstm, axis=1)

proba_complexconv1d_1 = np.loadtxt('stacking/complexconv1d_1.txt')
sub_complexconv1d_1  = pd.read_csv('data/submit_template.csv')
sub_complexconv1d_1['behavior_id'] = np.argmax(proba_complexconv1d_1, axis=1)

proba_complexconv1d_2 = np.loadtxt('stacking/complexconv1d_2.txt')
sub_complexconv1d_2  = pd.read_csv('data/submit_template.csv')
sub_complexconv1d_2['behavior_id'] = np.argmax(proba_complexconv1d_2, axis=1)

# 加权bagging
proba_t = (proba_conv2d_withlabelsmoothing + proba_conv2d_withoutlabelsmoothing+ proba_conv1d_lstm + proba_complexconv1d_1 + 2*proba_complexconv1d_2) /6


#结果输出
sub_final  = pd.read_csv('data/submit_template.csv')
sub_final['behavior_id']= np.argmax(proba_t, axis=1)
if not os.path.exists('./submit'):
  os.makedirs('./submit')
sub_final.to_csv('submit/sub_final.csv', index=False)
