# # -*- coding: utf-8 -*-
# """
# Created on Mon Jul 27 16:58:39 2020

# @author: Zzzz
# """

import pandas as pd
import numpy as np
import random
from scipy.signal import resample
from tqdm import tqdm
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.losses import CategoricalCrossentropy
import os

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(2020)


def z_score(x):
    x_ave=np.average(x,axis = 0)
    x_std=np.std(x,axis = 0)
    z=(x-x_ave)/x_std
    return z,x_ave,x_std

def get_dic(df,  main_col, fea_col, agg):
    dic = df.groupby(main_col)[fea_col].agg(agg).to_dict()
    fea_name = '_'.join([main_col, fea_col, agg])
    return fea_name, dic
    
def get_1st_order_xyz_features(df, fea_cols, main_col = 'fragment_id'): 
    df_fea           = pd.DataFrame()
    df_fea[main_col] = df[main_col].unique()
    ## count 特征 ##
    _, dic = get_dic(df, main_col, fea_cols[0], 'count') 
    df_fea['cnt']    = df_fea[main_col].map(dic).values
    
    ## 数值统计特征 ##
    for f in tqdm(fea_cols):
        for agg in ['min','max','mean','std','skew','median']:

            fea_name, dic       = get_dic(df, main_col, f, agg) 
            df_fea[fea_name]    = df_fea[main_col].map(dic).values
            
        df_fea['_'.join([main_col, f, 'gap'])]   = df_fea['_'.join([main_col, f, 'max'])] - df_fea['_'.join([main_col, f, 'min'])]     
    return df_fea



# 根据自己的文件目录进行修改
txdir='/content/drive/My Drive/xwbank2020/'
if os.path.exists(txdir):
  os.chdir(txdir)
df_train = pd.read_csv('data/sensor_train.csv')
df_test = pd.read_csv('data/sensor_test.csv')

df_train['acc'] = (df_train['acc_x'] ** 2 + df_train['acc_y'] ** 2 + df_train['acc_z'] ** 2) ** 0.5
df_train['accg'] = (df_train['acc_xg'] ** 2 + df_train['acc_yg'] ** 2 + df_train['acc_zg'] ** 2) ** 0.5

df_train['accxy'] = (df_train['acc_x'] ** 2 + df_train['acc_y'] ** 2) ** 0.5
df_train['accgxy'] = (df_train['acc_xg'] ** 2 + df_train['acc_yg'] ** 2 ) ** 0.5

df_test['acc'] = (df_test['acc_x'] ** 2 + df_test['acc_y'] ** 2 + df_test['acc_z'] ** 2) ** 0.5
df_test['accg'] = (df_test['acc_xg'] ** 2 + df_test['acc_yg'] ** 2 + df_test['acc_zg'] ** 2) ** 0.5

df_test['accxy'] = (df_test['acc_x'] ** 2 + df_test['acc_y'] ** 2) ** 0.5
df_test['accgxy'] = (df_test['acc_xg'] ** 2 + df_test['acc_yg'] ** 2 ) ** 0.5


origin_fea_cols = ['acc_x','acc_y','acc_z','acc_xg','acc_yg','acc_zg','acc','accg','accxy','accgxy']
df_fea0 = get_1st_order_xyz_features(df_train,origin_fea_cols,main_col='fragment_id')
df_fea1 = get_1st_order_xyz_features(df_test,origin_fea_cols,main_col='fragment_id')

x=np.zeros((7292*60,10))
for i in tqdm(range(7292)):
    temp=df_train[df_train.fragment_id==i][:60]
    x[i*60:i*60+60,:]=resample(temp.drop(['fragment_id','time_point','behavior_id'],axis=1),60,np.array(temp.time_point))[0]
train_data=z_score(x)[0].reshape([7292,60,10])
lables=df_train.groupby('fragment_id')['behavior_id'].min()

x1=np.zeros((7500*60,10))
for i in tqdm(range(7500)):
    temp=df_test[df_test.fragment_id==i][:60]
    x1[i*60:i*60+60,:]=resample(temp.drop(['fragment_id','time_point'],axis=1),60,np.array(temp.time_point))[0]
test_data=((x1-z_score(x)[1])/z_score(x)[2]).reshape([7500,60,10])

df_fea0 = np.array(df_fea0.drop(['fragment_id'],axis=1))
df_fea1 = np.array(df_fea1.drop(['fragment_id'],axis=1))

df_fea0=z_score(df_fea0)[0]
df_fea1=(df_fea1-z_score(df_fea0)[1])/z_score(df_fea0)[2]

def Net():
    input = Input(shape=(60, 10))
    hin = Input(shape=(71, ))
    X = Conv1D(filters= 256,
               kernel_size=5,
               padding='same',
               activation='relu')(input)
    X = BatchNormalization()(X)

    X = Conv1D(filters= 200,
               kernel_size=5,
               padding='same',
               activation='relu')(X)
    X = BatchNormalization()(X)
    X = MaxPooling1D()(X)
    X = Dropout(0.3)(X)

    X = Conv1D(filters= 128,
               kernel_size=5,
               padding='same',
               activation='relu')(X)
    X = BatchNormalization()(X)
    X = MaxPooling1D()(X)
    X = Dropout(0.3)(X)
    
    X = LSTM(128)(X)
    X = Dropout(0.3)(X)
    merge = concatenate([X, hin])
    merge = BatchNormalization()(merge)
    y = Dense(19, activation='softmax')(X)
    return Model(inputs=[input, hin], outputs=y)

y_ = to_categorical(lables, num_classes=19)
proba_t = np.zeros((7500, 19))
val_acc = []
seeds = [44, 2020, 527, 1527]
for seed in seeds:
    kfold = StratifiedKFold(5, shuffle=True, random_state=seed)
    for fold, (xx, yy) in enumerate(kfold.split(train_data, lables)):
      model= Net()
      model.compile(loss=CategoricalCrossentropy(label_smoothing=0.1),
                    optimizer=Adam(1e-3),metrics=['accuracy'])
      plateau = ReduceLROnPlateau(monitor="val_accuracy",
                                  verbose=0,
                                  mode='max',
                                  factor=0.6,
                                  patience=5)
      early_stopping = EarlyStopping(monitor='val_accuracy',
                                    verbose=0,
                                    mode='max',
                                    patience=15)
      if not os.path.exists('./models/conv1D_lstm'):
        os.makedirs('./models/conv1D_lstm')      
      checkpoint = ModelCheckpoint(f'./models/conv1D_lstm/fold{fold}.h5',
                                  monitor='val_accuracy',
                                  verbose=0,
                                  mode='max',
                                  save_best_only=True)

      hist = model.fit([train_data[xx],df_fea0[xx]], y_[xx],
                epochs=150,
                batch_size=256,
                verbose=2,
                shuffle=True,
                validation_data=([train_data[yy],df_fea0[yy]], y_[yy]),
                callbacks=[plateau, early_stopping, checkpoint])
      val_acc.append(np.max(hist.history['val_accuracy']))
      model.load_weights(f'./models/conv1D_lstm/fold{fold}.h5')
      proba_t += model.predict([test_data,df_fea1], verbose=0, batch_size=1024) / kfold.n_splits/ len(seeds)
      
print('val_acc:', np.mean(val_acc))    
if not os.path.exists('./stacking'):
        os.makedirs('./stacking')
np.savetxt('stacking/conv1d_lstm.txt',proba_t)