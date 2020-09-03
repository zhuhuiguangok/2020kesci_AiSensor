import tensorflow as tf
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold
import os
import numpy as np
import datetime
from tensorflow.keras.layers import Conv2D,Conv1D,Dense,Dropout,Input,GlobalMaxPooling2D,MaxPooling2D,MaxPooling1D,LayerNormalization,BatchNormalization,LSTM
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import regularizers
from sklearn.preprocessing import LabelEncoder
from datetime import *
import random
np.random.seed(2020)
random.seed(2020)

#半监督1
semi_test=pd.read_csv("data/semi_test_8_nobest_08-04-11-49_0.89783.csv")
semi_test.sort_values(by=["proba"],ascending=False,inplace=True)
semi_test_len=len(semi_test)
print(semi_test_len)

#设置加载预打标样本的比例
get_percentce=0.6
semi_test_percent=semi_test.iloc[:round(get_percentce*semi_test_len)]
sample_weight_train=semi_test_percent.proba.values


test = pd.read_csv('data/sensor_test.csv')
test=test.merge(semi_test_percent,how="inner")
test.fragment_id = test.fragment_id+7500
test.drop("proba",axis=1,inplace=True)

train = pd.read_csv('data/sensor_train.csv')
train=pd.concat([train,test])
train_fragment_id=train.fragment_id.unique()

test = pd.read_csv('data/sensor_test.csv')
sub = pd.read_csv('data/submit_template.csv')
mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
        4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5', 
        8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6', 
        12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6', 
        16: 'C_2', 17: 'C_5', 18: 'C_6'}
mapping3 = {'A':0, 'B':1, 'C':2, 'D':2}

### 处理标签数据
train['action_type'] = train['behavior_id'].map(mapping)
train['action_type_1'] = train['action_type'].map(lambda x:x.split('_')[0])
train['action_type_2'] = train['action_type'].map(lambda x:int(x.split('_')[1]))
train['sence'] = train['behavior_id'].map(lambda x:mapping3[mapping[x].split('_')[0]])
train['action'] = train['behavior_id'].map(lambda x:int(mapping[x].split('_')[1]))
y_cls = train.groupby('fragment_id')['behavior_id'].min()
y_scene = train.groupby('fragment_id')['sence'].min()
y_action = train.groupby('fragment_id')['action'].min()

train['accmod'] = (train.acc_x ** 2 + train.acc_y ** 2 + train.acc_z ** 2) ** .5
train['accmodg'] = (train.acc_xg ** 2 + train.acc_yg ** 2 + train.acc_zg ** 2) ** .5

test['accmod'] = (test.acc_x ** 2 + test.acc_y ** 2 + test.acc_z ** 2) ** .5
test['accmodg'] = (test.acc_xg ** 2 + test.acc_yg ** 2 + test.acc_zg ** 2) ** .5

x = np.zeros((len(train_fragment_id), 60, 8))  
t = np.zeros((7500, 60, 8))
for i in tqdm(range(len(train_fragment_id))):
    tmp = train[train.fragment_id ==train_fragment_id[i]][:60]
    x[i,:,:] = resample(tmp.drop(['fragment_id', 'time_point', 'behavior_id','action_type','action_type_1','action_type_2',"sence","action"],axis=1), 60, np.array(tmp.time_point))[0]
for i in tqdm(range(7500)):
    tmp = test[test.fragment_id == i][:60]
    t[i,:,:] = resample(tmp.drop(['fragment_id', 'time_point'],axis=1), 60, np.array(tmp.time_point))[0]

def standardization(X):
    #X batch_size 60 8
    b,h,w=X.shape[0],X.shape[1],X.shape[2]
    x1 = X.reshape(-1, X.shape[-1])
    mu = np.mean(x1, axis=0)
    sigma = np.std(x1, axis=0, ddof=1)
    x1 = ((x1 - mu) / (sigma))
    print(x1.shape)
    X_=x1.reshape(b,h,w)
    return X_

#标准化
x=standardization(x)
t=standardization(t)

class Disout(tf.keras.layers.Layer):
    '''
    disout
    论文：https://arxiv.org/abs/2002.11022
    '''
    def __init__(self, dist_prob, block_size=5, alpha=1, **kwargs):
        super(Disout, self).__init__(**kwargs)
        self.dist_prob = dist_prob
        self.weight_behind=None

        self.alpha = alpha
        self.block_size = block_size

    def build(self, input_shape):
        pass

    @tf.function
    def call(self, x):
        '''x：(batch_size,h,w,c)'''
        if not self.trainable:
            return x
        else:
            if tf.math.equal(tf.rank(x),4):
                x_shape = tf.shape(x)
                x_size = x_shape[1:3]
                x_size_f = tf.cast(x_size, tf.float32)
                # 计算block_size
                x_block_size_f = tf.constant((self.block_size, self.block_size), tf.float32)
                # x_block_size_f = x_size_f * self.block_size
                # x_block_size_f = tf.math.maximum(x_block_size_f, 1)
                x_block_size = tf.cast(x_block_size_f, tf.int32)
                # 根据dist_prob，计算block_num
                x_block_num = (x_size_f[0] * x_size_f[1]) * self.dist_prob / (x_block_size_f[0] * x_block_size_f[1])
                # 计算block在中心区域出现的概率
                x_block_rate = x_block_num / ((x_size_f[0] - x_block_size_f[0] + 1) * (x_size_f[1] - x_block_size_f[1] + 1))
                # tf.print('x_block_rate:', x_block_rate)
                # 根据概率生成block区域
                x_block_center = tf.random.uniform((x_shape[0], x_size[0] - x_block_size[0] + 1, x_size[1] - x_block_size[1] + 1, x_shape[3]), dtype=tf.float32)
                x_block_padding_t = x_block_size[0] // 2
                x_block_padding_b = x_size_f[0] - tf.cast(x_block_padding_t, tf.float32) - (x_size_f[0] - x_block_size_f[0] + 1.0)
                x_block_padding_b = tf.cast(x_block_padding_b, tf.int32)
                x_block_padding_l = x_block_size[1] // 2
                x_block_padding_r = x_size_f[1] - tf.cast(x_block_padding_l, tf.float32) - (x_size_f[1] - x_block_size_f[1] + 1.0)
                x_block_padding_r = tf.cast(x_block_padding_r, tf.int32)
                x_block_padding = tf.pad(x_block_center,[[0, 0],[x_block_padding_t, x_block_padding_b],[x_block_padding_l, x_block_padding_r],[0, 0]])
                x_block = tf.cast(x_block_padding<x_block_rate, tf.float32)
                x_block = tf.nn.max_pool2d(x_block, ksize=[self.block_size, self.block_size], strides=[1, 1], padding='SAME')
                # block百分比
                # x_block_percent_ones = tf.reduce_sum(x_block) / tf.reduce_prod(tf.cast(tf.shape(x_block), tf.float32))
                # tf.print('x_block_percent_ones:', x_block_percent_ones, tf.shape(x_block))
                # 特征叠加
                x_abs = tf.abs(x)
                x_sum = tf.math.reduce_sum(x_abs, axis=-1, keepdims=True)
                x_max = tf.math.reduce_max(x_sum, axis=(1, 2), keepdims=True)
                x_max_c = tf.math.reduce_max(x_abs, axis=(1, 2), keepdims=True)
                x_sum_c = tf.math.reduce_sum(x_max_c, axis=-1, keepdims=True)
                x_v = x_sum / x_sum_c
                # tf.print('x_v:', tf.shape(x_v), tf.math.reduce_min(x_v), tf.math.reduce_max(x_v))
                # 特征方差
                # x_variance = tf.math.reduce_variance(x_sum, axis=(1, 2), keepdims=True)
                # tf.print('x_variance:', tf.shape(x_variance), tf.math.reduce_min(x_variance), tf.math.reduce_max(x_variance))
                # 叠加扰动
                x_max = tf.reduce_max(x, axis=(1,2), keepdims=True)
                x_min = tf.reduce_min(x, axis=(1,2), keepdims=True)
                x_block_random = tf.random.uniform(x_shape, dtype=x.dtype) * (x_max - x_min) + x_min
                x_block_random = x_block_random * (self.alpha * x_v + 0.3) + x * (1.0 - self.alpha * x_v - 0.3)
                x = x * (1-x_block) + x_block_random * x_block
                return x
            else:
                return x

    def compute_output_shape(self, input_shape):
        '''计算输出shape'''
        return input_shape
        
class Disout1D(tf.keras.layers.Layer):
    '''
    disout
    论文：https://arxiv.org/abs/2002.11022
    '''

    def __init__(self, dist_prob, block_size=5, alpha=0.5, **kwargs):
        super(Disout1D, self).__init__(**kwargs)
        self.dist_prob = dist_prob

        self.alpha = alpha
        self.block_size = block_size

    def build(self, input_shape):
        pass

    @tf.function
    def call(self, x):
        '''x：(batch_size,h,w,c)'''
        if not self.trainable:
            return x
        else:
            if tf.math.equal(tf.rank(x),2):
                x_shape = tf.shape(x)
                x_size = x_shape[1]
                x_size_f = tf.cast(x_size, tf.float32)
                # 计算block_size
                x_block_size_f = tf.constant(self.block_size, tf.float32)
                # x_block_size_f = x_size_f * self.block_size
                # x_block_size_f = tf.math.maximum(x_block_size_f, 1)
                x_block_size = tf.cast(x_block_size_f, tf.int32)
                # 根据dist_prob，计算block_num
                x_block_num = (x_size_f) * self.dist_prob / (x_block_size_f)
                # 计算block在中心区域出现的概率
                x_block_rate = x_block_num / ((x_size_f - x_block_size_f + 1))
                # 根据概率生成block区域
                x_block_center = tf.random.uniform((x_shape[0], x_size - x_block_size + 1), dtype=tf.float32)
                x_block_padding_t = x_block_size // 2
                x_block_padding_b = x_size_f - tf.cast(x_block_padding_t, tf.float32) - (x_size_f - x_block_size_f + 1.0)
                x_block_padding_b = tf.cast(x_block_padding_b, tf.int32)
                x_block_padding = tf.pad(x_block_center,[[0, 0],[x_block_padding_t, x_block_padding_b]])
                x_block = tf.cast(x_block_padding<x_block_rate, tf.float32)
                x_block = tf.expand_dims(x_block, axis=-1)
                x_block = tf.nn.max_pool1d(x_block, ksize=[self.block_size], strides=[1], padding='SAME')
                x_block = tf.reshape(x_block, x_shape)
                # 叠加扰动
                x_max = tf.reduce_max(x, axis=1, keepdims=True)
                x_min = tf.reduce_min(x, axis=1, keepdims=True)
                x_block_random = tf.random.uniform(x_shape, dtype=x.dtype) * (x_max - x_min) + x_min
                x_block_random = x_block_random * (1.0 - self.alpha) + x * self.alpha
                x = x * (1-x_block) + x_block_random * x_block
                return x
            else:
                return x

    def compute_output_shape(self, input_shape):
        '''计算输出shape'''
        return input_shape

#根据赛题自定义的评价函数
def get_acc_combo():
    def combo(y, y_pred):
        # 数值ID与行为编码的对应关系
        mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
            4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5',
            8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
            12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
            16: 'C_2', 17: 'C_5', 18: 'C_6'}
        # 将行为ID转为编码
        code_y, code_y_pred = mapping[int(y)], mapping[int(y_pred)]
        if code_y == code_y_pred: #编码完全相同得分1.0
            return 1.0
        elif code_y.split("_")[0] == code_y_pred.split("_")[0]: #编码仅字母部分相同得分1.0/7
            return 1.0/7
        elif code_y.split("_")[1] == code_y_pred.split("_")[1]: #编码仅数字部分相同得分1.0/3
            return 1.0/3
        else:
            return 0.0

    confusionMatrix=np.zeros((19,19))
    for i in range(19):
        for j in range(19):
            confusionMatrix[i,j]=combo(i,j)
    confusionMatrix=tf.convert_to_tensor(confusionMatrix)

    def acc_combo(y, y_pred):
        y=tf.argmax(y,axis=1)
        y_pred = tf.argmax(y_pred, axis=1)
        indices=tf.stack([y,y_pred],axis=1)#在1轴增加一个维度
        scores=tf.gather_nd(confusionMatrix,tf.cast(indices,tf.int32))
        return tf.reduce_mean(scores)
    return acc_combo


#index 打乱
def shuffle_index_fun(index):
    random.shuffle(index)
    return index.copy()

#数据集打
def shuffle_data(data_x, data_y):
    shuffle_index = list(range(data_x.shape[0]))
    r_index = shuffle_index_fun(shuffle_index)
    return data_x[r_index], data_y[r_index]

#mixup beta分布
def mixup(batch_x, batch_y, random_max=0.2):
    shuffle_index = list(range(batch_x.shape[0]))
    r_index1 = shuffle_index_fun(shuffle_index)
    r_index2 = shuffle_index_fun(shuffle_index) 
    rd =np.random.beta(random_max, random_max)
    batch_x_mixup = (1-rd) * batch_x[r_index1] + rd * batch_x[r_index2]
    batch_y_mixup = (1-rd) * batch_y[r_index1] + rd * batch_y[r_index2]
    return batch_x_mixup, batch_y_mixup

#mixup uniform分布
def mixup_uniform(batch_x, batch_y, random_max=0.06):
    shuffle_index = list(range(batch_x.shape[0]))
    r_index1 = shuffle_index_fun(shuffle_index)
    r_index2 = shuffle_index_fun(shuffle_index) 
    rd =np.random.uniform(0.02, random_max)
    batch_x_mixup = (1-rd) * batch_x[r_index1] + rd * batch_x[r_index2]
    batch_y_mixup = (1-rd) * batch_y[r_index1] + rd * batch_y[r_index2]
    return batch_x_mixup, batch_y_mixup

#噪声增强
def noise(batch_x, batch_y, random_max=0.02):
    size = batch_x.shape
    batch_x_noise = batch_x + np.random.uniform(-random_max,random_max,size=size)
    batch_y_noise = batch_y
    return batch_x_noise, batch_y_noise

#信噪比噪声增强
def jitter(x,y,snr_db):
    """
    根据信噪比添加噪声
    :param x:
    :param snr_db:
    :return:
    """
    # 随机选择信噪比
    assert isinstance(snr_db, list)
    snr_db_low = snr_db[0]
    snr_db_up = snr_db[1]
    snr_db = np.random.randint(snr_db_low, snr_db_up, (1,))[0]

    snr = 10 ** (snr_db / 10)
    Xp = np.sum(x ** 2, axis=0, keepdims=True) / x.shape[0]  # 计算信号功率
    Np = Xp / snr  # 计算噪声功率
    n = np.random.normal(size=x.shape, scale=np.sqrt(Np), loc=0.0)  # 计算噪声 loc均值，scale方差
    xn = x + n
    yn=y
    return xn,yn

#整体数据增强合成
def data_generator(data_x, data_y,dataAugment=False, shuffle=False):

  # 先shuffle打乱
  if shuffle == True:
      data_x, data_y = shuffle_data(data_x, data_y)

  list_batch_x, list_batch_y = [], []
  list_batch_x.append(data_x), list_batch_y.append(data_y)

  # 数据增强 
  if dataAugment == True:

    # 数据增强 mixup beta
    batch_x_mixup, batch_y_mixup = mixup(data_x, data_y, random_max=0.2)
    list_batch_x.append(batch_x_mixup), list_batch_y.append(batch_y_mixup)
    print(batch_x_mixup.shape,batch_y_mixup.shape)

    # 数据增强 mixup uniform
    batch_x_mixup_uniform, batch_y_mixup_uniform = mixup_uniform(data_x, data_y, random_max=0.06)
    list_batch_x.append(batch_x_mixup_uniform), list_batch_y.append(batch_y_mixup_uniform)
    print(batch_x_mixup_uniform.shape,batch_y_mixup_uniform.shape)

    # 数据增强 随机正态噪声
    batch_x_noise, batch_y_noise = noise(data_x, data_y, random_max=0.1)
    list_batch_x.append(batch_x_noise), list_batch_y.append(batch_y_noise)
    print(batch_x_noise.shape,batch_y_noise.shape)
    # 数据增强 功率噪声
    batch_x_jitter, batch_y_jitter=jitter(data_x,data_y,[5,15])
    list_batch_x.append(batch_x_jitter), list_batch_y.append(batch_y_jitter)
    print(batch_x_jitter.shape,batch_y_jitter.shape)
  batch_x_yield, batch_y_yield = np.vstack(list_batch_x), np.vstack(list_batch_y)

  return batch_x_yield, batch_y_yield

def block(input, filters, kernal_size):
    cnn = tf.keras.layers.Conv1D(filters, 1, padding='SAME', activation='relu')(input)
    cnn = tf.keras.layers.LayerNormalization()(cnn)
    #cnn = tf.keras.layers.Dropout(0.3)(cnn)

    cnn = tf.keras.layers.Conv1D(filters, kernal_size, padding='SAME',activation='relu')(cnn)
    cnn = tf.keras.layers.LayerNormalization()(cnn)
    #cnn = tf.keras.layers.Dropout(0.3)(cnn)

    cnn = tf.keras.layers.Conv1D(filters, 1, padding='SAME', activation='relu')(cnn)
    cnn = tf.keras.layers.LayerNormalization()(cnn)
    input = tf.keras.layers.Conv1D(filters, 1)(input)
    output = tf.keras.layers.Add()([input, cnn])
    return output

def block2(input, filters=128, kernal_size=5):
    input = block(input, filters, kernal_size)
    input = tf.keras.layers.MaxPooling1D(2)(input)
    input = tf.keras.layers.SpatialDropout1D(0.5)(input)
    input = block(input, filters//2, kernal_size)
    output = tf.keras.layers.GlobalAveragePooling1D()(input)
    return output

def Simple_Resnet(input_shape, num_classes):
    inputs = tf.keras.layers.Input(shape=input_shape[1:])
    seq_3 = block2(inputs, kernal_size=3)
    seq_5 = block2(inputs, kernal_size=5)
    seq_7 = block2(inputs, kernal_size=7)
    seq = tf.keras.layers.concatenate([seq_3, seq_5, seq_7])
    seq = tf.keras.layers.Dense(512, activation='relu')(seq)
    seq = tf.keras.layers.Dropout(0.3)(seq)
    seq = tf.keras.layers.Dense(128, activation='relu')(seq)
    seq = tf.keras.layers.Dropout(0.3)(seq)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(seq)

    model = tf.keras.models.Model(inputs=[inputs], outputs=[outputs])

    return model

class CustomModel(tf.keras.Model):
    '''自定义模型'''
    def __init__(self):
        '''初始化模型层'''
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            filters=64, kernel_size=3,strides=1,padding='same',activation='relu')
        self.batchnorm1=tf.keras.layers.BatchNormalization()
        self.disout1 = Disout(0.3, block_size=3)


        self.conv2 = tf.keras.layers.Conv1D(
            filters=128, kernel_size=3,strides=1,padding='same',activation='relu')
        self.batchnorm2=tf.keras.layers.BatchNormalization()
        self.disout2 = Disout(0.3, block_size=3)
        self.pool1 = tf.keras.layers.MaxPool1D()

        self.conv3 = tf.keras.layers.Conv1D(
            filters=256, kernel_size=3,strides=1,padding='valid',activation='relu')
        #,kernel_regularizer=tf.keras.regularizers.l1_l2(0.01)
        self.batchnorm3=tf.keras.layers.BatchNormalization()
        self.disout3 = Disout(0.3, block_size=3)
        self.pool2 = tf.keras.layers.MaxPool1D()

        self.conv4 = tf.keras.layers.Conv1D(
            filters=256, kernel_size=3,strides=1,padding='same',activation='relu')
        self.batchnorm4=tf.keras.layers.BatchNormalization()
        self.disout4 = Disout(0.3, block_size=3)

        self.flatten = tf.keras.layerstf.keras.layers.MaxPool1D()
        self.disout5 = Disout1D(0.5, block_size=1)
        self.dropout= tf.keras.layers.Dropout(0.3)
        self.fc = tf.keras.layers.Dense(19, activation='softmax')

    def call(self, x):
        '''运算部分'''
        x=self.conv1(x)
        x=self.batchnorm1(x)
        #x=self.disout1(x)


        x=self.conv2(x)
        x=self.batchnorm2(x)
        #x=self.disout2(x)
        x=self.pool1(x)

        x=self.conv3(x)
        x=self.batchnorm3(x)
        #x=self.disout3(x)
        #x=self.pool2(x)

        x=self.conv4(x)
        x=self.batchnorm4(x)
        #x=self.disout4(x)

        x=self.flatten(x)
        x=self.dropout(x)
        x=self.fc(x)
        return x

proba_t = np.zeros((7500, 19))
val_loss = []
val_acc = []
val_acc_combo=[]
seeds = [44, 2020]
for seed in seeds:
  kfold = StratifiedKFold(10, shuffle=True, random_state=seed)
  for fold, (xx, yy) in enumerate(kfold.split(x, y_cls)):
      y_cls_ = tf.keras.utils.to_categorical(y_cls, num_classes=19)
      y_sence_ = tf.keras.utils.to_categorical(y_scene, num_classes=4)
      y_action_ = tf.keras.utils.to_categorical(y_action, num_classes=7)
      #model=CustomModel()
      model=Simple_Resnet(x.shape, 19)
      model.compile(loss=tf.losses.CategoricalCrossentropy(label_smoothing=0.1),
                    #"categorical_crossentropy"
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                    metrics=['acc',get_acc_combo()])
      plateau = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_acc_combo",
                                  verbose=1,
                                  mode='max',
                                  factor=0.4,
                                  patience=4)
      early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_acc_combo',
                                    verbose=1,
                                    mode='max',
                                    patience=15)
      if not os.path.exists('./models/ComplexConv1D_1'):
        os.makedirs('./models/ComplexConv1D_1')  

      checkpoint = tf.keras.callbacks.ModelCheckpoint(f'./models/ComplexConv1D_1/fold{fold}.h5',
                                  monitor='val_acc_combo',
                                  verbose=1,
                                  mode='max',
                                  save_best_only=True)
      fold_x,fold_y=data_generator(x[xx],y_cls_[xx], dataAugment=True, shuffle=True)
      model.summary()
      print("fold",fold)
      hist = model.fit(
                fold_x,fold_y,
                batch_size=256,
                epochs=150,
                verbose=1,
                shuffle=True,
                validation_data=(x[yy], y_cls_[yy]),
                callbacks=[plateau, early_stopping, checkpoint])
      
      val_loss.append(np.min(hist.history['val_loss']))
      val_acc.append(np.max(hist.history['val_acc']))
      val_acc_combo.append(np.max(hist.history['val_acc_combo']))
      model.load_weights(f'./models/ComplexConv1D_1/fold{fold}.h5')
      proba_t+= model.predict(t, verbose=1, batch_size=1024) / kfold.n_splits/ len(seeds)


current = datetime.now()
current=current.strftime('%m-%d-%H-%M')
txt_acc=np.mean(val_acc)
txt_acc_combo=np.mean(val_acc_combo)


if not os.path.exists('./stacking'):
  os.makedirs('./stacking')
#生成预测概率文件，用于stacking
np.savetxt("stacking/complexconv1d_1.txt",proba_t)

#生成半监督训练数据
sub["proba"]=np.max(proba_t,axis=1)
sub.to_csv("data/semi_test.csv",index=False)
