import os
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from tensorflow import keras

from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
import tensorflow as tf
import pandas as pd
import numpy as np
import random
import pickle
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# 特征列名称
src_names = ['acc_x', 'acc_y', 'acc_z', 'acc_xg', 'acc_yg', 'acc_zg', 'acc', 'acc_g']

def handle_features(data):
    data.drop(columns=['time_point'], inplace=True)

    data['acc'] = (data.acc_x ** 2 + data.acc_y ** 2 + data.acc_z ** 2) ** 0.5
    data['acc_g'] = (data.acc_xg ** 2 + data.acc_yg ** 2 + data.acc_zg ** 2) ** 0.5
    return data
    
# 构造numpy特征矩阵
def handle_mats(grouped_data):
    mats = [i.values for i in grouped_data]
    # padding
    for i in range(len(mats)):
        padding_times = 61 - mats[i].shape[0]
        for j in range(padding_times):
            mats[i] = np.append(mats[i], [[0 for _ in range(mats[i].shape[1])]], axis=0)

    mats_padded = np.zeros([len(mats), 61, mats[0].shape[1]])
    for i in range(len(mats)):
        mats_padded[i] = mats[i]

    return mats_padded

def get_test_data(use_scaler=True):
    FILE_NAME = "data/sensor_test.csv"
    data = handle_features(pd.read_csv(FILE_NAME))
    if use_scaler:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        data[src_names] = scaler.transform(data[src_names].values)

    grouped_data = [i.drop(columns='fragment_id') for _, i in data.groupby('fragment_id')]
    return handle_mats(grouped_data)

def get_train_data(use_scaler=True, shuffle=True, pseudo_labels_file=None):
    df = pd.read_csv("data/sensor_train.csv")

    # 简单拼接伪标签
    if pseudo_labels_file != None:
        df = df.append(pd.read_csv(pseudo_labels_file))
    data = handle_features(df)

    # 标准化，并将统计值保存
    if use_scaler:
        scaler = StandardScaler()
        scaler.fit(data[src_names].values)  
        with open('scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        data[src_names] = scaler.transform(data[src_names].values)

    grouped_data = [i.drop(columns='fragment_id') for _, i in data.groupby('fragment_id')]
    train_labels = np.array([int(i.iloc[0]['behavior_id']) for i in grouped_data])
    for i in range(len(grouped_data)):
        grouped_data[i].drop(columns='behavior_id', inplace=True)
    train_data = handle_mats(grouped_data)
    
    if shuffle:
        index = [i for i in range(len(train_labels))]
        np.random.seed(2020)
        np.random.shuffle(index)

        train_data = train_data[index]
        train_labels = train_labels[index]

    return train_data, train_labels

def get_train_test_data(use_scaler=True, shuffle=True, pseudo_labels_file=None):
    train_data, train_lables = get_train_data(use_scaler, shuffle, pseudo_labels_file=pseudo_labels_file)
    test_data = get_test_data(use_scaler)
    return train_data, train_lables, test_data

class WarmupExponentialDecay(Callback):
    def __init__(self,lr_base=0.0002,lr_min=0.0,decay=0,warmup_epochs=0):
        self.num_passed_batchs = 0   #一个计数器
        self.warmup_epochs=warmup_epochs  
        self.lr=lr_base #learning_rate_base
        self.lr_min=lr_min #最小的起始学习率,此代码尚未实现
        self.decay=decay  #指数衰减率
        self.steps_per_epoch=0 #也是一个计数器
    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        if self.steps_per_epoch==0:
            #防止跑验证集的时候呗更改了
            if self.params['steps'] == None:
                self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
            else:
                self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            K.set_value(self.model.optimizer.lr,
                        self.lr*(self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
        else:
            K.set_value(self.model.optimizer.lr,
                        self.lr*((1-self.decay)**(self.num_passed_batchs-self.steps_per_epoch*self.warmup_epochs)))
        self.num_passed_batchs += 1
    def on_epoch_begin(self,epoch,logs=None):
    #用来输出学习率的,可以删除
        print("learning_rate:",K.get_value(self.model.optimizer.lr))
        
        
kfcv_seed = 1998
kfold_func = StratifiedKFold
data_enhance_method = []
k = 10

def set_data_enhance(val):
    if not isinstance(val, list):
        val = [val]
    global data_enhance_method
    data_enhance_method = val

mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3', 
    4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5', 
    8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6', 
    12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6', 
    16: 'C_2', 17: 'C_5', 18: 'C_6'}

reversed_mapping = {value: key for key, value in mapping.items()}

def decode_label(label_code):
    str = mapping[label_code]
    scene_code = ord(str.split('_')[0]) - ord('A')
    action_code = ord(str.split('_')[1]) - ord('0')
    return scene_code, action_code

def kfcv_evaluate(model_name, x, y):
    kfold = kfold_func(n_splits=k, shuffle=True, random_state=kfcv_seed)
    evals = {'loss':0.0, 'accuracy':0.0}
    index = 0

    for train, val in kfold.split(x, np.argmax(y, axis=-1)):
        print('Processing fold: %d (%d, %d)' % (index, len(train), len(val)))
        
        model = keras.models.load_model('./models/%s/part_%d.h5' % (model_name, index))

        loss, acc = model.evaluate(x=x[val], y=y[val])
        evals['loss'] += loss / k
        evals['accuracy'] += acc / k
        index += 1
    return evals

def kfcv_predict(model_name, inputs):
    path = './models/' + model_name + '/'
    models = []
    for i in range(k):
        models.append(keras.models.load_model(path + 'part_%d.h5' % i))

    print('%s loaded.' % model_name)
    result = []
    for j in range(k):
        result.append(models[j].predict(inputs))

    print('result got')
    result = sum(result) / k
    return result

def kfcv_fit(builder, x, y, epochs, checkpoint_path, verbose=2, batch_size=64):
    kfold = kfold_func(n_splits=k, shuffle=True, random_state=kfcv_seed)
    histories = []
    evals = []

    if checkpoint_path[len(checkpoint_path) - 1] != '/':
        checkpoint_path += '/'

    for i in range(k):
        if os.path.exists(checkpoint_path + 'part_%d.h5' % i):
            os.remove(checkpoint_path + 'part_%d.h5' % i)

    for index, (train, val) in enumerate(kfold.split(x, np.argmax(y, axis=-1))):
        print('Processing fold: %d (%d, %d)' % (index, len(train), len(val)))
        model = builder()

        x_train = x[train]
        y_train = y[train]

        if len(data_enhance_method) > 0:
            x_train_copy = np.copy(x_train)
            y_train_copy = np.copy(y_train)
            for method in data_enhance_method:
                x_, y_ = data_enhance(method, x_train_copy, y_train_copy)
                x_train = np.r_[x_train, x_]
                y_train = np.r_[y_train, y_]
            x_train, y_train = shuffle(x_train, y_train)
            print('Data enhanced (%s) => %d' % (' '.join(data_enhance_method), len(x_train)))

        checkpoint = keras.callbacks.ModelCheckpoint(checkpoint_path + 'part_%d.h5' % index,
                                 monitor='val_accuracy',
                                 verbose=0,
                                 mode='max',
                                 save_best_only=True)
        warmup = WarmupExponentialDecay(lr_base=0.001,decay=0.00002,warmup_epochs=5)
        early_stopping = keras.callbacks.EarlyStopping(patience=10),
        Reduce = keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy',factor=0.2,patience=5,verbose=1,mode='auto',epsilon=0.0001)
        h = model.fit(x=x_train, y=y_train,
                epochs=epochs,
                verbose=verbose,
                validation_data=(x[val], y[val]),
                callbacks=[checkpoint,Reduce,early_stopping],
                batch_size=batch_size,
                shuffle=True
                )
        evals.append(model.evaluate(x=x[val], y=y[val]))
        histories.append(h)
        del model
        gc.collect()
    return histories, evals

def data_enhance(method, train_data, train_labels):
    if method == 'noise':
        noise = train_data + np.random.normal(0, 0.1, size=train_data.shape)
        return noise, train_labels
    
    elif method == 'mixup':
        index = [i for i in range(len(train_labels))]
        np.random.shuffle(index)

        x_mixup = np.zeros(train_data.shape)
        y_mixup = np.zeros(train_labels.shape)

        for i in range(len(train_labels)):
            x1 = train_data[i]
            x2 = train_data[index[i]]
            y1 = train_labels[i]
            y2 = train_labels[index[i]]

            factor = np.random.beta(0.1, 0.3)

            x_mixup[i] = x1 * factor + x2 * (1 - factor)
            y_mixup[i] = y1 * factor + y2 * (1 - factor)

        return x_mixup, y_mixup

def save_results_avg(path, output):
    print('saving...')

    df_r = pd.DataFrame(columns=['fragment_id', 'behavior_id'])
    for i in range(len(output)):
        behavior_id = output[i]
        df_r = df_r.append(
            {'fragment_id': i, 'behavior_id': behavior_id}, ignore_index=True)
    df_r.to_csv(path, index=False)
    
def save_results_prob(path, output):
    np.savetxt(path, output)

    
def infer(model_name, inputs, csv_output):
    result = kfcv_predict(model_name, inputs)
    save_results_prob(csv_output, result)
    print('- END -')

def shuffle(data, labels, seed=None):
    index = [i for i in range(len(labels))]
    if seed != None:
        np.random.seed(seed)
    np.random.shuffle(index)
    return data[index], labels[index]

def BLOCK(seq, filters, kernal_size):
    cnn = keras.layers.Conv1D(filters, 1, padding='SAME', activation='relu')(seq)
    cnn = keras.layers.LayerNormalization()(cnn)
    
    cnn = keras.layers.Dropout(0.3)(cnn)
    cnn = keras.layers.Conv1D(filters, kernal_size, padding='SAME', activation='relu')(cnn)
    cnn = keras.layers.LayerNormalization()(cnn)
    
    cnn = keras.layers.Dropout(0.3)(cnn)
    cnn = keras.layers.Conv1D(filters, 1, padding='SAME', activation='relu')(cnn)
    cnn = keras.layers.LayerNormalization()(cnn)
    
    #cnn = keras.layers.Dropout(0.3)(cnn)
    seq = keras.layers.Conv1D(filters, 1)(seq)
    seq = keras.layers.Add()([seq, cnn])
    return seq

def BLOCK2(seq, filters=128, kernal_size=5):
    seq = BLOCK(seq, filters, kernal_size)
    seq = keras.layers.MaxPooling1D(2)(seq)
    seq = keras.layers.SpatialDropout1D(0.3)(seq)
    seq = BLOCK(seq, filters//2, kernal_size)
    seq = keras.layers.GlobalAveragePooling1D()(seq)
    return seq

def ComplexConv1D(input_shape, num_classes):
    inputs = keras.layers.Input(shape=input_shape[1:])
    seq_3 = BLOCK2(inputs, kernal_size=3)
    seq_5 = BLOCK2(inputs, kernal_size=5)
    seq_7 = BLOCK2(inputs, kernal_size=7)
    seq = keras.layers.concatenate([seq_3, seq_5, seq_7])
    seq = keras.layers.Dense(512, activation='relu')(seq)
    seq = keras.layers.Dropout(0.3)(seq)
    seq = keras.layers.Dense(64, activation='relu')(seq)
    seq = keras.layers.Dropout(0.3)(seq)
    outputs = keras.layers.Dense(num_classes, activation='softmax')(seq)

    model = keras.models.Model(inputs=[inputs], outputs=[outputs])

    model.compile(optimizer=tf.optimizers.Adam(1e-3),
            loss=tf.losses.CategoricalCrossentropy(label_smoothing=0.1),           
            metrics=['accuracy'])

    return model


# 根据自己的文件目录进行修改
txdir='/content/drive/My Drive/xwbank2020/'
if os.path.exists(txdir):
  os.chdir(txdir)

# 导入精心挑选的pseudo labels
train_data, train_labels, test_data = get_train_test_data(pseudo_labels_file=
                                                'data/pseudo_train_2756.csv')
# 设置数据增强方式 (noise, mixup or both)
set_data_enhance(['noise','mixup'])

# 转换成float32节省显存，以及one_hot编码
num_classes = 19
train_data = tf.cast(train_data, tf.float32).numpy()
train_labels = tf.one_hot(train_labels, num_classes).numpy()



if not os.path.exists('./models/ComplexConv1D_2'):
  os.makedirs('./models/ComplexConv1D_2')
# 训练
histories, evals = kfcv_fit(builder=lambda : ComplexConv1D(train_data.shape, 19),
                                x=train_data, y=train_labels,
                                epochs=100,
                                checkpoint_path = './models/ComplexConv1D_2/',
                                batch_size=64
                                )


# 评估
kfcv_evaluate('ComplexConv1D_2', train_data, train_labels)

# 推断
infer('ComplexConv1D_2', get_test_data(), 'stacking/complexconv1d_2.txt')