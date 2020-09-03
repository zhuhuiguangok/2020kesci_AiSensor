import numpy as np
import os
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
# 基于 tensorflow.keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.activations import relu
import tensorflow.keras.backend as K


def standardization(X):
    # x1 = X.transpose(0, 1, 3, 2)
    x1 = X
    x2 = x1.reshape(-1, x1.shape[-2])
    # mean = [8.03889039e-03, -6.41381949e-02, 2.37856977e-02, 8.64949391e-01,
    #         2.80964889e+00, 7.83041714e+00, 6.44853358e-01, 9.78580749e+00]
    # std = [0.6120893, 0.53693888, 0.7116134, 3.22046385, 3.01195336, 2.61300056, 0.87194132, 0.68427254]
    mu=np.mean(x2,axis=0)
    sigma=np.std(x2,axis=0)
    print(mu,sigma)
    x3 = ((x2 - mu) / (sigma))
    # x4 = x3.reshape(x1.shape).transpose(0, 1, 3, 2)
    x4 = x3.reshape(x1.shape)
    return x4


# 根据自己的文件目录进行修改
txdir='/content/drive/My Drive/xwbank2020/'
if os.path.exists(txdir):
  os.chdir(txdir)

sub = pd.read_csv('data/submit_template.csv')
df_train = pd.read_csv( 'data/sensor_train.csv' )
df_test  = pd.read_csv( 'data/sensor_test.csv' )
y = df_train.groupby('fragment_id')['behavior_id'].min()

df_test['fragment_id'] += 10000
df_data = pd.concat([df_train, df_test],axis=0,ignore_index=True)

df_data['acc'] = (df_data['acc_x'] ** 2 + df_data['acc_y'] ** 2 + df_data['acc_z'] ** 2) ** 0.5
df_data['accg'] = (df_data['acc_xg'] ** 2 + df_data['acc_yg'] ** 2 + df_data['acc_zg'] ** 2) ** 0.5


label_feat = 'behavior_id'
train = df_data[df_data[label_feat].isna()==False]
test = df_data[df_data[label_feat].isna()==True]
test['fragment_id'] -=10000

x = np.zeros((7292, 60, 8, 1))
t = np.zeros((7500, 60, 8, 1))
for i in tqdm(range(7292)):
    tmp = train[train.fragment_id == i][:60]
    x[i,:,:, 0] = resample(tmp.drop(['fragment_id', 'time_point', 'behavior_id'],
                                    axis=1), 60, np.array(tmp.time_point))[0]
for i in tqdm(range(7500)):
    tmp = test[test.fragment_id == i][:60]
    t[i,:,:, 0] = resample(tmp.drop(['fragment_id', 'time_point', 'behavior_id'],
                                    axis=1), 60, np.array(tmp.time_point))[0]
x = standardization(x)
t = standardization(t)                                    



df_tr_te = pd.read_pickle("pickel/df_fea1.pkl")

label_feat = 'behavior_id'
train_df = df_tr_te[((df_tr_te[label_feat].isna()==False) & (df_tr_te[label_feat] >=0))].reset_index(drop=True)
test_df  = df_tr_te[((df_tr_te[label_feat].isna()==True) | (df_tr_te[label_feat] < 0))].reset_index(drop=True)

drop_feat = []
used_feat = [f for f in train_df.columns if f not in (['fragment_id', label_feat] + drop_feat)]
print(len(used_feat))
print(used_feat)

# 根据lightgbm重要性排序得到的特征
selected_feat = ['fragment_id_acc_yg_max',
 'fragment_id_xy_median',
 'fragment_id_acc_yg_min',
 'acc_xg_fftA_0',
 
 'acc_yg_fftA_0',
 'accg_fftA_0',
 'fragment_id_acc_yg_gap',
 'fragment_id_xy_g_max',
 'fragment_id_acc_xg_mean',
 'fragment_id_acc_xg_max',
 'cnt',
 'acc_y_fftA_0',
 'fragment_id_acc_yg_mean',
 'fragment_id_acc_yg_std',
 'fragment_id_acc_yg_median',
 'acc_zg_fftA_0',
 'fragment_id_acc_zg_min',
 'fragment_id_accg_median',
 'fragment_id_acc_xg_min',
 'fragment_id_acc_xg_median']
train_stat = train_df[selected_feat]
test_stat  = test_df[selected_feat]
train_stat = np.array(train_stat.values)
test_stat = np.array(test_stat.values)
def autos(X):
    m, n = X.shape[0], X.shape[1] 
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0, ddof=1)
    X_ = ((X - mu) / (sigma))
    return X_
train_stat = autos(train_stat)
test_stat = autos(test_stat)


from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(zoom_range = 0,
                            height_shift_range = 0.2,
                            width_shift_range = 0,
                            rotation_range = 0)

# Here is the function that merges our two generators
# We use the exact same generator with the same random seed for both the y and angle arrays
def gen_flow_for_two_inputs(X1, X2, y, batch_size):
    genX1 = datagen.flow(X1,y,  batch_size=batch_size,seed = 2020)
    genX2 = datagen.flow(X1,X2, batch_size=
                         batch_size,seed = 2020)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            #Assert arrays are equal - this was for peace of mind, but slows down training
            #np.testing.assert_array_equal(X1i[0],X2i[0])
            yield [X1i[0], X2i[1]], X1i[1]



# 加种子
def Net():
    K.clear_session()
    input = Input(shape=(60, 8, 1))
    hin = Input(shape=(20, ))
    X = Conv2D(filters= 64,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(input)
    X = BatchNormalization()(X)
    X = Conv2D(filters= 128,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    X = BatchNormalization()(X)
    X = MaxPooling2D()(X)
    X = Conv2D(filters=256,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    X = BatchNormalization()(X)
    X = Conv2D(filters=512,
               kernel_size=(3, 3),
               activation='relu',
               padding='same')(X)
    X = BatchNormalization()(X)
    X = GlobalMaxPooling2D()(X)

    magic = Dense(64, activation='relu')(hin)####
    merge = concatenate([X, magic])
    bp = BatchNormalization()(merge)
    bp = Dense(64, activation='relu')(bp)####
    bp = BatchNormalization()(bp)
    merge = Dropout(0.5)(bp)
    y = Dense(19, activation='softmax')(merge)
    return Model(inputs=[input, hin], outputs=y)


proba_t = np.zeros((7500, 19))
val_loss = []
val_acc = []
histories = []
batch_size = 128
seeds = [44, 2020, 527, 1527]
for seed in seeds:
  kfold = StratifiedKFold(5, shuffle=True, random_state=seed)
  for fold, (xx, yy) in enumerate(kfold.split(x, y)):
      y_ = to_categorical(y, num_classes=19)
      model = Net()
      print(model.summary())
      model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(1e-3),
                    metrics=['acc'])
      plateau = ReduceLROnPlateau(monitor="val_acc",
                                  verbose=0,
                                  mode='max',
                                  factor=0.6,
                                  patience=4)
      early_stopping = EarlyStopping(monitor='val_acc',
                                    verbose=0,
                                    mode='max',
                                    patience=12)
      if not os.path.exists('./models/conv2d_withoutlabelsmoothing'):
        os.makedirs('./models/conv2d_withoutlabelsmoothing')  
      checkpoint = ModelCheckpoint(f'./models/conv2d_withoutlabelsmoothing/fold{fold}.h5',
                                  monitor='val_acc',
                                  verbose=0,
                                  mode='max',
                                  save_best_only=True)
      gen_flow = gen_flow_for_two_inputs(x[xx], train_stat[xx], y_[xx], batch_size)
      print(x[xx].shape[0])
      hist =  model.fit(gen_flow,steps_per_epoch=x[xx].shape[0] / batch_size,
                epochs=150,
                verbose=1,
                shuffle=True,
                validation_data=([x[yy],train_stat[yy]], y_[yy]),
                callbacks=[plateau, early_stopping, checkpoint])
      val_loss.append(np.min(hist.history['val_loss']))
      val_acc.append(np.max(hist.history['val_acc']))
      histories.append(hist)
      model.load_weights(f'./models/conv2d_withoutlabelsmoothing/fold{fold}.h5')
      proba_t += model.predict([t,test_stat], verbose=0, batch_size=1024) / kfold.n_splits/ len(seeds)
print('log loss:', np.mean(val_loss))
print('val_acc:', np.mean(val_acc))

if not os.path.exists('./stacking'):
  os.makedirs('./stacking')
np.savetxt('stacking/conv2d_withoutlabelsmoothing.txt',proba_t)