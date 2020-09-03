import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.signal import resample
from scipy.fftpack import fft
from scipy.signal import resample
from scipy.signal import welch

df_train = pd.read_csv( 'data/sensor_train.csv' )
df_test  = pd.read_csv( 'data/sensor_test.csv' )
y = df_train.groupby('fragment_id')['behavior_id'].min()

df_test['fragment_id'] += 10000
df_data = pd.concat([df_train, df_test],axis=0,ignore_index=True)
df = df_data.drop_duplicates(subset=['fragment_id']).reset_index(drop=True)[['fragment_id', 'behavior_id']]
df.head()

df_data['acc'] = (df_data['acc_x'] ** 2 + df_data['acc_y'] ** 2 + df_data['acc_z'] ** 2) ** 0.5
df_data['accg'] = (df_data['acc_xg'] ** 2 + df_data['acc_yg'] ** 2 + df_data['acc_zg'] ** 2) ** 0.5
df_data['xy'] = (df_data['acc_x'] ** 2 + df_data['acc_y'] ** 2) ** 0.5
df_data['xy_g'] = (df_data['acc_xg'] ** 2 + df_data['acc_yg'] ** 2) ** 0.5


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
        df_fea['_'.join([main_col, f, 'skew2'])] = (df_fea['_'.join([main_col, f, 'mean'])] - df_fea['_'.join([main_col, f, 'median'])]) / df_fea['_'.join([main_col, f, 'std'])]
        
    return df_fea

def get_fft_values(y_values, N, f_s):
    f_values = np.linspace(0.0, f_s/2.0, N//2)
    fft_values_ = fft(y_values)
    fft_values = 2.0/N * np.abs(fft_values_[0:N//2])
    return f_values, fft_values


def get_psd_values(y_values, N, f_s):
    f_values, psd_values = welch(y_values, fs=f_s)
    return f_values, psd_values


def show_topK_peak(fft_values,top_peak_number=5):  
    '''
    show_topK_peak(fft_values)
    '''
    peak_list = []
    for index,values in enumerate(fft_values):
        if index == 0 or index == len(fft_values) - 1:
            continue
        if fft_values[index] > fft_values[index - 1] and fft_values[index] > fft_values[index + 1]:
            peak_list.append((values,index))
    t_res = sorted(peak_list)[::-1]
#     print(t_res)
#     t_res = sorted(zip(fft_values,range(len(fft_values))))[::-1]
#     print(t_res)
    top_peak_A = [A for A,P in t_res[:top_peak_number]]
    top_peak_P = [P for A,P in t_res[:top_peak_number]]
#     print(top_peak_A)
    plt.plot(range(len(fft_values)),fft_values)
    plt.scatter(top_peak_P,top_peak_A)
    plt.show()
    
    
top_peak_number = 5    
def get_fft_topK_AP(array_with_time,feat_name,K=top_peak_number):
#     print(array_with_time)
    x,t = resample(array_with_time[feat_name], 120, np.array(array_with_time["time_point"]))
    f_values, fft_values = get_fft_values(x, N=120, f_s=5)
    peak_list = []
    for index,values in enumerate(fft_values):
        if index == 0 or index == len(fft_values) - 1:
            continue
        if fft_values[index] > fft_values[index - 1] and fft_values[index] > fft_values[index + 1]:
            peak_list.append((values,index))
    if len(peak_list) < 5:
        cnt = 5 - len(peak_list)
        for i in range(cnt):
            peak_list.append((0,-1))
    t_res = sorted(zip(fft_values,range(len(fft_values))))[::-1]
#     print(t_res)
    top_peak_A = [A for A,P in t_res[:top_peak_number]]
    top_peak_P = [P for A,P in t_res[:top_peak_number]]
    return [top_peak_A + top_peak_P]


top_peak_number = 5    
def get_psd_topK_AP(array_with_time,feat_name,K=top_peak_number):
#     print(array_with_time)
    x,t = resample(array_with_time[feat_name], 120, np.array(array_with_time["time_point"]))
    f_values, fft_values = get_psd_values(x, N=120, f_s=5)
    peak_list = []
    for index,values in enumerate(fft_values):
        if index == 0 or index == len(fft_values) - 1:
            continue
        if fft_values[index] > fft_values[index - 1] and fft_values[index] > fft_values[index + 1]:
            peak_list.append((values,index))
    if len(peak_list) < 5:
        cnt = 5 - len(peak_list)
        for i in range(cnt):
            peak_list.append((0,-1))
    
    t_res = sorted(zip(fft_values,range(len(fft_values))))[::-1]

    top_peak_A = [A for A,P in t_res[:top_peak_number]]
    top_peak_P = [P for A,P in t_res[:top_peak_number]]
     
    '''
    (19) Root mean square of the differences between two successive peaks;
    (20) Standard deviation of the intervals between two successive peaks;
    (21) The number of pairs of successive peaks intervals that differ by more than 50 ms.
    '''
    t_res2 = sorted(peak_list,key=lambda x:x[1])[::-1]
    diff_of_successive_peaks = np.zeros(len(t_res2) - 1)
    intervals_of_successive_peaks = np.zeros(len(t_res2) - 1)
    peak_values = np.array([p for p,i in peak_list])
    for index,i in enumerate(t_res2):
        if index == 0:
            continue
        diff_of_successive_peaks[index-1] = t_res2[index][0] - t_res2[index-1][0]
        intervals_of_successive_peaks[index-1] = t_res2[index][1] - t_res2[index-1][1]
    return [top_peak_A + top_peak_P]



# 获得统计特征
origin_fea_cols = ['acc_x','acc_y','acc_z','acc_xg','acc_yg','acc_zg','accg','xy','xy_g']
df_xyz_fea1 = get_1st_order_xyz_features(df_data,origin_fea_cols,main_col='fragment_id')


# 获得傅里叶特征
top_peak_number = 5    
oral_item = ['acc_x','acc_y','acc_z','acc_xg','acc_yg','acc_zg','accg','xy','xy_g']

for item in tqdm(oral_item):
    tmp = df_data[["fragment_id",item,"time_point"]].groupby(["fragment_id"],as_index=False)[item].agg(get_fft_topK_AP,feat_name=item)
    
    for A in range(top_peak_number):
        print(A)
        tmp[item+"_fftA_"+str(A)] = tmp[item].apply(lambda x:x[A])
        df = df.merge(tmp[["fragment_id",item+"_fftA_"+str(A)]],on='fragment_id',how='left')
    for P in range(top_peak_number):
        print(P)
        tmp[item+"_fftP_"+str(P)] = tmp[item].apply(lambda x:x[top_peak_number+P])
        df = df.merge(tmp[["fragment_id",item+"_fftP_"+str(P)]],on='fragment_id',how='left')
    
    tmp = df_data[["fragment_id",item,"time_point"]].groupby(["fragment_id"],as_index=False)[item].agg(get_psd_topK_AP,feat_name=item)
    for A in range(top_peak_number):
        tmp[item+"_psdA_"+str(A)] = tmp[item].apply(lambda x:x[A])
        df = df.merge(tmp[["fragment_id",item+"_psdA_"+str(A)]],on='fragment_id',how='left')
    for P in range(top_peak_number):
        tmp[item+"_psdP_"+str(P)] = tmp[item].apply(lambda x:x[top_peak_number+P])
        df = df.merge(tmp[["fragment_id",item+"_psdP_"+str(P)]],on='fragment_id',how='left')
df_tr_te = df.merge(df_xyz_fea1, on ='fragment_id', how = 'left')
if not os.path.exists('./pickle'):
        os.makedirs('./pickle')
df_tr_te.to_pickle( "./pickle/df_fea1.pkl")#保存一阶统计特征以及傅里叶特征


