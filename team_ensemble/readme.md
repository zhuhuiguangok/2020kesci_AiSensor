### 1. 环境配置

- scikit-learn
- tqdm
- pandas
- numpy
- scipy
- tensorflow==2.3.0 (其他版本是否可以还需确定)

### 2. 特征提取
生成的特征将保存在pickle文件夹下
```shell
python extract_feature.py
```

### 5.训练模型

下面的命令行无需按照先后顺序，
运行后生成测试数据预测概率文件将保存在stacking文件夹下
保存的模型将会新建在models文件夹下
在使用Tesla K80GPU, 以下单个程序平均运行2~3小时
data文件夹下有半监督生成的伪标签数据, pseudo_train_2756.csv和semi_test_8_nobest_08-04-11-49_0.89783.csv

```shell
python train_conv1D_lstm.py
python train_complexconv1d_1.py
python train_complexconv1d_2.py
python train_conv2D_withlablesmoothing.py
python train_conv2D_withoutlablesmoothing.py

```

### 5. 最终结果预测

输出的结果保存在submit文件夹下

```shell
python weightbagging.py

```
