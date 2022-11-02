# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from Model import train, predict

'''
if frame == "pytorch":
    from model.model_pytorch import train, predict
elif frame == "keras":
    from model.model_keras import train, predict
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
elif frame == "tensorflow":
    from model.model_tensorflow import train, predict
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'    # tf和keras下会有很多tf的warning，但不影响训练
else:
    raise Exception("Wrong frame selection")
'''
class Config:
    ## data parameters
    feature_columns = [1, 2, 3, 4, 5]     # select feature columns you want for input and output
    label_columns = [4, 5]                  # columns for output

    label_in_feature_index = (lambda x,y: [x.index(i) for i in y])(feature_columns, label_columns)

    predict_day = 1
    # the length of your output, this LSTM model is originally designed for stock price prediction,
    # but I am too lazy to do correction. 変数名を変更したいなら、すべての所で変更すべきだ

    ## network parameters
    input_size = len(feature_columns)
    output_size = len(label_columns)

    hidden_size = 256           # hidden layer size.2^nが推奨
    lstm_layers = 2             # hidden layer数
    dropout_rate = 0.1          # dropout rate
    time_step = 167             # how much data used for training in one step
    # For g2tau data, one trail has 501 points so I split each of them into 3 parts to seperate prior, middle, post period.
    # Compared to time_step=501, 167の方の検証精度が良くなった

    ## training parameters
    do_train = True
    do_predict = True
    add_train = False           # haven't try this yet but should be able to add trained data to increase accuracy
    shuffle_train_data = True   # shuffle flag, make sure this is true otherwise your training would be meaningless
    use_cuda = True          # using GPU or not during training. set this to false if you don't have cuda installed
    # Cuda installationと環境構築は私が作ったマニュアルに参考できる

    train_data_rate = 0.9
    valid_data_rate = 0.1
    # train-test data split for 9:1, similar to leave-one-out

    batch_size = 501            # up to your dataset size
    learning_rate = 0.0001     # important! try different rate to avoid overfitting
    epoch = 200                 # max epoch number for training
    patience = 3                # if loss hasn't been reduced for 3 times in a row then stop
    random_seed = 48            # random seed to ensure the reproductivity

    do_continue_train = False    # 各batchの訓練で、前のbatchの結果を今回のinitiate stateとして使う
    continue_flag = ""           # g2tau dataはtrailsが独立であって使用していなかった
    if do_continue_train:
        shuffle_train_data = False
        batch_size = 501
        continue_flag = "continue_"

    ## debug mode
    debug_mode = False  # for debug
    debug_num = 500  #speedup to check warnings

    ## framework
    used_frame = "pytorch"       # using pytorch for training
    model_postfix = {"pytorch": ".pth", "keras": ".h5", "tensorflow": ".ckpt"}
    model_name = "model_" + continue_flag + used_frame + model_postfix[used_frame]

    ## directory
    train_data_path = "./gtau2.3edited.csv"       # dataset directory
    model_save_path = "./checkpoint/" + used_frame + "/"        # trained-model saving directory
    figure_save_path = "./figure/"
    log_save_path = "./log/"
    do_log_print_to_screen = True
    do_log_save_to_file = True                    # save log or not
    do_figure_save = False
    do_train_visualized = False                   # planned to do visualization but failed
    # It would be delightful if any one can continue to do visualization work

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)              # makedirs to make file if save path not exists
    if not os.path.exists(figure_save_path):
        os.mkdir(figure_save_path)
    if do_train and (do_log_save_to_file or do_train_visualized):
        cur_time = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        log_save_path = log_save_path + cur_time + '_' + used_frame + "/"
        os.makedirs(log_save_path)


class Data:
    def __init__(self, config):
        self.config = config
        self.data, self.data_column_name = self.read_data()

        self.data_num = self.data.shape[0]
        self.train_num = int(self.data_num * self.config.train_data_rate)

        self.mean = np.mean(self.data, axis=0)              #　平均値と偏差
        self.std = np.std(self.data, axis=0)
        self.norm_data = (self.data - self.mean)/self.std   # データ正規化

        self.start_num_in_test = 0

    ## データ読み取り
    def read_data(self):
        if self.config.debug_mode:
            init_data = pd.read_csv(self.config.train_data_path, nrows=self.config.debug_num,
                                    usecols=self.config.feature_columns)
        else:
            init_data = pd.read_csv(self.config.train_data_path, usecols=self.config.feature_columns)
        return init_data.values, init_data.columns.tolist()     # .columns.tolist() はcolumnのラベルを読み取ること

    def get_train_and_valid_data(self):
        feature_data = self.norm_data[:self.train_num]
        label_data = self.norm_data[:, self.config.label_in_feature_index]

        if not self.config.do_continue_train:
            # every 501 data will be made into one sample and do training
            # つまり、1~501,502~1002のようにデータを分離している
            train_x = [
                feature_data[i * self.config.time_step: (i + 1) * self.config.time_step]
                for i in range(self.train_num // self.config.time_step)]
            train_y = [
                label_data[i * self.config.time_step: (i + 1) * self.config.time_step]
                for i in range(self.train_num // self.config.time_step)]

            # train_x = [feature_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
            # train_y = [label_data[i:i+self.config.time_step] for i in range(self.train_num-self.config.time_step)]
        else:
            # ignore this part please
            '''train_x = [feature_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]
            train_y = [label_data[start_index + i*self.config.time_step : start_index + (i+1)*self.config.time_step]
                       for start_index in range(self.config.time_step)
                       for i in range((self.train_num - start_index) // self.config.time_step)]
'''
            train_x = [



                feature_data[i * self.config.time_step: (i + 1) * self.config.time_step]
                for start_index in range(self.config.time_step)
                for i in range((self.train_num - start_index) // self.config.time_step)]
            train_y = [
                label_data[i * self.config.time_step: (i + 1) * self.config.time_step]
                for start_index in range(self.config.time_step)
                for i in range((self.train_num - start_index) // self.config.time_step)]
        train_x, train_y = np.array(train_x), np.array(train_y)

        train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=self.config.valid_data_rate,
                                                              random_state=self.config.random_seed,
                                                              shuffle=self.config.shuffle_train_data)
        # split train and validation and do the shuffle (in samples aspect)
        # we had train and test splited before and this time we further split train dataset into train and valid
        return train_x, valid_x, train_y, valid_y

    def get_test_data(self, return_label_data=False):
        feature_data = self.norm_data[self.train_num:]
        sample_interval = min(feature_data.shape[0], self.config.time_step)     # prevent time_step > dataset size, designed for debug mode
        self.start_num_in_test = feature_data.shape[0] % sample_interval  # abandon those left over for 1 sample
        # e.q. dataset size is 1000, only 1 sample (501) will be took, and the first 499 points will be abandoned since they can't fill up 1 sample

        time_step_size = feature_data.shape[0] // sample_interval


        test_x = [feature_data[self.start_num_in_test+i*sample_interval : self.start_num_in_test+(i+1)*sample_interval]
                   for i in range(time_step_size)]
        if return_label_data:
            label_data = self.norm_data[self.train_num + self.start_num_in_test:, self.config.label_in_feature_index]
            return np.array(test_x), label_data
        return np.array(test_x)

def load_logger(config):
    logger = logging.getLogger()
    logger.setLevel(level=logging.DEBUG)

    # StreamHandler
    if config.do_log_print_to_screen:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter(datefmt='%Y/%m/%d %H:%M:%S',
                                      fmt='[ %(asctime)s ] %(message)s')
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    # FileHandler
    if config.do_log_save_to_file:
        file_handler = RotatingFileHandler(config.log_save_path + "out.log", maxBytes=1024000, backupCount=5)
        file_handler.setLevel(level=logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # record parameter settings in log file
        config_dict = {}
        for key in dir(config):
            if not key.startswith("_"):
                config_dict[key] = getattr(config, key)
        config_str = str(config_dict)
        config_list = config_str[1:-1].split(", '")
        config_save_str = "\nConfig:\n" + "\n'".join(config_list)
        logger.info(config_save_str)

    return logger

def draw(config: Config, origin_data: Data, logger, predict_norm_data: np.ndarray):
    label_data = origin_data.data[origin_data.train_num + origin_data.start_num_in_test : ,
                                            config.label_in_feature_index]
    predict_data = predict_norm_data * origin_data.std[config.label_in_feature_index] + \
                   origin_data.mean[config.label_in_feature_index]
    assert label_data.shape[0] == predict_data.shape[0], "The element number in origin and predicted data is different"

    label_name = [origin_data.data_column_name[i] for i in config.label_in_feature_index]
    label_column_num = len(config.label_columns)

    ## the following method of calculating accuracy will also do the job, you can try it if you want
    # label_norm_data = origin_data.norm_data[origin_data.train_num + origin_data.start_num_in_test:,
    #              config.label_in_feature_index]
    # loss_norm = np.mean((label_norm_data[config.predict_day:] - predict_norm_data[:-config.predict_day]) ** 2, axis=0)
    # logger.info("The mean squared error of stock {} is ".format(label_name) + str(loss_norm))

    loss = np.mean((label_data[config.predict_day:] - predict_data[:-config.predict_day] ) ** 2, axis=0)
    loss_norm = loss/(origin_data.std[config.label_in_feature_index] ** 2)
    average_label_data = np.mean(label_data[config.predict_day:])

    # print(np.shape(label_data))
    phantom = np.mean((label_data[config.predict_day:] - average_label_data) ** 2, axis=0)
    Rsquared = 1 - loss/phantom             # calculate the R square
    logger.info("The MSE of {} is ".format(label_name) + str(loss))
    logger.info("The mean squared error of {} is ".format(label_name) + str(loss_norm))
    logger.info("The R squared of {} is ".format(label_name) + str(Rsquared))
    label_X = range(origin_data.data_num - origin_data.train_num - origin_data.start_num_in_test)
    predict_X = [ x + config.predict_day for x in label_X]
    label_X_length = len(label_data[:, 0])

    '''
    df = []
    a1 = []
    a2 = []
    a3 = []
    a4 = []
    for ii in range(label_X_length):
        a1.append(label_data[ii, 0])
        a2.append(label_data[ii, 1])
        a3.append(predict_data[ii, 0])
        a4.append(predict_data[ii, 1])
    df = np.array([a1, a2, a3, a4])
    dataframe = pd.DataFrame(df.T)
    dataframe.to_csv("C:/Users/Liu Siwei/Desktop/DCS/Result.csv")
    '''

    if not sys.platform.startswith('linux'):
        for i in range(label_column_num):
            fig = plt.figure(i+1)                     # plotting figures of bottom and above
            plt.plot(label_X[10:50000], label_data[10:50000, i], label='label')
            plt.plot(predict_X[10:50000], predict_data[10:50000, i], label='predict')
            plt.title("Predict {} velocity".format(label_name[i]))
            fig.legend()
            # logger.info("The predicted {} velocity is: ".format(label_name[i]) +
                  # str(np.squeeze(predict_data[-config.predict_day:, i])))
            if config.do_figure_save:
                plt.savefig(config.figure_save_path+"{}predict_{}_with_{}.png".format(config.continue_flag, label_name[i], config.used_frame))

        plt.show()

def main(config):
    starttime = time.time()
    logger = load_logger(config)
    try:
        np.random.seed(config.random_seed)  # implement random seed
        data_gainer = Data(config)

        if config.do_train:
            train_X, valid_X, train_Y, valid_Y = data_gainer.get_train_and_valid_data()
            train(config, logger, [train_X, train_Y, valid_X, valid_Y])

        if config.do_predict:
            test_X, test_Y = data_gainer.get_test_data(return_label_data=True)
            pred_result = predict(config, test_X)       # this result hasn't be sent back to do regulation
            draw(config, data_gainer, logger, pred_result)
    except Exception:
        logger.error("Run Error", exc_info=True)
    endtime = time.time()
    print("Processing time is " + str(endtime - starttime))

if __name__=="__main__":
    import argparse
    # argparse方便于命令行下输入参数，可以根据需要增加更多
    parser = argparse.ArgumentParser()
    # parser.add_argument("-t", "--do_train", default=False, type=bool, help="whether to train")
    # parser.add_argument("-p", "--do_predict", default=True, type=bool, help="whether to train")
    # parser.add_argument("-b", "--batch_size", default=64, type=int, help="batch size")
    # parser.add_argument("-e", "--epoch", default=20, type=int, help="epochs num")
    args = parser.parse_args()

    con = Config()
    for key in dir(args):               # dir(args) 函数获得args所有的属性
        if not key.startswith("_"):     # 去掉 args 自带属性，比如__name__等
            setattr(con, key, getattr(args, key))   # 将属性值赋给Config

    main(con)