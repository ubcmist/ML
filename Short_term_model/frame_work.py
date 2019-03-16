'''
This code is a framework to get biometric data, process data, create & train models.
Inspired from work of https://www.kaggle.com/coni57/model-from-arxiv-1805-00794
'''

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in
import math
import random
import pickle
import itertools
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error
from sklearn.utils import shuffle
from scipy.signal import resample
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Softmax, Add, Flatten, Activation# , Dropout
from keras import backend as K
from keras.optimizers import Adam, sgd
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os

def GetTimeRangeSampledDataFrame(df_x, time_y, sample_time_range, sampling_method = 'retrospective'):
    if sampling_method == 'retrospective':
        start_time = time_y - sample_time_range
        end_time = time_y
    elif sampling_method == 'uniform':
        start_time = time_y - (sample_time_range / 2)
        end_time = time_y + (sample_time_range / 2)
    elif sampling_method == 'prospective':
        start_time = time_y
        end_time = time_y + sample_time_range
    else:
        raise ("sampling_method not recognized")
    return df_x[(df_x.Time > start_time) & (df_x.Time < end_time)]

def plot_model_network(model, dst):
    print(model.summary())
    from keras.utils import plot_model
    plot_model(model, show_layer_names=True, show_shapes=True, to_file=dst)

# ********************************************************************
# ********************* Start of the script **************************
# ********************************************************************
# region HYPER-PARAMETERS
np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

# region Model hyper parameters
BATCH_SIZE = 50
LEARNING_RATE = 0.001
SELECTED_ROWS_IN_SAMPLING = 50   # same as time dimension of model
# endregion Model hyper parameters

# region sampling hyper parameters
SAMPLING_TIME_MINUTES = 5
SAMPLING_PERIOD_SECONDS = 5
SAMPLING_METHOD = 'retrospective'  # can be ['retrospective', 'uniform', 'prospective']
sampling_period = pd.to_timedelta(SAMPLING_PERIOD_SECONDS, unit='s')  # TODO not used yet. implement in sampling
print("sampling_period is: " + str(sampling_period))
sample_time_range = pd.to_timedelta(SAMPLING_TIME_MINUTES, unit='m')
print("sample_time_range is: " + str(sample_time_range))
print("sampling method is: " + SAMPLING_METHOD + ". It can be [retrospective, uniform, prospective]")
# endregion sampling hyper parameters
# endregion


# to make model making reproducible for comparisons
np.random.seed(42)

# region Data Processing
#Data addresses and folders
Fitbit_m_dir = "Data/Raw/Heart/Heart_Rate_Data/Fitbit_m/"
Fitbit_h_dir = "Data/Raw/Heart/Heart_Rate_Data/Fitbit_h/"
fitbit_directories_list = [Fitbit_m_dir, Fitbit_h_dir]

# region scanning the data folders to collect file addresses for input and output
input_files_list = []
output_files_list = []
for Selected_fitbit_dir in fitbit_directories_list:
    for root, dirs, files in os.walk(Selected_fitbit_dir):
        for file in files:
            file_address = os.path.join(Selected_fitbit_dir, file)
            if file.endswith("_x.csv"):
                input_files_list.append(file_address)
            elif file.endswith("_y.csv"):
                output_files_list.append(file_address)
    assert len(input_files_list) == len(output_files_list), "we have ODD number of files in the folder. It should be EVEN to have both inputs and outputs."
print(input_files_list)
print(output_files_list)
# endregion scanning the data folders to collect file addresses for input and output

#region creating the input and output database from the csv files
heart_rate_datapoints_list = []
anxiety_level_datapoints_list = []
for i, input_file_address  in enumerate(input_files_list):
    # region reading data as dataframes.
    output_file_address = output_files_list[i]
    assert input_file_address[:-6] == output_file_address[:-6], "Wrong pair of csv data files selected"

    df_x = pd.read_csv(input_file_address)
    df_y = pd.read_csv(output_file_address)
    # removing the empty rows (rows without Anxiety_Level label) and resetting the index of rows
    df_y = df_y[df_y.Anxiety_Level.isnull() == False].reset_index(drop = True)
    # endregion reading data as dataframes.

    # region Parsing Time BAD
    # df_x['Time'] = (df_x.Time.str.replace(':', '').astype(float).astype(int))
    # df_y['Time'] = (df_y.Time.str.replace(':', '').astype(float).astype(int))
    # df_y.loc[:, 'Time'] *= 100  # to increase Time resolution to seconds in the entire column of Time
    # df_y.loc[df_y.Time % 10000 == 0, 'Time'] -= 4001 # to make 10 o clock, 9:59:59 for example. for future easier subtraction
    # endregion Parsing Time BAD

    # region Parsing Time
    df_x['Time'] = pd.to_datetime(df_x.Time,format= '%H:%M:%S')
    df_y['Time'] = pd.to_datetime(df_y.Time,format= '%H:%M')
    # endregion Parsing Time

    # region example of timedelta by subtracting Timestamp objects
    # print(df_y.head())
    # print(type(df_y.Time[0]))
    # print(type(df_y.Time[3] - df_y.Time[2]))
    # temp_delta = df_y.Time[3] - df_y.Time[2]
    # print(temp_delta)
    # print(temp_delta / 2)
    # endregion example code of timedelta by subtracting Timestamp objects

    # region creating the data points for all the available outputs
    for df_y_row_index, time_y in enumerate(df_y.Time):
        # df_x_sampled = GetTimeRangeSampledDataFrame(df_x, time_y, sample_time_range, SAMPLING_METHOD)
        df_x_sampled = df_x[(df_x.Time < time_y)]
        if len(df_x_sampled) == 0:
            continue

        last_row_index = len(df_x_sampled)
        start_row_index = last_row_index - SELECTED_ROWS_IN_SAMPLING
        if start_row_index < 0:
            continue
        selected_rows_list = [x for x in range(start_row_index,last_row_index)]

        selected_dataFrame = df_x_sampled.iloc[selected_rows_list]
        heart_rate_datapoints_list.append(list(selected_dataFrame['Heart Rate']))
        anxiety_level_datapoints_list.append(list(df_y[df_y.Time == time_y].Anxiety_Level))
    # endregion creating the data points for all the available outputs

print("Data collection ended. Total of {} data points collected" .format(len(anxiety_level_datapoints_list)))
#TODO add data splitter to train and valid. Low Priority
all_data_dict = {'train': {'Data': np.asarray(heart_rate_datapoints_list),
                           'labels': np.asarray(anxiety_level_datapoints_list)}}
print("input heart rate data points shape is: {}" .format(all_data_dict['train']['Data'].shape))
print("output anxiety level data points shape is: {}" .format(all_data_dict['train']['labels'].shape))
#endregion creating the input and output database from the csv files

#region deleting the dataframes
del df_x_sampled
del selected_dataFrame
del df_x
del df_y
#endregion deleting the dataframes

# region showing the data label distribution
train_label_df =  pd.DataFrame(data=all_data_dict.get('train').get('labels'),columns=['Anxiety_Level'])
print(train_label_df.Anxiety_Level.value_counts())
del train_label_df
# endregion showing the data label distribution
# endregion

y_train = all_data_dict.get('train').get('labels')
x_train = all_data_dict.get('train').get('Data')
x_train = np.expand_dims(x_train, 2)

print('=' * 80)
print('Start Training')
print("[INFO] training network...")

# region make and compile model and optimizer
# region Model Architecture
K.clear_session()

# ************** preparation for Keras layers ****************
K.set_image_data_format('channels_last')  # to make sure the input shape is [ hr time, channel]

n_obs, time, depth = x_train.shape
batch_size = 500

input_tensor = Input(shape=(time, depth),name='input_HR')
kernel_init = "he_uniform"
kernel_regul = keras.regularizers.l2(l=1e-4)

C11 = Conv1D(filters=32, kernel_size=5, padding='valid', strides=1,
           activation='linear', kernel_initializer=kernel_init, use_bias=False,
           kernel_regularizer=kernel_regul)(input_tensor)

C12 = Conv1D(filters=32, kernel_size=5, padding='same', strides=1,
           activation='relu', kernel_initializer=kernel_init, use_bias=False,
           kernel_regularizer=kernel_regul)(C11)
C13 = Conv1D(filters=32, kernel_size=5, padding='same', strides=1,
           activation='linear', kernel_initializer=kernel_init, use_bias=False,
           kernel_regularizer=kernel_regul)(C12)
S11 = Add()([C13, C11])
A12 = Activation("relu")(S11)
M11 = MaxPooling1D(pool_size=5, strides=2)(A12)

C21 = Conv1D(filters=32, kernel_size=5, padding='same', strides=1,
           activation='relu', kernel_initializer=kernel_init, use_bias=False,
           kernel_regularizer=kernel_regul)(M11)
C22 = Conv1D(filters=32, kernel_size=5, padding='same', strides=1,
           activation='linear', kernel_initializer=kernel_init, use_bias=False,
           kernel_regularizer=kernel_regul)(C21)
S21 = Add()([C22, M11])
A22 = Activation("relu")(S21)
M21 = MaxPooling1D(pool_size=5, strides=2)(A22)

# C31 = Conv1D(filters=32, kernel_size=5, padding='same', strides=1,
#            activation='relu', kernel_initializer=kernel_init, use_bias=False,
#            kernel_regularizer=kernel_regul)(M21)
# C32 = Conv1D(filters=32, kernel_size=5, padding='same', strides=1,
#            activation='linear', kernel_initializer=kernel_init, use_bias=False,
#            kernel_regularizer=kernel_regul)(C31)
# S31 = Add()([C32, M21])
# A32 = Activation("relu")(S31)
# M31 = MaxPooling1D(pool_size=5, strides=2)(A32)
#
# C41 = Conv1D(filters=32, kernel_size=5, padding='same', strides=1,
#            activation='relu', kernel_initializer=kernel_init, use_bias=False,
#            kernel_regularizer=kernel_regul)(M31)
# C42 = Conv1D(filters=32, kernel_size=5, padding='same', strides=1,
#            activation='linear', kernel_initializer=kernel_init, use_bias=False,
#            kernel_regularizer=kernel_regul)(C41)
# S41 = Add()([C42, M31])
# A42 = Activation("relu")(S41)
# M41 = MaxPooling1D(pool_size=5, strides=2)(A42)
#
# C51 = Conv1D(filters=32, kernel_size=5, padding='same', strides=1,
#            activation='relu', kernel_initializer=kernel_init, use_bias=False,
#            kernel_regularizer=kernel_regul)(M41)
# C52 = Conv1D(filters=32, kernel_size=5, padding='same', strides=1,
#            activation='linear', kernel_initializer=kernel_init, use_bias=False,
#            kernel_regularizer=kernel_regul)(C51)
# S51 = Add()([C52, M41])
# A52 = Activation("relu")(S51)
# M51 = MaxPooling1D(pool_size=5, strides=2)(A52)

F1 = Flatten()(M21)

D1 = Dense(units=32, activation='relu')(F1)
D2 = Dense(units=32, activation='relu')(D1)
output_tensor = Dense(units=1, activation='sigmoid')(D2)

model = Model(inputs=input_tensor, outputs=output_tensor)
print(model.summary())
# plot_model_network(model=model, dst=os.path.join('model_keras.png'))# TODO Fix graphviz stuff
# endregion Model Architecture

# opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
opt = sgd(lr = LEARNING_RATE, decay=0.825)

# opt = sgd(lr = self.meta_data['base_lr'],
#                           decay=self.meta_data['decay'],
#                           momentum = self.meta_data['momentum'],
#                           nesterov=self.meta_data['nesterov_bool'])

model.compile(optimizer=opt # TODO try the model with "categorical_crossentropy" as well
              , loss='mean_squared_error', metrics=['accuracy'])
# , loss=self.meta_data['loss'],
# loss_weights=self.meta_data['loss_weights'],
# metrics=self.meta_data['metric'])
# endregion make and compile model and optimizer

H = model.fit(
    x = x_train, y = y_train[:,0], batch_size=BATCH_SIZE, epochs=100, verbose=2,
    validation_split=0.2, shuffle=True,
    # class_weight=None, #TODO use this
    # sample_weight=None, #TODO use this maybe
    # initial_epoch=0,
    # steps_per_epoch=None,
    # validation_steps=None,
    # validation_freq=1
    )

print("End of Training Epoch\n", "-" * 80)