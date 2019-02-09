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
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Softmax, Add, Flatten, Activation# , Dropout
from keras import backend as K
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler, ModelCheckpoint
import os

# to make model making reproducible for comparisons
np.random.seed(42)

# region Data Processing
#Data addresses and folders
Fitbit_m_dir = "Data/Raw/Heart/Heart_Rate_Data/Fitbit_m/"
Fitbit_h_dir = "Data/Raw/Heart/Heart_Rate_Data/Fitbit_h/"
Selected_fitbit_dir = Fitbit_m_dir

# region scanning the data folders to collect file addresses for input and output
input_files_list = []
output_files_list = []
for root, dirs, files in os.walk(Selected_fitbit_dir):
    for file in files:
        if file.endswith("_x.csv"):
            input_files_list += [file]
        elif file.endswith("_y.csv"):
            output_files_list += [file]
if len(input_files_list) != len(output_files_list):
    raise("we have ODD number of files in the folder. It should be EVEN to have both inputs and outputs.")
print(input_files_list)
print(output_files_list)
# endregion scanning the data folders to collect file addresses for input and output

#region creating the input and output database from the csv files
for i in range(len(input_files_list)):
    # region reading data as dataframes.
    input_file_address = os.path.join(Selected_fitbit_dir, input_files_list[i])
    output_file_address = os.path.join(Selected_fitbit_dir, output_files_list[i])
    if (input_file_address[:-6] != output_file_address[:-6]):
        raise("Wrong pair of csv data files selected")

    df_x = pd.read_csv(input_file_address)
    df_y = pd.read_csv(output_file_address)
    df_y = df_y[df_y.Anxiety_Level.isnull() == False] #removing the empty rows (rows without Anxiety_Level label)
    # endregion reading data as dataframes.

    # region Parsing Time BAD
    # df_x['Time'] = (df_x.Time.str.replace(':', '').astype(float).astype(int))
    # df_y['Time'] = (df_y.Time.str.replace(':', '').astype(float).astype(int))
    # df_y.loc[:, 'Time'] *= 100  # to increase Time resolution to seconds in the entire column of Time
    # df_y.loc[df_y.Time % 10000 == 0, 'Time'] -= 4001 # to make 10 o clock, 9:59:59 for example. for future easier subtraction
    # endregion Parsing Time BAD

    # region Parsing Time
    df_x['Time'] = pd.to_datetime(df_x.Time,format= '%H:%M:%S' ).dt.time
    df_y['Time'] = pd.to_datetime(df_y.Time,format= '%H:%M' ).dt.time
    # endregion Parsing Time

#endregion creating the input and output database from the csv files