import pandas as pd
from pandas import read_csv, datetime
import os
import os.path as osp
import argparse


parser = argparse.ArgumentParser(
    description='Code to concatenate the label column to data csv files',
    epilog="For more information call me baby")

parser.add_argument('-i', '--input', type=str, default=None, help='address of the comined data csv requiring the labels')
parser.add_argument('-l', '--labels', type=str, default=None, help='address of the csv file containing all labels with timestamp')
args = vars(parser.parse_args())
# Input_csv = args['input']
# Label_csv = args['labels']
Input_csv = 'C:/Users/Hooman007/Documents/MIST_REPOS/ML/FITBIT_Data_Collection_Interface/Data/Heart_Rate_Data_Old/Subj_H_Work/All_HR.csv'
Label_csv = 'C:/Users/Hooman007/Documents/MIST_REPOS/ML/FITBIT_Data_Collection_Interface/Data/Heart_Rate_Data_Old/Subj_H_Work/All_Anxiety_Level.csv'
input_df = read_csv(Input_csv)
input_df['Time']= pd.to_datetime(input_df.Time)
input_df = input_df.set_index('Time')

label_df = read_csv(Label_csv)
label_df['Time']= pd.to_datetime(label_df.Time)
label_df = label_df.set_index('Time')

if 'Not_Used' in label_df.keys():
    label_df = label_df[label_df.Not_Used != 1]

for idx in range(len(label_df)) :
    time_idx = label_df.index[idx]
    input_df.loc[time_idx].Anxiety_Level = label_df.Anxiety_Level[idx]


output_csv = osp.join(osp.dirname(Input_csv), 'Labeled_' + osp.basename(Input_csv))
input_df.to_csv(output_csv, header =True, date_format='%Y-%m-%d-%H:%M:%S')