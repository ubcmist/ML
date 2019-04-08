import pandas as pd
from pandas import read_csv, datetime
import os
import os.path as osp


Input_csv = 'Data/VideoGames_Apex_HR_GSR_Hooman/All_Data.csv'
Label_csv = 'Data/VideoGames_Apex_HR_GSR_Hooman/Anxiety_Level_Labels.csv'

input_df = read_csv(Input_csv)
input_df['Time']= pd.to_datetime(input_df.Time)
input_df = input_df.set_index('Time')

label_df = read_csv(Label_csv)
label_df['Time']= pd.to_datetime(label_df.Time)
label_df = label_df.set_index('Time')

for idx in range(len(label_df)) :
    time_idx = label_df.index[idx]
    input_df.loc[time_idx].Anxiety_Level = label_df.Anxiety_Level[idx]


output_csv = osp.join(osp.dirname(Input_csv), 'Labeled_' + osp.basename(Input_csv))
input_df.to_csv(output_csv, header =True, date_format='%Y-%m-%d-%H:%M:%S')