'''
Inspired by:
https://machinelearningmastery.com/resample-interpolate-time-series-data-python/
'''

import pandas as pd
from pandas import read_csv
import argparse
import os
import os.path as osp

parser = argparse.ArgumentParser(
    description='Code to Downsample GSR data to 1S',
    epilog="For more information call me baby")

parser.add_argument('-i', '--input', type=str, default=None, help='address of the csv GSR data requiring down sampling')
parser.add_argument('-o', '--output', type=str, default=None, help='address of the down sampled csv data GSR to be saved')
parser.add_argument('-t', '--type', type=str, default=None, help='re-sampling type: (mean/nearest)')
args = vars(parser.parse_args())
Input_GSR_csv = args['input']
output_GSR_csv = args['output']
resampling_type = args['type']
# Input_GSR_csv = 'VideoGames_Apex_HR_GSR_Hooman/GSR/VideoGame_Hooman_Mar23-24_Night24.csv'
# output_GSR_csv = 'VideoGames_Apex_HR_GSR_Hooman/GSR_DownSampled/VideoGame_Hooman_Mar23-24_Night24_Downsampled.csv'

gsr_original = read_csv(Input_GSR_csv,header=0)
gsr_original['Time'] = pd.to_datetime(gsr_original.Time,format='%H:%M:%S:%f')
if resampling_type == "mean":
    resampled = gsr_original.resample('1S', on='Time').mean().round()
elif resampling_type == "nearest":
    time_series = gsr_original.set_index('Time')
    resampled = time_series.resample('1S').nearest().round()

if len(resampled[resampled.GSR.isnull() == True]) != 0:
    print("WARNING: there are rows that could not be resampled due to not having enough data:")
    print(resampled[resampled.GSR.isnull() == True])

if not osp.exists(osp.dirname(output_GSR_csv)):
    os.makedirs(osp.dirname(output_GSR_csv))
resampled.to_csv(output_GSR_csv[:-4] + "_" + resampling_type +'.csv' , index = True,  date_format='%H:%M:%S')