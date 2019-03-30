'''
Inspired by:
https://machinelearningmastery.com/resample-interpolate-time-series-data-python/

sample script arguments
python GSR_Downsampling_1S.py
    -i Data\GSR\data_Mar27_game2.csv
    -o Data\GSR_DownSampled_1S\data_Mar27_game2.csv   ( or use default for similar names )
    -t nearest
    -d 2019-03-27
'''

import pandas as pd
from pandas import read_csv, datetime
import argparse
import os
import os.path as osp

parser = argparse.ArgumentParser(
    description='Code to Downsample GSR data to 1S',
    epilog="For more information call me baby")

parser.add_argument('-i', '--input', type=str, default=None, help='address of the csv GSR data requiring down sampling')
parser.add_argument('-o', '--output', type=str, default=None, help='address of the down sampled csv data GSR to be saved')
parser.add_argument('-t', '--type', type=str, default=None, help='re-sampling type: (mean/nearest)')
parser.add_argument('-d', '--date', type=str, default=None, help='Date data was taken. formate: yyyy-mm-dd')
args = vars(parser.parse_args())
Input_GSR_csv = args['input']
output_GSR_csv = args['output']
resampling_type = args['type']
collection_Date = args['date']
# Input_GSR_csv = 'Data/VideoGames_Apex_HR_GSR_Hooman/GSR/VideoGame_Hooman_Mar22_game1.csv'
# output_GSR_csv = 'Data/VideoGames_Apex_HR_GSR_Hooman/GSR_DownSampled_1S/VideoGame_Hooman_Mar22_game1.csv'
# resampling_type = "mean"
# collection_Date = '2019-03-22'


def date_time_parser(x):
	return datetime.strptime(collection_Date+ '-' +x, '%Y-%m-%d-%H:%M:%S:%f')

gsr_original = read_csv(Input_GSR_csv, header=0, parse_dates=[0], index_col=0, squeeze=True,  date_parser=date_time_parser)
if resampling_type == "mean":
    resampled = gsr_original.resample('1S').mean().round()
elif resampling_type == "nearest":
    resampled = gsr_original.resample('1S').nearest().round()

if len(resampled[resampled.isnull() == True]) != 0:
    print("WARNING: there are rows that could not be resampled due to not having enough data:")
    print(resampled[resampled.isnull() == True])

if output_GSR_csv is None:
    output_GSR_csv = osp.join(osp.dirname(Input_GSR_csv) + '_DownSampled_1S', osp.basename(Input_GSR_csv))

if not osp.exists(osp.dirname(output_GSR_csv)):
    os.makedirs(osp.dirname(output_GSR_csv))
resampled.to_csv(output_GSR_csv[:-4] + "_" + resampling_type +'.csv' , header =True, date_format='%Y-%m-%d-%H:%M:%S')