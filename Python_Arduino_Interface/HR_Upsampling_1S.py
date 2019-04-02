'''
Inspired by:
https://machinelearningmastery.com/resample-interpolate-time-series-data-python/

sample script arguments
python HT_Upsampling_1S.py
    -i Data\HR\data_Mar27_game2.csv
    -o Data\HR_UpSampled_1S\data_Mar27_game2.csv   ( or use default for similar names )
    -d 2019-03-27
'''

import pandas as pd
from pandas import read_csv, datetime
import argparse
import os
import os.path as osp

parser = argparse.ArgumentParser(
    description='Code to Up Sample HR data to 1S',
    epilog="For more information call me baby")

parser.add_argument('-i', '--input', type=str, default=None, help='address of the csv HR data requiring up sampling')
parser.add_argument('-o', '--output', type=str, default=None, help='address of the up sampled csv data HR to be saved. If not given, the same name of input data is used with parent folder + _UpSampled_1S')
parser.add_argument('-t', '--type', type=str, default='linear', help='up-sampling interpolation type: (linear/spline). Linear is best option')
parser.add_argument('-d', '--date', type=str, default=None, help='Date data was taken. format: yyyy-mm-dd')
args = vars(parser.parse_args())
Input_GSR_csv = args['input']
output_GSR_csv = args['output']
resampling_type = args['type']
collection_Date = args['date']
# Input_GSR_csv = 'Data/VideoGames_Apex_HR_GSR_Hooman/HR/20190322_x.csv'
# output_GSR_csv = 'Data/VideoGames_Apex_HR_GSR_Hooman/HR_UpSampled_1S/20190322_x.csv'
# resampling_type = "linear"
# collection_Date = '2019-03-22'


def date_time_parser(x):
	return datetime.strptime(collection_Date+ '-' +x, '%Y-%m-%d-%H:%M:%S')

HR_original = read_csv(Input_GSR_csv, header=0, parse_dates=[0], index_col=0, squeeze=True,  date_parser=date_time_parser)
resampled = HR_original.resample('1S')
if resampling_type == "linear":
    interpolated = resampled.interpolate(method='linear')
    interpolated = interpolated.round()
elif resampling_type == "spline":
    interpolated = resampled.interpolate(method='spline', order=2)

if len(interpolated[interpolated.isnull() == True]) != 0:
    print("WARNING: there are rows that could not be resampled due to not having enough data:")
    print(interpolated[interpolated.isnull() == True])

if output_GSR_csv is None:
    output_GSR_csv = osp.join(osp.dirname(Input_GSR_csv) + '_UpSampled_1S', osp.basename(Input_GSR_csv))

if not osp.exists(osp.dirname(output_GSR_csv)):
    os.makedirs(osp.dirname(output_GSR_csv))
interpolated.to_csv(output_GSR_csv[:-4] + "_" + resampling_type +'.csv' , header =True, date_format='%Y-%m-%d-%H:%M:%S')