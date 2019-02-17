import math
import random
import pickle
import itertools
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from scipy.signal import resample
import matplotlib.pyplot as plt
import pickle
import os


class LoadData():
    def __init__(self):
        self.SELECTED_ROWS_IN_SAMPLING = 10   # same as time dimension of model
        # endregion Model hyper parameters
        # region sampling hyper parameters
        self.SAMPLING_TIME_MINUTES = 5
        self.SAMPLING_PERIOD_SECONDS = 5
        self.SAMPLING_METHOD = 'retrospective'  # can be ['retrospective', 'uniform', 'prospective']
        self.sampling_period = pd.to_timedelta(self.SAMPLING_PERIOD_SECONDS, unit='s')  # TODO not used yet. implement in sampling
        print("sampling_period is: " + str(self.sampling_period))
        self.sample_time_range = pd.to_timedelta(self.SAMPLING_TIME_MINUTES, unit='m')
        print("sample_time_range is: " + str(self.sample_time_range))
        print("sampling method is: " + self.SAMPLING_METHOD + ". It can be [retrospective, uniform, prospective]")            
        # endregion sampling hyper parameters
        # endregion

    def GetTimeRangeSampledDataFrame(self, df_x, time_y, sample_time_range, sampling_method = 'retrospective'):
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
            raise Exception("sampling_method not recognized")
            return df_x[(df_x.Time > start_time) & (df_x.Time < end_time)]

    def CheckFilesSynced(self, output_files_list, input_files_list):
        helper_variable = False
        for i, input_file_address  in enumerate(input_files_list):
            for s, output_file_address in enumerate(output_files_list):
                if (input_file_address[:-6] == output_file_address[:-6]):
                    helper_variable = True
            if (helper_variable == False):
                return False
            else: helper_variable = False
        return True

    def RegionScanDataFolders(self, fitbit_directories_list):
        
        input_files_list = []
        output_files_list = []
        List =[input_files_list,output_files_list]
        for Selected_fitbit_dir in fitbit_directories_list:
            for root, dirs, files in os.walk(Selected_fitbit_dir):
                for file in files:
                    file_address = os.path.join(Selected_fitbit_dir, file)
                    if file.endswith("_x.csv"):
                        input_files_list.append(file_address)
                    elif file.endswith("_y.csv"):
                        output_files_list.append(file_address)
            if len(input_files_list) != len(output_files_list):
                raise Exception("we have ODD number of files in the folder. It should be EVEN to have both inputs and outputs.")
        print(input_files_list)
        print(output_files_list)
        return List
   
    def RegionCreateDataFolders(self, input_files_list, output_files_list):
        
        heart_rate_datapoints_list = []
        anxiety_level_datapoints_list = []
        List = [heart_rate_datapoints_list, anxiety_level_datapoints_list]
        if(self.CheckFilesSynced(output_files_list, input_files_list) == False):
            raise Exception("Data Files are not correctly matched")
        for i, input_file_address  in enumerate(input_files_list):
            # region reading data as dataframes.
            output_file_address = output_files_list[i]
   

            df_x = pd.read_csv(input_file_address)
            df_y = pd.read_csv(output_file_address)
            # removing the empty rows (rows without Anxiety_Level label) and resetting the index of rows
            df_y = df_y[df_y.Anxiety_Level.isnull() == False].reset_index(drop = True)
            # endregion reading data as dataframes.




            # region Parsing Time
            df_x['Time'] = pd.to_datetime(df_x.Time,format= '%H:%M:%S')
            df_y['Time'] = pd.to_datetime(df_y.Time,format= '%H:%M')
            # endregion Parsing Time



            # region creating the data points for all the available outputs
            for df_y_row_index, time_y in enumerate(df_y.Time):
                # df_x_sampled = GetTimeRangeSampledDataFrame(df_x, time_y, sample_time_range, SAMPLING_METHOD)
                df_x_sampled = df_x[(df_x.Time < time_y)]
                if len(df_x_sampled) == 0:
                    continue

                last_row_index = len(df_x_sampled)
                start_row_index = last_row_index - self.SELECTED_ROWS_IN_SAMPLING
                if start_row_index < 0:
                    continue
                selected_rows_list = [x for x in range(start_row_index,last_row_index)]

                selected_dataFrame = df_x_sampled.iloc[selected_rows_list]
                heart_rate_datapoints_list.append(list(selected_dataFrame['Heart Rate']))
                anxiety_level_datapoints_list.append(list(df_y[df_y.Time == time_y].Anxiety_Level))
            # endregion creating the data points for all the available outputs
        return List


    

    def loadData(self):
      
        # to make model making reproducible for comparisons
        np.random.seed(42)

        # region Data Processing
        #Data addresses and folders
        Fitbit_m_dir = "../Fitbit_m/"
        Fitbit_h_dir = "../Fitbit_h/"
        fitbit_directories_list = [Fitbit_m_dir, Fitbit_h_dir]

        # region scanning the data folders to collect file addresses for input and output
        List = self.RegionScanDataFolders(fitbit_directories_list)
        input_files_list = List[0]
        output_files_list = List[1]
        
        #region creating the input and output database from the csv files
        List = self.RegionCreateDataFolders(input_files_list, output_files_list)
        heart_rate_datapoints_list = List[0]
        anxiety_level_datapoints_list = List[1]
        
        
        print("Data collection ended. Total of {} data points collected" , format(len(anxiety_level_datapoints_list)))
        #TODO add data splitter to train and valid. Low Priority
        all_data_dict = {'train': {'Data': np.asarray(heart_rate_datapoints_list),
                           'labels': np.asarray(anxiety_level_datapoints_list)}}
        print("input heart rate data points shape is: {}", format(all_data_dict['train']['Data'].shape))
        print("output anxiety level data points shape is: {}", format(all_data_dict['train']['labels'].shape))
        #print(all_data_dict)
        #endregion creating the input and output database from the csv files

        fig = plt.figure()
        plt.plot(all_data_dict['train']['Data'],all_data_dict['train']['labels'],'bo', markersize=2)
        fig.suptitle('Anxiety Level vs Heart Rate ', fontsize=20)
        plt.xlabel('Heart Rate(bpm)', fontsize=18)
        plt.ylabel('Anxiety Level(1-10)', fontsize=16)
        fig.savefig('../figures/scatterPlot.pdf')
        return all_data_dict
        # endregion






