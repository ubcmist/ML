import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import os.path as osp
import pickle
from sklearn.model_selection import train_test_split
import scipy.io as sio

class DataHandler:
    def __init__(self, meta_data, framework_root):
        print('initialize data handler')
        self.framework_root = framework_root
        self.train_randomized_indices = None
        self.set_meta_data(meta_data)
        # create the datastructure dictionary containing all data with
        self.all_data_not_splited = {'output_df':None}
        self.all_data_dict = {'df_list': [],
                              'train': {'input': None,
                                        'output_df': None,  # pandas dataframe containing all the outputs
                                        # 'output': {'Anxiety_Level': [],
                                        #            'Time': []},
                                        # 'map_to_df_list': [],
                                        'iterator': 0},
                              'valid': {'input': None,
                                        'output_df': None,  # pandas dataframe containing all the outputs
                                        # 'output': {'Anxiety_Level': [],
                                        #            'Time': []},
                                        # 'map_to_df_list': [],
                                        'iterator': 0}}


        print("------ sampling method is: {}. It can be [retrospective, uniform, prospective] ------".format(self.meta_data['sampling_method']))
        print("------ sampling_period is: {} seconds ------" .format(self.meta_data['sampling_period_seconds']))
        if self.meta_data['use_total_time_minutes']:
            print("------ use_total_time_minutes is true. ------")
            total_sampled_time = self.meta_data['total_sampled_time_minutes']
        else:
            print("------ use_total_time_minutes is false. => Will use sampling_selected_rows. ------")
            print("------ sampling_selected_rows is: {} rows".format(self.meta_data['sampling_selected_rows']))
            total_sampled_time = self.meta_data['sampling_selected_rows'] * self.meta_data['sampling_period_seconds'] / 60.0
        print("------ total_sampled_time is: {} minutes".format(total_sampled_time))

    #region metadata handling methods
    def set_meta_data(self, meta_data):
        self.meta_data = meta_data
    def get_meta_data(self):
        return self.meta_data
    def update_meta_data(self, input_meta_data):
        self.meta_data.update(input_meta_data)
    #endregion

    def GetTimeRangeSampledDataFrame(self, df_x, time_y, random_time_shift_value_seconds=0,
                                     total_time_minutes=0,
                                     period_seconds=5,
                                     sampling_method='retrospective',
                                     use_total_time_minutes=False,
                                     sampling_selected_rows=None): #TODO make this possible to use either #of rows or period and total time
        if use_total_time_minutes:
            if total_time_minutes is None:
                raise Exception("Error: use_total_time_minutes is True but total_time_minutes is not provided")
            if total_time_minutes == 0:
                raise Exception("Error: total_time_minutes cannot be 0 minutes")
        else:
            if sampling_selected_rows == None:
                raise Exception("Error: sampling_selected_rows is not provided")
            total_time_minutes = np.ceil(sampling_selected_rows * period_seconds/60)

        total_time_minutes = pd.Timedelta(total_time_minutes, unit='m')
        period_seconds = pd.Timedelta(period_seconds, unit='s')

        # shift end time randomly
        if random_time_shift_value_seconds != 0:
            shift_value = np.round(np.random.uniform(-random_time_shift_value_seconds, random_time_shift_value_seconds))
            time_delta = pd.Timedelta(shift_value, unit='s')
            time_y = time_y + time_delta

        # get start and end time of the sample period
        if sampling_method == 'retrospective':
            start_time = time_y - total_time_minutes
            end_time = time_y
        elif sampling_method == 'uniform':
            start_time = time_y - (total_time_minutes / 2)
            end_time = time_y + (total_time_minutes / 2)
        elif sampling_method == 'prospective':
            start_time = time_y
            end_time = time_y + total_time_minutes
        else:
            raise ("sampling_method not recognized")

        if use_total_time_minutes:
            sampled_time_list = pd.date_range(start_time, end_time, freq=period_seconds)
        else:
            sampled_time_list = pd.date_range(end=end_time, periods=sampling_selected_rows, freq=period_seconds)
        return df_x.loc[sampled_time_list]

    def load_data(self, csv_data_address, train_valid_test=None):
        '''loads data from .csv file, it will create the the data structure for train_valid_test of choice.
        Args:
            csv_data_address: Data in a csv format. inputs and outputs of the model are in a single csv file.
            train_valid_test: a subset of the list ['train','valid', 'test'] to collect data for
        '''
        # region reading data as dataframes.
        def date_time_parser(x):
            return pd.datetime.strptime(x, '%Y-%m-%d-%H:%M:%S')
        df_experiment = pd.read_csv(csv_data_address, parse_dates=[0], squeeze=False, date_parser=date_time_parser)
        # df_experiment = pd.read_csv(csv_data_address)
        # region Parsing Time
        # df_experiment['Time'] = pd.to_datetime(df_experiment.Time)
        # df_experiment = df_experiment.set_index('Time')
        # endregion Parsing Time
        # endregion reading data as dataframes.

        # region creating the data structure without splitting into train_valid
        df_avail_lables = df_experiment[df_experiment.Anxiety_Level.isnull() == False]
        data_points_anxiety_level_list = df_avail_lables.get(['Time','Anxiety_Level']).reset_index(drop=True)

        # making map_to_df_list
        number_of_datapoints = len(data_points_anxiety_level_list)
        df_index = len(self.all_data_dict.get('df_list'))
        map_to_df_list = [df_index for i in range(number_of_datapoints)]
        data_points_anxiety_level_list = pd.concat([data_points_anxiety_level_list,
                                                    pd.DataFrame({'map_to_df_list':map_to_df_list})], axis=1)
        data_points_anxiety_level_list = data_points_anxiety_level_list.set_index('Time')

        df_experiment = df_experiment.set_index('Time')
        self.all_data_dict.get('df_list').append(df_experiment)

        if self.all_data_not_splited.get('output_df') is None:
            self.all_data_not_splited['output_df'] = pd.DataFrame()
        output_df = self.all_data_not_splited['output_df']
        self.all_data_not_splited['output_df'] = pd.concat([output_df,
                                                            data_points_anxiety_level_list])
        print("------ File {} added to the data structure --------".format(osp.basename(csv_data_address)))
        print("------ number of data points added: {}. Total accumulated datapoints: {} --------"
              .format(number_of_datapoints, len(self.all_data_not_splited.get('output_df'))))
        # endregion creating the data structure without splitting into train_valid

    def split_data(self):
        data_points_indices = [x for x in range(len(self.all_data_not_splited.get('output_df')))]
        y_train_indices, y_test_indices = train_test_split(data_points_indices, shuffle=True,
                                                           test_size=0.4, random_state=123)

        data_not_splited = self.all_data_not_splited.get('output_df')
        self.all_data_dict['train']['output_df'] = data_not_splited.iloc[y_train_indices]
        self.all_data_dict['valid']['output_df'] = data_not_splited.iloc[y_test_indices]
        print("------ Data Split Done. Total Train : {}   Total Validation: {} --------"
              .format(len(y_train_indices), len(y_test_indices)))

    #region mat data handling
    def save_mat_file(self,data_dict, file_address):
        ''' saves a dictionary into a .mat file.

        :param data_dict: dictionary containing data to be saved as .mat file
        :param file_address: address to the .mat file to be saved
        '''
        sio.savemat(file_address, data_dict, do_compression=True,
            long_field_names=True, oned_as='column')

    #endregion

    # region pickle data handling
    def save_pickle(self, data_dict, file_address):
        '''saves some data in pickle file. data_dict can be a dictionary format
        Args:
            data_dict: dictionary to save as pickle file
            file_address: address of the file to be saved
        '''
        with open(file_address, 'wb') as handle:
            pickle.dump(data_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_pickle(self, file_address):
        '''loads a pickle file. file outputs a dictionary
         Args:
            file_address: address of the .pickle file to be loaded
        Returns:
            data_dict: dictionary to loaded from pickle file
            '''
        with open(file_address, 'rb') as handle:
            data_dict = pickle.load(handle)
        return data_dict

    # endregion

    def get_file_addresses_in_dir(self, dir, file_format='.csv'):
        '''lists all the files in a directory with a certain file format. output list is sorted alphabetically
        Args:
            dir = address to directory with files of the same format
            file_format : .csv or other types
        Returns:
            a list of addresses to each file inside the dir , sorted alphabetically
            '''
        return [os.path.join(dir, image_filename)
                           for image_filename in sorted(os.listdir(dir))
                           if image_filename.endswith(file_format)]

    def get_dataset_size(self, train_valid_test='train'):
        ''' gets the size of each dataset. train or valid
        Args:
            train_valid_test: determines which dataset cases to count
        Returns:
            cases : number of cases in that dataset
        '''
        # cases = len(self.all_data_dict[train_valid_test]['output']['Anxiety_Level'])
        cases = len(self.all_data_dict[train_valid_test]['output_df'])
        return cases

    def get_batch(self, batch_size, train_valid_test='train', data_traversing='random_weighted'):
        '''creates a batch of data
        Args:
            batch_size : size of cases in each batch. for iterative data_traversing (validation/test) last step of epoch might have nb_cases < batch_size
            train_valid_test: determines which dataset cases to create a batch for
            data_traversing: one of 'iterative' , 'random_weighted', 'random_stratified'
        Returns:
            batch_x : batch input data.
            batch_y : batch output data.
            '''
        df_list = self.all_data_dict.get('df_list')
        input_data = self.all_data_dict.get(train_valid_test).get('input')
        output_df = self.all_data_dict.get(train_valid_test).get('output_df')
        # output_data = self.all_data_dict.get(train_valid_test).get('output').get('Anxiety_Level')
        # output_time_index = self.all_data_dict.get(train_valid_test).get('output').get('Time')
        # map_to_df_list = self.all_data_dict.get(train_valid_test).get('map_to_df_list')
        iter = self.all_data_dict.get(train_valid_test).get('iterator')

        #region getting indices of cases based on data_traversing
        if data_traversing == 'random_stratified':
            # TODO import the features required for this and enable the random stratified version if needed
            # groups_dict = self.all_data_dict.get(train_valid_test).get('label_map')
            # if batch_size % len(groups_dict) != 0:
            #     raise (
            #         "batch size {} is not a multiplication of total groups :{}. stratified batch making won't be stratifies.".format(
            #             batch_size, len(groups_dict)))
            # selected_indices = []
            # batch_size_gp = batch_size // len(groups_dict)
            # for key, item in groups_dict.items():
            #     startIndx = iter % len(item)
            #     list_indices = np.mod(np.arange(startIndx, batch_size_gp+startIndx),len(item), dtype=int)
            #     selected_indices.extend((item[list_indices]))
            #
            # if iter + batch_size_gp >= len(output_data):
            #     iter = 0
            # else:
            #     iter = iter + batch_size_gp
            # self.all_data_dict.get(train_valid_test).update({'iterator':iter})
            raise Exception("------- Stratified batch making NOT IMPLEMENTED YET -------")
        elif data_traversing == 'random_weighted' or data_traversing == 'iterative':
            if data_traversing == 'random_weighted':
                selected_indices = self.train_randomized_indices[iter: iter + batch_size]#TODO make this happen in model training part at the beginning of each epoch
            else:
                selected_indices = (list(range(len(output_df))))[iter: iter + batch_size]
            if iter + batch_size >= len(output_df):
                iter = 0
            else:
                iter = iter + batch_size
            self.all_data_dict.get(train_valid_test).update({'iterator':iter})
        else:
            raise("data traversing {}  method not recognized.".format(data_traversing))
        #endregion

        # region creating the data points for all the selected outputs
        sampling_method = self.meta_data['sampling_method']
        sampling_selected_rows = self.meta_data['sampling_selected_rows']
        sampling_period_seconds = self.meta_data['sampling_period_seconds']
        total_sampled_time_minutes = self.meta_data['total_sampled_time_minutes']
        if train_valid_test == 'train':
            random_time_shift_value_seconds = self.meta_data['random_time_shift_value_seconds']
        else:
            random_time_shift_value_seconds = 0
        use_total_time_minutes = self.meta_data['use_total_time_minutes']

        batch_output = output_df.iloc[selected_indices]
        hr_batch = []
        gsr_batch = []
        for df_y_row_index, time_y in enumerate(batch_output.index):
            df_x = df_list[batch_output.map_to_df_list[df_y_row_index]]
            df_x_sampled = df_x[(df_x.index < time_y)]
            if len(df_x_sampled) == 0:
                continue
            df_x_sampled = self.GetTimeRangeSampledDataFrame(df_x, time_y,
                                                             random_time_shift_value_seconds,
                                                             total_sampled_time_minutes,
                                                             sampling_period_seconds,
                                                             sampling_method,
                                                             use_total_time_minutes,
                                                             sampling_selected_rows)

            df_x_sampled = df_x_sampled.fillna(0) #TODO not sure if this is fine
            hr_batch.append(list(df_x_sampled['Heart Rate']))
            if 'GSR' in df_x_sampled.keys():
                gsr_batch.append(list(df_x_sampled['GSR']))
            else:
                gsr_batch.append([0 for  x in range(len(df_x_sampled['Heart Rate']))])
        # endregion creating the data points for all the selecetd outputs

        return np.asarray(hr_batch), np.asarray(gsr_batch), np.asarray(batch_output.Anxiety_Level)

    def load_label_map_in_dataset_dict(self, train_valid_test):# TODO FIX this  part for stratified if needed
        '''
            creates the label map for the corresponding dataset
        :param train_valid_test:  the correspinding dataset
        '''
        if 'group_by_list' in self.meta_data:
            group_by_list = self.meta_data['group_by_list']
        else:
            group_by_list = self.meta_data['output_labels_list']

        for dataset_name in train_valid_test:
            label_groups, label_map, label_headers = self.create_label_map(
                self.all_data_dict.get(dataset_name).get('labels'), group_by_list)
            self.all_data_dict.get(dataset_name).update({'label_map': label_map, 'label_headers': label_headers,
                                                         'label_groups': label_groups})



if __name__ == "__main__":
    meta_data = {
        'sampling_method': 'retrospective',  # can be ['retrospective', 'uniform', 'prospective']
        'use_total_time_minutes': False,
        'sampling_selected_rows': 50,
        'sampling_period_seconds': 5,
        'total_sampled_time_minutes': 5,
        'random_time_shift_value_seconds': 20}

    dataHanlder = DataHandler(meta_data, './')

    csv_data_address = 'Data/csv_files/VideoGames_Apex_HR_GSR_Hooman/20190322.csv'
    dataHanlder.load_data(csv_data_address)
    hr_batch, gsr_batch, output_batch = dataHanlder.get_batch(batch_size=10, train_valid_test='train', data_traversing='iterative')
    print("batch created")