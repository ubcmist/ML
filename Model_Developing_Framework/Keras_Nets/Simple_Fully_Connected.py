import keras
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Softmax, Add, Flatten, Activation, Dropout
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.callbacks import *
import os.path as osp
from root_model import RootModel
import pandas as pd

class SimpleFcModel(RootModel):
    '''
        a class for a simple fully connected model
    '''
    def __init__(self, external_dict=None):
        super(SimpleFcModel, self).__init__(external_dict)

    def update_net_specific_meta_data(self):
        self.meta_data.update({
            'tf_occupy_full_gpu': True,

            # optimizer
            'optimizer': 'adam',
            'base_lr': 0.001,
            'metric': ['mean_absolute_error'],
            'model_type': 'regression',  # values={classification, regression}
            'loss': 'mean_squared_error',
            'max_epoch': 200,  # maximum number of epochs to train

            # DataHandler: sampling inputs
            'sampling_method': 'retrospective',  # can be ['retrospective', 'uniform', 'prospective']
            'use_total_time_minutes': False,
            'sampling_selected_rows': 50,
            'sampling_period_seconds': 5,
            'total_sampled_time_minutes': 5,
            'random_time_shift_value_seconds': 20,
            'random_time_shift_method': 'uniform',  # values={'uniform', 'normal'} #TODO Normal not implemented yet
            'scale_data': 1,  # values={1: do not rescale, else: rescale all inputs intensities by the given value}
            'subtract_mean': False,  # calculate the mean value of training data, and subtract it from each sample
            'categorical': False,

            # DataHandler:scale label
            'scale_label_value': 0,  # values={0: do not rescale, else: divide all labels by the given value}
            'scale_label': False,

            # DataHandler: batch selection strategies
            'batch_size': 10,
            'data_traversing': 'iterative',  # values={'iterative', 'random_weighted', 'random_stratified'}
        })

    def model_arch(self):
        '''
            Creates a FC model
        '''
        K.set_image_data_format('channels_last')

        if self.meta_data['use_total_time_minutes']:
            # TODO fix this part to use total Time Minutes as well
            total_time_minutes = pd.Timedelta(self.meta_data['total_sampled_time_minutes'], unit='m')
            period_seconds = pd.Timedelta(self.meta_data['sampling_period_seconds'], unit='s')
            # time_d = len(pd.date_range(start_time, start_time+total_time_minutes, freq=period_seconds))
        else:
            time_d = self.meta_data['sampling_selected_rows']

        input_shape = (time_d, 1)

        input_tensor = Input(shape=input_shape, name='input_HR')
        kernel_init = "he_uniform"
        kernel_regul = keras.regularizers.l2()
        KERNEL_SIZE = 5
        USE_BIAS = True
        FILTERS = 32

        x = BatchNormalization()(input_tensor)
        x = Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, padding='valid', strides=1,
                   activation='linear', kernel_initializer=kernel_init, use_bias=USE_BIAS,
                   kernel_regularizer=kernel_regul)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2, strides=2)(x)
        x = Dropout(0.5)(x)

        x = Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, padding='valid', strides=1,
                   activation='relu', kernel_initializer=kernel_init, use_bias=USE_BIAS,
                   kernel_regularizer=kernel_regul)(input_tensor)
        x = Conv1D(filters=FILTERS, kernel_size=KERNEL_SIZE, padding='valid', strides=1,
                   activation='linear', kernel_initializer=kernel_init, use_bias=USE_BIAS,
                   kernel_regularizer=kernel_regul)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = MaxPooling1D(pool_size=2, strides=2)(x)
        x = Dropout(0.5)(x)

        x = Flatten()(x)

        x = Dense(units=5, activation='relu')(x)
        x = Dense(units=5, activation='relu')(x)
        output_tensor = Dense(units=1, activation='sigmoid', name='Anxiety_Level')(x)

        self.net_model = Model(inputs=input_tensor, outputs=output_tensor)

    def getInputPredArgs(self, hr_batch, gsr_batch, output_batch):
        input_args = {'input_HR': np.expand_dims(hr_batch, 2)}
        pred_args = {'Anxiety_Level': output_batch}
        return input_args, pred_args

    def getPredictOnValidation(self):
        # new test method. Simpler, Less Work
        return self.net_model.predict(self.data_handler.all_data_dict['valid']['Data'][:,0], batch_size=300, verbose=1)
    # endregion