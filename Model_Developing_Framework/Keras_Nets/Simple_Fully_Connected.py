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
            'base_lr': 0.0001,
            'lr_decay': 0.825,
            'lr_decay_step': 2,
            'metric': ['mean_absolute_error'],
            'model_type': 'regression',  # values={classification, regression}
            'loss': 'mean_squared_error',
            'max_epoch': 50,  # maximum number of epochs to train

            # DataHandler: sampling inputs
            'sampling_method': 'retrospective',  # can be ['retrospective', 'uniform', 'prospective']
            'use_total_time_minutes': True,
            'sampling_selected_rows': 150,
            'sampling_period_seconds': 10,
            'total_sampled_time_minutes': 5,
            'random_time_shift_value_seconds': 0, #TODO add this for higher accuracy later
            'random_time_shift_method': 'uniform',  # values={'uniform', 'normal'} #TODO Normal not implemented yet
            'scale_data': 1,  # values={1: do not rescale, else: rescale all inputs intensities by the given value}
            'subtract_mean': False,  # calculate the mean value of training data, and subtract it from each sample
            'categorical': False,

            # DataHandler:scale label
            'scale_label_value': 0,  # values={0: do not rescale, else: divide all labels by the given value}
            'scale_label': False,

            # DataHandler: batch selection strategies
            'batch_size': 20,
            'data_traversing': 'iterative',  # values={'iterative', 'random_weighted', 'random_stratified'}
        })

    def model_arch(self):
        '''
            Creates a FC model
        '''
        K.set_image_data_format('channels_last')

        if self.meta_data['use_total_time_minutes']:
            # TODO double check here
            time_d = 1 + int(self.meta_data['total_sampled_time_minutes'] * 60 /
                             self.meta_data['sampling_period_seconds'])
        else:
            time_d = self.meta_data['sampling_selected_rows']

        input_shape = (time_d,)

        input_tensor = Input(shape=input_shape, name='input_HR')

        x = Dense(units=time_d, activation='relu')(input_tensor)
        x = Dense(units=180, activation='relu')(x)
        x = Dense(units=140, activation='relu')(x)
        x = Dense(units=100, activation='relu')(x)
        x = Dense(units=50, activation='relu')(x)
        # for classification
        output_tensor = Dense(units=6, activation='relu', name='Anxiety_Level')(x)
        # for regression
        # output_tensor = Dense(units=1, activation='sigmoid', name='Anxiety_Value')(x)

        self.net_model = Model(inputs=input_tensor, outputs=output_tensor)
        print(self.net_model.summary())

    def getInputPredArgs(self, hr_batch, gsr_batch, output_batch):
        # input_args = {'input_HR': np.expand_dims(hr_batch, 2)}
        input_args = {'input_HR': hr_batch}

        # for regression
        # pred_args = {'Anxiety_Value': (output_batch - 1)/6}
        # for classification
        pred_args = {'Anxiety_Level': keras.utils.to_categorical(y=(output_batch - 1),num_classes=6)}

        return input_args, pred_args

    # endregion

class SimpleFcModel_Softmax(RootModel):
    '''
        a class for a simple fully connected model
    '''
    def __init__(self, external_dict=None):
        super(SimpleFcModel_Softmax, self).__init__(external_dict)

    def update_net_specific_meta_data(self):
        self.meta_data.update({
            'tf_occupy_full_gpu': True,

            # optimizer
            'optimizer': 'adam',
            'base_lr': 0.0001,
            'lr_decay': 0.825,
            'lr_decay_step': 10,
            'metric': ['categorical_accuracy'],
            'model_type': 'classification',  # values={classification, regression}
            'loss': 'categorical_crossentropy',
            'max_epoch': 50,  # maximum number of epochs to train

            # DataHandler: sampling inputs
            'sampling_method': 'retrospective',  # can be ['retrospective', 'uniform', 'prospective']
            'use_total_time_minutes': True,
            'sampling_selected_rows': 150,
            'sampling_period_seconds': 10,
            'total_sampled_time_minutes': 5,
            'random_time_shift_value_seconds': 0, #TODO add this for higher accuracy later
            'random_time_shift_method': 'uniform',  # values={'uniform', 'normal'} #TODO Normal not implemented yet
            'scale_data': 1,  # values={1: do not rescale, else: rescale all inputs intensities by the given value}
            'subtract_mean': False,  # calculate the mean value of training data, and subtract it from each sample
            'categorical': False,

            # DataHandler:scale label
            'scale_label_value': 0,  # values={0: do not rescale, else: divide all labels by the given value}
            'scale_label': False,

            # DataHandler: batch selection strategies
            'batch_size': 20,
            'data_traversing': 'iterative',  # values={'iterative', 'random_weighted', 'random_stratified'}
        })

    def model_arch(self):
        '''
            Creates a FC model
        '''
        K.set_image_data_format('channels_last')

        if self.meta_data['use_total_time_minutes']:
            # TODO double check here
            time_d = 1 + int(self.meta_data['total_sampled_time_minutes'] * 60 /
                             self.meta_data['sampling_period_seconds'])
        else:
            time_d = self.meta_data['sampling_selected_rows']

        input_shape = (time_d,)

        input_tensor = Input(shape=input_shape, name='input_HR')

        x = Dense(units=time_d, activation='relu')(input_tensor)
        x = Dense(units=180, activation='relu')(x)
        # x = Dropout(0.5)(x)
        x = Dense(units=140, activation='relu')(x)
        x = Dense(units=100, activation='relu')(x)
        # x = Dropout(0.5)(x)
        x = Dense(units=50, activation='relu')(x)
        # for classification
        output_tensor = Dense(units=6, activation='softmax', name='Anxiety_Level')(x)

        self.net_model = Model(inputs=input_tensor, outputs=output_tensor)
        print(self.net_model.summary())

    def getInputPredArgs(self, hr_batch, gsr_batch, output_batch):
        # input_args = {'input_HR': hr_batch}
        input_args = {'input_HR': gsr_batch}

        # for classification
        pred_args = {'Anxiety_Level': keras.utils.to_categorical(y=(output_batch - 1),num_classes=6)}

        return input_args, pred_args

    def plot_Statistics_History(self, model_name="", accName='categorical_accuracy', H=None, dst_dir=None):
        import matplotlib.pyplot as plt
        N = np.arange(0, len(H["loss"]))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H["loss"], label="train_loss")
        plt.plot(N, H["val_loss"], label="valid_loss")
        plt.title("Loss plot of model {}".format(model_name))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()
        # save the figure
        plt.savefig(os.path.join(dst_dir, 'LossPlot'))
        plt.close()

        plt.figure()
        plt.plot(N, H[accName], label="train_mean_abs_error")
        plt.plot(N, H['val_' + accName], label="valid_mean_abs_error")
        plt.title("Mean Absolute Error plot of model {}".format(model_name))
        plt.xlabel("Epoch #")
        plt.ylabel("Mean Absolute Error")
        plt.legend()
        # save the figure
        plt.savefig(os.path.join(dst_dir, 'MeanAbsoluteErrorPlot'))
        plt.close()

    # endregion

class SimpleFcModel_Softmax_HR_GSR(RootModel):
    '''
        a class for a simple fully connected model
    '''
    def __init__(self, external_dict=None):
        super(SimpleFcModel_Softmax_HR_GSR, self).__init__(external_dict)

    def update_net_specific_meta_data(self):
        self.meta_data.update({
            'tf_occupy_full_gpu': True,

            # optimizer
            'optimizer': 'adam',
            'base_lr': 0.0001,
            'lr_decay': 0.825,
            'lr_decay_step': 10,
            'metric': ['categorical_accuracy'],
            'model_type': 'classification',  # values={classification, regression}
            'loss': 'categorical_crossentropy',
            'max_epoch': 50,  # maximum number of epochs to train

            # DataHandler: sampling inputs
            'sampling_method': 'retrospective',  # can be ['retrospective', 'uniform', 'prospective']
            'use_total_time_minutes': True,
            'sampling_selected_rows': 150,
            'sampling_period_seconds': 10,
            'total_sampled_time_minutes': 5,
            'random_time_shift_value_seconds': 0, #TODO add this for higher accuracy later
            'random_time_shift_method': 'uniform',  # values={'uniform', 'normal'} #TODO Normal not implemented yet
            'scale_data': 1,  # values={1: do not rescale, else: rescale all inputs intensities by the given value}
            'subtract_mean': False,  # calculate the mean value of training data, and subtract it from each sample
            'categorical': False,

            # DataHandler:scale label
            'scale_label_value': 0,  # values={0: do not rescale, else: divide all labels by the given value}
            'scale_label': False,

            # DataHandler: batch selection strategies
            'batch_size': 20,
            'data_traversing': 'iterative',  # values={'iterative', 'random_weighted', 'random_stratified'}
        })

    def model_arch(self):
        '''
            Creates a FC model
        '''
        K.set_image_data_format('channels_last')

        if self.meta_data['use_total_time_minutes']:
            # TODO double check here
            time_d = 1 + int(self.meta_data['total_sampled_time_minutes'] * 60 /
                             self.meta_data['sampling_period_seconds'])
        else:
            time_d = self.meta_data['sampling_selected_rows']

        input_shape = (time_d,)

        input_tensor_HR = Input(shape=input_shape, name='input_HR')
        x_HR = Dense(units=50, activation='relu')(input_tensor_HR)
        x_HR = Dense(units=30, activation='relu')(x_HR)
        input_tensor_GSR = Input(shape=input_shape, name='input_GSR')
        x_GSR = Dense(units=50, activation='relu')(input_tensor_GSR)
        x_GSR = Dense(units=30, activation='relu')(x_GSR)
        x = keras.layers.concatenate([x_HR, x_GSR])

        x = Dense(units=180, activation='relu')(x)
        # x = Dropout(0.5)(x)
        x = Dense(units=140, activation='relu')(x)
        x = Dense(units=100, activation='relu')(x)
        # x = Dropout(0.5)(x)
        x = Dense(units=50, activation='relu')(x)
        # for classification
        output_tensor = Dense(units=6, activation='softmax', name='Anxiety_Level')(x)

        self.net_model = Model(inputs=[input_tensor_HR, input_tensor_GSR], outputs=output_tensor)
        print(self.net_model.summary())

    def getInputPredArgs(self, hr_batch, gsr_batch, output_batch):
        input_args = {'input_HR': hr_batch,
                      'input_GSR': gsr_batch}
        # for classification
        pred_args = {'Anxiety_Level': keras.utils.to_categorical(y=(output_batch - 1),num_classes=6)}

        return input_args, pred_args

    def plot_Statistics_History(self, model_name="", accName='categorical_accuracy', H=None, dst_dir=None):
        import matplotlib.pyplot as plt
        N = np.arange(0, len(H["loss"]))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H["loss"], label="train_loss")
        plt.plot(N, H["val_loss"], label="valid_loss")
        plt.title("Loss plot of model {}".format(model_name))
        plt.xlabel("Epoch #")
        plt.ylabel("Loss")
        plt.legend()
        # save the figure
        plt.savefig(os.path.join(dst_dir, 'LossPlot'))
        plt.close()

        plt.figure()
        plt.plot(N, H[accName], label="train_Acc")
        plt.plot(N, H['val_' + accName], label="valid_Acc")
        plt.title("Acc plot of model {}".format(model_name))
        plt.xlabel("Epoch #")
        plt.ylabel("Acc")
        plt.legend()
        # save the figure
        plt.savefig(os.path.join(dst_dir, 'AccPlot'))
        plt.close()

    # endregion