from dataHandler import DataHandler
import os
import os.path as osp
import tensorflow as tf
from keras import backend as K
from keras.optimizers import adam, RMSprop, sgd

class root_model():
    def __init__(self, external_meta_data=None):
        self.external_meta_data = external_meta_data
        self.meta_data = None
        self.get_meta_data()
        self.set_directories()
        self.data_handler = DataHandler(self.meta_data, self.framework_root)

    def set_directories(self):
        self.framework_root = osp.dirname(osp.dirname(osp.realpath(__file__)))
        self.model_dir = osp.join(self.framework_root,'trained',self.__class__.__name__)
        self.snapshot_dir = osp.join(self.model_dir, 'snapshots')
        self.write_filename = osp.join(self.snapshot_dir, self.meta_data['model_name'])
        if not osp.exists(self.write_filename):
            os.makedirs(self.write_filename)

    def get_meta_data(self):
        if self.meta_data is None:
            self.meta_data = self.init_meta_data()
            if self.external_meta_data is not None:
                self.meta_data.update(self.external_meta_data)
            self.meta_data.update({'model_output_fld': self.write_filename})
            self.update_net_specific_meta_data()
        return self.meta_data

    def set_solver(self):
        # to occupy the GPU ram ONLY as much as required, not completely
        if self.meta_data['tf_occupy_full_gpu']:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            K.set_session(sess)

        if self.meta_data['optimizer'] == 'adam':
            opt = adam(lr=self.meta_data['base_lr'])
        elif self.meta_data['optimizer'] == 'RMSprop':
            opt = RMSprop(lr=self.meta_data['base_lr'])
        elif self.meta_data['optimizer'] == 'sgd':
            opt = sgd(lr=self.meta_data['base_lr'],
                      decay=self.meta_data['sgd_decay'],
                      momentum=self.meta_data['sgd_momentum'],
                      nesterov=self.meta_data['sgd_nesterov_bool'])
        else:
            raise Exception("Error, invalid optimizer selected")

        self.net_model.compile(optimizer=opt,
                      loss=self.meta_data['loss'],
                      metrics=self.meta_data['metric'])

    def init_meta_data(self):
        '''
            creates a dictionary for the framework configuration. Can be overwritten by each model
        '''
        m = {
            'model_name': '...',

            'tf_occupy_full_gpu': False,

            # optimizer
            'loss': 'sgd',
            'optimizer': 'adam',
            'base_lr': 0.001,
            'metric': ['mean_absolute_error'],
            'model_type': 'regression',  # values={classification, regression}
            'max_epoch': 200,  # maximum number of epochs to train
            # configs for sgd optimizer
            'sgd_decay': 1e-6,
            'sgd_momentum': 0.9,
            'sgd_nesterov_bool': True,

            # DataHandler: sampling inputs
            'sampling_method': 'retrospective',  # can be ['retrospective', 'uniform', 'prospective']
            'sampling_selected_rows': 50,
            'sampling_period_seconds': 5,
            'total_sampled_time_minutes': 5,
            'random_time_shift_value_seconds': 20,
            'random_time_shift_method': 'uniform',  # values={'uniform', 'normal'} #TODO Normal not implemented yet

            'scale_data': 1,  # values={1: do not rescale, else: rescale all inputs intensities by the given value}
            'subtract_mean': False,  # calculate the mean value of training data, and subtract it from each sample
            'categorical': False,
            'num_classes': 4,  # if categorical is True, this represents the number of classes; else, ignore this value

            # DataHandler:scale label
            'scale_label_value': 0,  # values={0: do not rescale, else: divide all labels by the given value}
            'scale_label': True,

            # DataHandler: batch selection strategies
            'batch_size': 10,
            'data_traversing': 'iterative',  # values={'iterative', 'random_weighted', 'random_stratified'}
        }
        return m

    # @abstractmethod
    def update_net_specific_meta_data(self):
        pass