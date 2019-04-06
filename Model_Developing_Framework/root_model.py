from dataHandler import DataHandler
import os
import os.path as osp
import tensorflow as tf
from keras import backend as K
from keras.optimizers import adam, RMSprop, sgd
import pandas as pd
import numpy as np

class RootModel:
    def __init__(self, external_meta_data=None):
        self.external_meta_data = external_meta_data
        self.set_directories()
        self.meta_data = None
        self.get_meta_data()
        self.data_handler = DataHandler(self.meta_data, self.framework_root)

    def set_directories(self):
        self.framework_root = osp.dirname(osp.dirname(osp.realpath(__file__)))
        self.model_dir = osp.join(self.framework_root,'trained',self.__class__.__name__)
        self.snapshot_dir = osp.join(self.model_dir, 'snapshots')
        if not osp.exists(self.snapshot_dir):
            os.makedirs(self.snapshot_dir)

    def get_meta_data(self):
        if self.meta_data is None:
            self.meta_data = self.init_meta_data()
            if self.external_meta_data is not None:
                self.meta_data.update(self.external_meta_data)

            self.write_filename = osp.join(self.snapshot_dir, self.meta_data['model_name'])
            if not osp.exists(self.write_filename):
                os.makedirs(self.write_filename)
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
            # TODO use the configurations bellow for each network class
            'tf_occupy_full_gpu': False,

            # optimizer
            # 'loss': 'mean_squared_error',
            # 'optimizer': 'adam',
            # 'base_lr': 0.001,
            # 'metric': ['mean_absolute_error'],
            # 'model_type': 'regression',  # values={classification, regression}
            # 'max_epoch': 200,  # maximum number of epochs to train

            # configs for sgd optimizer
            # 'sgd_decay': 1e-6,
            # 'sgd_momentum': 0.9,
            # 'sgd_nesterov_bool': True,

            # DataHandler: sampling inputs
            # 'sampling_method': 'retrospective',  # can be ['retrospective', 'uniform', 'prospective']
            # 'use_total_time_minutes': False,
            # 'sampling_selected_rows': 50,
            # 'sampling_period_seconds': 5,
            # 'total_sampled_time_minutes': 5,
            # 'random_time_shift_value_seconds': 20,
            # 'random_time_shift_method': 'uniform',  # values={'uniform', 'normal'} #TODO Normal not implemented yet

            # 'scale_data': 1,  # values={1: do not rescale, else: rescale all inputs intensities by the given value}
            # 'subtract_mean': False,  # calculate the mean value of training data, and subtract it from each sample
            # 'categorical': False,
            # 'num_classes': 4,  # if categorical is True, this represents the number of classes; else, ignore this value

            # DataHandler:scale label
            # 'scale_label_value': 0,  # values={0: do not rescale, else: divide all labels by the given value}
            # 'scale_label': False,

            # DataHandler: batch selection strategies
            # 'batch_size': 10,
            # 'data_traversing': 'iterative',  # values={'iterative', 'random_weighted', 'random_stratified'}
        }
        return m

    def set_data(self, data_address_csv, train_valid_test=None):
        experiments_address_df = pd.read_csv(data_address_csv)
        for indx in range(len(experiments_address_df)):
            self.data_handler.load_data(experiments_address_df.file_address[indx],
                                        train_valid_test=experiments_address_df.train_valid_test[indx])

    def train_validate_Optimized(self, base_model=None):
        print('=' * 80)
        print('Initialize network...')
        np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

        self.nb_batches_train = np.ceil(self.data_handler.get_dataset_size('train')/self.meta_data['batch_size'])

        # if base_model is not None:
        #     print('loading weights from file: ', base_model)
        #     # with CustomObjectScope({'mean_Diff': self.mean_Diff}):
        #     self.net_model = load_model(base_model)


        # To get the weight of the classes
        # if self.meta_data['data_traversing'] == 'randomizedBatch_iterative':
        #     # self.labels = ['quality_label', 'quality_label_t2', 'view_label']
        #     labels = ['view_label']
        #     self.data_handler.compute_class_weight(labels)  # to create the weight
        #     classView_weight = self.data_handler.get_class_weight_dict('view_label')
        #     # TODO make the callback for creating the randomized list for indices

        # region Callbacks
        # monitor = 'val_' + self.meta_data['metric'][0]
        # LR_Scheduler = LearningRateScheduler(self.poly_decay)
        # self.time_recorded = time.time()
        # # class TimeEpochCallback(keras.callbacks.Callback):
        # #     def on_epoch_end(self, epoch, logs=None):
        # #         print("time spend on epoch : {}".format(epoch) + "= {}".format(time.time() - self.time_recorded))
        # #     def on_epoch_begin(self, epoch, logs=None):
        # #         self.time_recorded = time.time()
        # # timeEpoch = TimeEpochCallback()
        # csvLogger = CSVLogger(os.path.join(self.write_filename, 'TrainingData.csv') )
        # modelSaveCallback = ModelCheckpoint(os.path.join(self.write_filename, 'model.hdf5'),
        #                                     monitor=monitor, verbose=1, save_best_only=True)
        # tensorBoardCallback =TensorBoard(log_dir=os.path.join(self.write_filename,'tensorboard_logs') ,
        #                                   histogram_freq=1, batch_size=self.meta_data['batch_size'],
        #                                   write_graph=True, write_grads=True, write_images=True,
        #                                   embeddings_freq=0, embeddings_metadata=None)# TODO fix embedding
        # earlyStopCallback = EarlyStopping(monitor=monitor,
        #                                   min_delta=1, patience=5, verbose=1)

        # callbacks = [LR_Scheduler, csvLogger, modelSaveCallback, tensorBoardCallback, earlyStopCallback,
        #              TerminateOnNaN(), BaseLogger(stateful_metrics=['binary_accuracy'])]#, ProgbarLogger(count_mode='samples')]
        # # endregion

        # region train the network
        print('=' * 80)
        print('Start Training')
        print("[INFO] training network...")

        H = self.net_model.fit_generator(
            generator=self.Data_Generator(train_valid='train'),
            steps_per_epoch=self.nb_batches_train,
            # validation_data=self.Data_Generator(train_valid='valid'),
            # validation_steps=self.nb_batches_valid,
            validation_data=None,
            # validation_steps=self.data_handler.get_dataset_size(),
            # epochs=self.meta_data['max_epoch'],
            epochs=self.meta_data['max_epoch'],
            # callbacks=callbacks,
            verbose=1,
            class_weight=None,
            max_queue_size=10,
            workers=1,
            use_multiprocessing=False,
            shuffle=True,
            initial_epoch=0
            )

        print("End of Training Epoch\n", "-" * 80)
        # endregion

        # region plotting accuracy and loss
        # *************************************************
        # ************ Plotting accuracy and loss ***********
        # import utilities.plotting_Utils as utilities
        # utilities.plot_Statistics_History(model_name = self.meta_data['model_name'],
        #                                   H = H.history, accName =self.meta_data['metric'][0] ,
        #                                   dst_dir= self.write_filename)
        # endregion

        K.clear_session()

    def Data_Generator(self, train_valid='train'):
        while True:
            batch_size = self.meta_data['batch_size']
            if train_valid == 'train':
                data_traversing = self.meta_data['data_traversing']
            else:
                data_traversing = 'iterative'

            hr_batch, gsr_batch, output_batch = self.data_handler.get_batch(batch_size=batch_size,
                                                                            train_valid_test=train_valid,
                                                                            data_traversing=data_traversing)

            input_args, pred_args = self.getInputPredArgs(hr_batch, gsr_batch, output_batch)

            yield input_args, pred_args

    def getInputPredArgs(self, hr_batch, gsr_batch, output_batch):
        pass

    # @abstractmethod
    def update_net_specific_meta_data(self):
        pass