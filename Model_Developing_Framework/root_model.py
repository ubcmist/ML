from dataHandler import DataHandler
import os
import os.path as osp
import tensorflow as tf
from keras import backend as K
import keras
from keras.optimizers import adam, RMSprop, sgd
from keras.callbacks import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class RootModel:
    def __init__(self, external_meta_data=None):
        self.external_meta_data = external_meta_data
        self.set_directories()
        self.meta_data = None
        self.get_meta_data()
        self.data_handler = DataHandler(self.meta_data, self.framework_root)

    def set_directories(self):
        self.framework_root = osp.dirname(osp.realpath(__file__))
        self.model_dir = osp.join(self.framework_root,'trained',self.__class__.__name__)
        if not osp.exists(self.model_dir):
            os.makedirs(self.model_dir)

    def get_meta_data(self):
        if self.meta_data is None:
            self.meta_data = self.init_meta_data()
            if self.external_meta_data is not None:
                self.meta_data.update(self.external_meta_data)

            self.write_filename = osp.join(self.model_dir, self.meta_data['model_name'])
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

    def set_data(self, data_address_csv):
        pickle_data_address = osp.join(osp.dirname(data_address_csv), osp.basename(data_address_csv)[:-4] + '.pickle')
        if osp.exists(pickle_data_address):
            self.data_handler.all_data_dict = self.data_handler.load_pickle(pickle_data_address)
        else:
            experiments_address_df = pd.read_csv(data_address_csv)
            for indx in range(len(experiments_address_df)):
                self.data_handler.load_data(experiments_address_df.file_address[indx])
            self.data_handler.split_data()
            self.data_handler.save_pickle(self.data_handler.all_data_dict,pickle_data_address)

        print("---- Value distribution for training data:")
        print(self.data_handler.all_data_dict['train']['output_df'].Anxiety_Level.value_counts().sort_index())

        print("---- Value distribution for Validation data:")
        print(self.data_handler.all_data_dict['valid']['output_df'].Anxiety_Level.value_counts().sort_index())

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

        monitor = 'val_' + self.meta_data['metric'][0]
        LR_Scheduler = LearningRateScheduler(self.poly_decay)

        model_saving_address = osp.join(self.write_filename, 'models')
        if not osp.exists(model_saving_address):
            os.makedirs(model_saving_address)
        modelSaveCallback = ModelCheckpoint(os.path.join(model_saving_address,
                                                         '{epoch:02d}-{val_loss:.2f}-model.hdf5'),
                                            monitor='val_loss', verbose=1, save_best_only=True)
        tensorBoardCallback =TensorBoard(log_dir=os.path.join(self.write_filename,'tensorboard_logs') ,
                                          histogram_freq=1, batch_size=self.meta_data['batch_size'],
                                          write_graph=True, write_grads=True, write_images=True,
                                          embeddings_freq=0, embeddings_metadata=None)# TODO fix embedding
        # earlyStopCallback = EarlyStopping(monitor=monitor,
        #                                   min_delta=1, patience=5, verbose=1)

        callbacks = [
                     LR_Scheduler,
                     modelSaveCallback,
                     tensorBoardCallback,
                     # earlyStopCallback,
                     TerminateOnNaN(),
                     BaseLogger(stateful_metrics=['mean_absolute_error']),
                     ]

        H = self.net_model.fit_generator(
            generator=self.Data_Generator(train_valid='train'),
            steps_per_epoch=self.nb_batches_train,
            validation_data=self.Gat_Valid(),
            validation_steps=self.data_handler.get_dataset_size('valid'),
            epochs=self.meta_data['max_epoch'],
            # callbacks=callbacks,
            callbacks=callbacks,
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
        self.plot_Statistics_History(model_name=self.meta_data['model_name'],
                                          H=H.history, accName=self.meta_data['metric'][0],
                                          dst_dir=self.write_filename)
        # endregion

        input_valid, output_valid = self.Gat_Valid()
        output_pred = self.net_model.predict(input_valid, verbose=1)
        prediction_dict = {'output_valid': output_valid, 'output_pred': output_pred}
        self.data_handler.save_pickle(prediction_dict, osp.join(self.write_filename, 'training_results.pickle'))
        self.data_handler.save_mat_file(prediction_dict, osp.join(self.write_filename, 'training_results.mat'))

        prediction = np.argmax(output_pred, axis=1) + 1
        gt = np.argmax(output_valid.get('Anxiety_Level'), axis=1) + 1
        accuracy = int(sum(gt == prediction) / len(gt) * 100)
        # accuracy_vacinty = int(sum(((prediction - 1) == gt) | ((prediction + 1) == gt) | (prediction == gt)) / len(gt)* 100)
        accuracy_vacinty = int(sum(np.abs(prediction-gt) <= 1)/len(gt) * 100)

        prediction_argmax_df = pd.DataFrame({'gt': gt, 'prediction': prediction})
        prediction_argmax_df.to_csv(osp.join(self.write_filename, 'argmax_pred_result_{}%_{}%.csv'.format(accuracy,
                                                                                                          accuracy_vacinty)))

        os.rename(self.write_filename, self.write_filename+'_{}%_{}%.csv'.format(accuracy,accuracy_vacinty))
        K.clear_session()

    def plot_Statistics_History(self, model_name="", accName='categorical_accuracy', H=None, dst_dir=None):
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
        plt.plot(N, H[accName], label="train_accuracy")
        plt.plot(N, H['val_' + accName], label="valid_accuracy")
        plt.title("accuracy plot of model {}".format(model_name))
        plt.xlabel("Epoch #")
        plt.ylabel("Accuracy")
        plt.legend()
        # save the figure
        plt.savefig(os.path.join(dst_dir, 'Accuracy'))
        plt.close()

    def poly_decay(self, epoch):
        # initialize the maximum number of epochs, base learning rate,
        # and power of the polynomial
        baseLR = self.meta_data['base_lr']
        power = epoch//self.meta_data['lr_decay_step']

        # compute the new learning rate based on polynomial decay
        alpha = baseLR * self.meta_data['lr_decay'] ** power
        print('learning rate: {}'.format(alpha))
        # return the new learning rate
        return alpha

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

    def Gat_Valid(self):
        if self.data_handler.all_data_dict.get('valid').get('input') is None:

            hr_batch, gsr_batch, output_batch = self.data_handler.get_batch(batch_size=self.data_handler.get_dataset_size('valid'),
                                                                            train_valid_test='valid',
                                                                            data_traversing='iterative')
            input_args, pred_args = self.getInputPredArgs(hr_batch, gsr_batch, output_batch)
            self.data_handler.all_data_dict['valid']['input'] = input_args
        else:
            input_args = self.data_handler.all_data_dict['valid']['input']
            label = np.asarray(self.data_handler.all_data_dict.get('valid').get('output_df').get('Anxiety_Level'))
            _, pred_args = self.getInputPredArgs(None, None, label)

        return input_args, pred_args

    def getInputPredArgs(self, hr_batch, gsr_batch, output_batch):
        pass

    # @abstractmethod
    def update_net_specific_meta_data(self):
        pass