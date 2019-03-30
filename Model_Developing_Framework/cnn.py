import math
import random
import pickle
import itertools
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, label_ranking_average_precision_score, label_ranking_loss, coverage_error
from sklearn.utils import shuffle
from scipy.signal import resample
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import OneHotEncoder
import keras
from keras.models import Model
from keras.layers import Input, Dense, Conv1D, MaxPooling1D, Softmax, Add, Flatten, Activation# , Dropout
from keras import backend as K
from keras.optimizers import Adam, sgd
from keras.callbacks import LearningRateScheduler, ModelCheckpoint





class CNN():
    def __init__(self, dict):
        self.dict = dict
        self.BATCH_SIZE = 50
        self.LEARNING_RATE = 0.001
        self.SELECTED_ROWS_IN_SAMPLING = 50   # same as time dimension of model

        

    def fit(self):
        y_train = self.dict.get('train').get('labels')
        x_train = self.dict.get('train').get('Data')
        x_train = np.expand_dims(x_train, 2)
        print('=' * 80)
        print('Start Training')
        print("[INFO] training network...")

        # region make and compile model and optimizer
        # region Model Architecture
        K.clear_session()

        # ************** preparation for Keras layers ****************
        K.set_image_data_format('channels_last')  # to make sure the input shape is [ hr time, channel]

        n_obs, time, depth = x_train.shape

        input_tensor = Input(shape=(time, depth), name='input_HR')
        kernel_init = "he_uniform"
        kernel_regul = keras.regularizers.l2(l=1e-4)

        C11 = Conv1D(filters=1, kernel_size=5, padding='valid', strides=1,
                     activation='linear', kernel_initializer=kernel_init, use_bias=False,
                     kernel_regularizer=kernel_regul)(input_tensor)

        C12 = Conv1D(filters=1, kernel_size=5, padding='same', strides=1,
                     activation='relu', kernel_initializer=kernel_init, use_bias=False,
                     kernel_regularizer=kernel_regul)(C11)
        C13 = Conv1D(filters=1, kernel_size=5, padding='same', strides=1,
                     activation='linear', kernel_initializer=kernel_init, use_bias=False,
                     kernel_regularizer=kernel_regul)(C12)
        S11 = Add()([C13, C11])
        A12 = Activation("relu")(S11)
        M11 = MaxPooling1D(pool_size=5, strides=2)(A12)

        C21 = Conv1D(filters=1, kernel_size=5, padding='same', strides=1,
                     activation='relu', kernel_initializer=kernel_init, use_bias=False,
                     kernel_regularizer=kernel_regul)(M11)
        C22 = Conv1D(filters=1, kernel_size=5, padding='same', strides=1,
                     activation='linear', kernel_initializer=kernel_init, use_bias=False,
                     kernel_regularizer=kernel_regul)(C21)
        S21 = Add()([C22, M11])
        A22 = Activation("relu")(S21)
        M21 = MaxPooling1D(pool_size=5, strides=2)(A22)

      

        F1 = Flatten()(M21)

        D1 = Dense(units=1, activation='relu')(F1)
        D2 = Dense(units=1, activation='relu')(D1)
        output_tensor = Dense(units=1, activation='sigmoid')(D2)

        model = Model(inputs=input_tensor, outputs=output_tensor)
        print(model.summary())
        # plot_model_network(model=model, dst=os.path.join('model_keras.png'))# TODO Fix graphviz stuff
        # endregion Model Architecture

        # opt = Adam(lr = 0.001, beta_1 = 0.9, beta_2 = 0.999)
        opt = sgd(lr=self.LEARNING_RATE, decay=0.85)

        # opt = sgd(lr = self.meta_data['base_lr'],
        #                           decay=self.meta_data['decay'],
        #                           momentum = self.meta_data['momentum'],
        #                           nesterov=self.meta_data['nesterov_bool'])

        model.compile(optimizer=opt  # TODO try the model with "categorical_crossentropy" as well
                      , loss='mean_squared_error', metrics=['accuracy'])
        # , loss=self.meta_data['loss'],
        # loss_weights=self.meta_data['loss_weights'],
        # metrics=self.meta_data['metric'])
        # endregion make and compile model and optimizer

        H = model.fit(
            x=x_train, y=y_train[:, 0], batch_size=self.BATCH_SIZE, epochs=100, verbose=2,
            validation_split=0.2, shuffle=True,
            # class_weight=None, #TODO use this
            # sample_weight=None, #TODO use this maybe
            # initial_epoch=0,
            # steps_per_epoch=None,
            # validation_steps=None,
            # validation_freq=1
        )

        print("End of Training Epoch\n", "-" * 80)


