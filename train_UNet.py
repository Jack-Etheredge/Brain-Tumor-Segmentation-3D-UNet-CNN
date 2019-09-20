from unet_utils import *

import pickle

from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D
from keras.engine import Model
from keras.optimizers import Adam
from keras.optimizers import RMSprop

import pandas as pd

import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D, Flatten, Dense, GlobalAveragePooling3D

# K.set_image_dim_ordering('th')
K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_first')

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
print(len(device_lib.list_local_devices()))

def train_unet(num_outputs):
    
    if num_outputs == 3:
        model = Model(inputs=inputs, outputs=[activation_block,tumortype_block,survival_block])
        model.load_weights("./weights/model_2_weights.h5", by_name=True)

        model.compile(optimizer=RMSprop(lr=5e-4), 
                    loss={'activation_block': weighted_dice_coefficient_loss, 'survival_block': 'mean_squared_error', 'tumortype_block': 'binary_crossentropy'}, 
                    loss_weights={'activation_block': 1., 'survival_block': 0.2, 'tumortype_block': 0.2},
                    metrics={'activation_block': ['accuracy',weighted_dice_coefficient, dice_coefficient], 'survival_block': ['accuracy', 'mae'], 'tumortype_block': ['accuracy']})

        model.summary(line_length=150) # add the parameter that allows me to show everything instead of cutting it off

        params = {'dim': (160,192,160),
                'batch_size': 1,
                'n_classes': 3,
                'n_channels': 4,
                'shuffle': True}

        # Generators
        training_generator = DataGenerator(train_val_test_dict['train'], num_outputs, **params)
        validation_generator = DataGenerator(train_val_test_dict['test'], num_outputs, **params)

        cb_1=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
        cb_2=keras.callbacks.ModelCheckpoint(filepath="./weights/3pred_weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        results = model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                        epochs=100, 
                        nb_worker=4,
                        callbacks=[cb_1,cb_2])

        model.save_weights("./weights/model_3_weights.h5")
        print("Saved model to disk")


    if num_outputs == 2:
        model = Model(inputs=inputs, outputs=[activation_block,tumortype_block])
        model.load_weights("./weights/model_1_weights.h5", by_name=True) # the by_name=True allows you to use a different architecture and bring in the weights from the matching layers 

        model.compile(optimizer=RMSprop(lr=5e-4), 
                    loss={'activation_block': weighted_dice_coefficient_loss, 'tumortype_block': 'binary_crossentropy'}, 
                    loss_weights={'activation_block': 1., 'tumortype_block': 0.2},
                    metrics={'activation_block': ['accuracy',weighted_dice_coefficient, dice_coefficient], 'tumortype_block': ['accuracy']})

        model.summary(line_length=150) # add the parameter that allows me to show everything instead of cutting it off
                params = {'dim': (160,192,160),
                        'batch_size': 1,
                        'n_classes': 3,
                        'n_channels': 4,
                        'shuffle': True}

        # Generators
        training_generator = DataGenerator(train_val_test_dict['train'], num_outputs, **params)
        validation_generator = DataGenerator(train_val_test_dict['test'], num_outputs, **params)

        cb_1=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
        cb_2=keras.callbacks.ModelCheckpoint(filepath="./weights/2pred_weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        results = model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                        epochs=100, 
                        nb_worker=4,
                        callbacks=[cb_1,cb_2])

        model.save_weights("./weights/model_2_weights.h5")
        print("Saved model to disk")
    
    if num_outputs == 1:
        model = Model(inputs=inputs, outputs=[activation_block])
        model.load_weights("./weights/1pred_weights.25--0.08.hdf5", by_name=True) # by_name=True allows you to use a different architecture and bring in the weights from the matching layers 

        model.compile(optimizer=RMSprop(lr=5e-4), 
                    loss={'activation_block': weighted_dice_coefficient_loss}, 
                    loss_weights={'activation_block': 1.},
                    metrics={'activation_block': ['accuracy',weighted_dice_coefficient, dice_coefficient]})

        model.summary(line_length=150) # add the parameter that allows me to show everything instead of cutting it off

        params = {'dim': (160,192,160),
          'batch_size': 1,
          'n_classes': 3,
          'n_channels': 4,
          'shuffle': True}

        # Generators
        training_generator = DataGenerator(train_val_test_dict['train'], num_outputs, **params)
        validation_generator = DataGenerator(train_val_test_dict['test'], num_outputs, **params)

        cb_1=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
        cb_2=keras.callbacks.ModelCheckpoint(filepath="./weights/1Bpred_weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

        results = model.fit_generator(generator=training_generator,
                            validation_data=validation_generator,
                        epochs=100, 
                        nb_worker=4,
                        callbacks=[cb_1,cb_2])
        model.save_weights("./weights/model_1_weights.h5")
        print("Saved model to disk")

