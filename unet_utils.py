from functools import partial

from pathlib import Path

import pandas as pd

import pickle

import numpy as np
import keras
import nibabel as nib
from nilearn.image.image import check_niimg
from nilearn.image.image import _crop_img_to as crop_img_to

from keras.engine import Input, Model
from keras.optimizers import Adam, RMSprop, Adadelta, SGD
from keras.initializers import he_normal
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D, Flatten, Dense, GlobalAveragePooling3D, concatenate, Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D
from keras import backend as K

data_dir = Path("./data/")
weights_dir = Path("./weights/")
log_dir = Path("./logs/")
pred_dir = Path("./predictions/")

def get_project_root() -> Path:
    """Returns project root folder.
    Usage: 
    from unet_utils import get_project_root
    root = get_project_root()
    """
    return Path(__file__).parent.parent

def crop_img(img, rtol=1e-8, copy=True, return_slices=False):
    """Crops img as much as possible
    Will crop img, removing as many zero entries as possible
    without touching non-zero entries. Will leave one voxel of
    zero padding around the obtained non-zero area in order to
    avoid sampling issues later on.
    Parameters
    ----------
    img: Niimg-like object
        See http://nilearn.github.io/manipulating_images/input_output.html
        img to be cropped.
    rtol: float
        relative tolerance (with respect to maximal absolute
        value of the image), under which values are considered
        negligeable and thus croppable.
    copy: boolean
        Specifies whether cropped data is copied or not.
    return_slices: boolean
        If True, the slices that define the cropped image will be returned.
    Returns
    -------
    cropped_img: image
        Cropped version of the input image
    """

    img = check_niimg(img)
    data = img.get_data()
    infinity_norm = max(-data.min(), data.max())
    passes_threshold = np.logical_or(data < -rtol * infinity_norm,
                                     data > rtol * infinity_norm)

    if data.ndim == 4:
        passes_threshold = np.any(passes_threshold, axis=-1)
    coords = np.array(np.where(passes_threshold))
    start = coords.min(axis=1)
    end = coords.max(axis=1) + 1

    # pad with one voxel to avoid resampling problems
    start = np.maximum(start - 1, 0)
    end = np.minimum(end + 1, data.shape[:3])

    slices = [slice(s, e) for s, e in zip(start, end)]

    if return_slices:
        return slices

    return crop_img_to(img, slices, copy=copy)

def dice_coefficient(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coefficient_loss(y_true, y_pred):
    return -dice_coefficient(y_true, y_pred)


def weighted_dice_coefficient(y_true, y_pred, axis=(-3, -2, -1), smooth=0.00001):
    """
    Weighted dice coefficient. Default axis assumes a "channels first" data structure
    :param smooth:
    :param y_true:
    :param y_pred:
    :param axis:
    :return:
    """
    return K.mean(2. * (K.sum(y_true * y_pred,
                              axis=axis) + smooth/2)/(K.sum(y_true,
                                                            axis=axis) + K.sum(y_pred,
                                                                               axis=axis) + smooth))


def weighted_dice_coefficient_loss(y_true, y_pred):
    return -weighted_dice_coefficient(y_true, y_pred)


def label_wise_dice_coefficient(y_true, y_pred, label_index):
    return dice_coefficient(y_true[:, label_index], y_pred[:, label_index])


def get_label_dice_coefficient_function(label_index):
    f = partial(label_wise_dice_coefficient, label_index=label_index)
    f.__setattr__('__name__', 'label_{0}_dice_coef'.format(label_index))
    return f


# dice_coef = dice_coefficient
# dice_coef_loss = dice_coefficient_loss

def create_convolution_block(input_layer, n_filters, batch_normalization=False, kernel=(3, 3, 3), activation=None,
                             padding='same', strides=(1, 1, 1), instance_normalization=False):
    """
    :param strides:
    :param input_layer:
    :param n_filters:
    :param batch_normalization:
    :param kernel:
    :param activation: Keras activation layer to use. (default is 'relu')
    :param padding:
    :return:
    """
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides, kernel_initializer=he_normal())(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization
        except ImportError:
            raise ImportError("Install keras_contrib in order to use instance normalization."
                              "\nTry: pip install git+https://www.github.com/farizrahman4u/keras-contrib.git")
        layer = InstanceNormalization(axis=1)(layer)
    if activation is None:
        return Activation('relu')(layer)
    else:
        return activation()(layer)


def compute_level_output_shape(n_filters, depth, pool_size, image_shape):
    """
    Each level has a particular output shape based on the number of filters used in that level and the depth or number 
    of max pooling operations that have been done on the data at that point.
    :param image_shape: shape of the 3d image.
    :param pool_size: the pool_size parameter used in the max pooling operation.
    :param n_filters: Number of filters used by the last node in a given level.
    :param depth: The number of levels down in the U-shaped model a given node is.
    :return: 5D vector of the shape of the output node 
    """
    output_image_shape = np.asarray(np.divide(image_shape, np.power(pool_size, depth)), dtype=np.int32).tolist()
    return tuple([None, n_filters] + output_image_shape)


def get_up_convolution(n_filters, pool_size, kernel_size=(2, 2, 2), strides=(2, 2, 2),
                       deconvolution=False):
    if deconvolution:
        return Deconvolution3D(filters=n_filters, kernel_size=kernel_size,
                               strides=strides)
    else:
        return UpSampling3D(size=pool_size)

def create_localization_module(input_layer, n_filters):
    convolution1 = create_convolution_block(input_layer, n_filters)
    convolution2 = create_convolution_block(convolution1, n_filters, kernel=(1, 1, 1))
    return convolution2


def create_up_sampling_module(input_layer, n_filters, size=(2, 2, 2)):
    up_sample = UpSampling3D(size=size)(input_layer)
    convolution = create_convolution_block(up_sample, n_filters)
    return convolution


def create_context_module(input_layer, n_level_filters, dropout_rate=0.3, data_format="channels_first"):
    convolution1 = create_convolution_block(input_layer=input_layer, n_filters=n_level_filters)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format=data_format)(convolution1)
    convolution2 = create_convolution_block(input_layer=dropout, n_filters=n_level_filters)
    return convolution2


create_convolution_block = partial(create_convolution_block, activation=LeakyReLU, instance_normalization=True)

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, batch_size=2, dim=(240,240,155), n_channels=4,
                 n_classes=3, shuffle=True, num_outputs=3):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        # self.labels = labels
        self.tumor_type_dict = pickle.load( open( "tumor_type_dict.pkl", "rb" ) )
        self.survival_data = pd.read_csv('survival_data.csv')
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.num_outputs = num_outputs

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        if self.num_outputs == 1:
            # Generate data
            X, y1 = self.__data_generation(list_IDs_temp)

            return X, y1

        if self.num_outputs == 2:
            # Generate data
            X, y1, y2 = self.__data_generation(list_IDs_temp)

            return X, [y1, y2]

        if self.num_outputs==3:
            # Generate data
            X, y1, y2, y3 = self.__data_generation(list_IDs_temp)

            return X, [y1, y2, y3]                      

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):

        data_dir = Path("./data/")

        if self.num_outputs == 1:
            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization
            X = np.empty((self.batch_size, self.n_channels, *self.dim))
            y1 = np.empty((self.batch_size, 3, *self.dim))

            # Generate data
            # Decode and load the data
            for i, ID in enumerate(list_IDs_temp):
                
                # 1) the "enhancing tumor" (ET), 2) the "tumor core" (TC), and 3) the "whole tumor" (WT) 
                # The ET is described by areas that show hyper-intensity in T1Gd when compared to T1, but also when compared to “healthy” white matter in T1Gd. 
                # The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (fluid-filled) and the non-enhancing (solid) parts of the tumor. 
                # The appearance of the necrotic (NCR) and the non-enhancing (NET) tumor core is typically hypo-intense in T1-Gd when compared to T1. 
                # The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edema (ED), which is typically depicted by hyper-intense signal in FLAIR.
                # The labels in the provided data are: 
                # 1 for NCR & NET (necrotic (NCR) and the non-enhancing (NET) tumor core) = TC ("tumor core")
                # 2 for ED ("peritumoral edema")
                # 4 for ET ("enhancing tumor")
                # 0 for everything else

                X[i,] = pickle.load( open( data_dir / f"{ID}_images.pkl", "rb" ) )
                y1[i,] = pickle.load( open( data_dir / f"{ID}_seg_mask_3ch.pkl", "rb" ) )
                
            return X, y1

        if self.num_outputs == 2:
            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization
            X = np.empty((self.batch_size, self.n_channels, *self.dim))
            y1 = np.empty((self.batch_size, 3, *self.dim))
            y2 = np.empty(self.batch_size)
    #         y2 = list()

            # Generate data
            # Decode and load the data
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = pickle.load( open( data_dir / f"{ID}_images.pkl", "rb" ) )
                y1[i,] = pickle.load( open( data_dir / f"{ID}_seg_mask_3ch.pkl", "rb" ) )
                y2[i,] = self.tumor_type_dict[ID]
                
            return X, y1, y2


        if self.num_outputs==3:
            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization
            X = np.empty((self.batch_size, self.n_channels, *self.dim))
            y1 = np.empty((self.batch_size, 3, *self.dim))
            y2 = np.empty(self.batch_size)
            y3 = np.empty(self.batch_size)

            # Generate data
            # Decode and load the data
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = pickle.load( open( data_dir / f"{ID}_images.pkl", "rb" ) )
                y1[i,] = pickle.load( open( data_dir / f"{ID}_seg_mask_3ch.pkl", "rb" ) )            
                y2[i,] = self.tumor_type_dict[ID]
                y3[i,] = self.survival_data[self.survival_data.Brats17ID==ID].Survival.astype(int).values.item(0)

            return X, y1, y2, y3

# change the number of labels?
# loss_function={'activation_block': weighted_dice_coefficient_loss, 'survival_block': 'mean_squared_error'}
# selected_optimizer = RMSprop
# selected_initial_learning_rate = 5e-4

def create_model(input_shape=(4, 160, 192, 160),
    n_base_filters=12,
    depth=5,
    dropout_rate=0.3,
    n_segmentation_levels=3,
    n_labels=3,
    num_outputs=3,
    optimizer='adam',
    learning_rate=1e-3,
    activation_name="sigmoid"):
    """
    This function builds a model proposed by Isensee et al. for the BRATS 2017 competition:
    https://www.cbica.upenn.edu/sbia/Spyridon.Bakas/MICCAI_BraTS/MICCAI_BraTS_2017_proceedings_shortPapers.pdf
    This network is highly similar to the model proposed by Kayalibay et al. "CNN-based Segmentation of Medical
    Imaging Data", 2017: https://arxiv.org/pdf/1701.03056.pdf
    :param input_shape:
    :param n_base_filters:
    :param depth:
    :param dropout_rate:
    :param n_segmentation_levels:
    :param n_labels:
    :param optimizer:
    :param initial_learning_rate:
    :param loss_function:
    :param activation_name:
    :return:
    """

    if optimizer.lower() == 'adam':
        optimizer = Adam(lr=learning_rate)

    inputs = Input(input_shape)

    current_layer = inputs
    level_output_layers = list()
    level_filters = list()
    for level_number in range(depth):
        n_level_filters = (2**level_number) * n_base_filters
        level_filters.append(n_level_filters)

        if current_layer is inputs:
            in_conv = create_convolution_block(current_layer, n_level_filters)
        else:
            in_conv = create_convolution_block(current_layer, n_level_filters, strides=(2, 2, 2))

        context_output_layer = create_context_module(in_conv, n_level_filters, dropout_rate=dropout_rate)

        summation_layer = Add()([in_conv, context_output_layer])
        level_output_layers.append(summation_layer)
        current_layer = summation_layer

    segmentation_layers = list()
    for level_number in range(depth - 2, -1, -1):
        up_sampling = create_up_sampling_module(current_layer, level_filters[level_number])
        concatenation_layer = concatenate([level_output_layers[level_number], up_sampling], axis=1)
        localization_output = create_localization_module(concatenation_layer, level_filters[level_number])
        current_layer = localization_output
        if level_number < n_segmentation_levels:
            segmentation_layers.insert(0, create_convolution_block(current_layer, n_filters=n_labels, kernel=(1, 1, 1)))

    output_layer = None
    for level_number in reversed(range(n_segmentation_levels)):
        segmentation_layer = segmentation_layers[level_number]
        if output_layer is None:
            output_layer = segmentation_layer
        else:
            output_layer = Add()([output_layer, segmentation_layer])

        if level_number > 0:
            output_layer = UpSampling3D(size=(2, 2, 2))(output_layer)

    activation_block = Activation(activation = activation_name, name='activation_block')(output_layer)
    #     survival_block = Activation("linear")(summation_layer)
    #     activation_block = Dense(1, activation=activation_name, name='activation_block')(output_layer)
    #     flatten = Flatten(name='flatten')(summation_layer)
    #     survival_block = Dense(1, activation='linear', name='survival_block')(flatten)

    survival_conv_1 = Conv3D(filters=n_level_filters, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1), name='survival_conv_1')(summation_layer)
    survival_conv_2 = Conv3D(filters=n_level_filters, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1), name='survival_conv_2')(survival_conv_1)
    dropout = SpatialDropout3D(rate=dropout_rate, data_format='channels_first', name='dropout')(survival_conv_2)
    survival_conv_3 = Conv3D(filters=n_level_filters, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1), name='survival_conv_3')(dropout)
    survival_GAP = GlobalAveragePooling3D(name='survival_GAP')(survival_conv_3)
    #     flatten = Flatten(name='flatten')(survival_GAP)
    #     survival_block = Activation("linear", name='survival_block')(flatten)
    survival_block = Dense(1, activation='linear', name='survival_block')(survival_GAP)

    tumortype_conv_1 = Conv3D(filters=n_level_filters, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1), name='tumortype_conv_1')(summation_layer)
    tumortype_conv_2 = Conv3D(filters=n_level_filters, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1), name='tumortype_conv_2')(tumortype_conv_1)
    tumortype_dropout = SpatialDropout3D(rate=dropout_rate, data_format='channels_first', name='tumortype_dropout')(tumortype_conv_2)
    tumortype_conv_3 = Conv3D(filters=n_level_filters, kernel_size=(3, 3, 3), padding='same', strides=(1, 1, 1), name='tumortype_conv_3')(tumortype_dropout)
    tumortype_GAP = GlobalAveragePooling3D(name='tumortype_GAP')(tumortype_conv_3)
    #     flatten = Flatten(name='flatten')(tumortype_GAP)
    #     tumortype_block = Activation("linear", name='tumortype_block')(flatten)
    tumortype_block = Dense(1, activation='sigmoid', name='tumortype_block')(tumortype_GAP)  

    model = Model(inputs=inputs, outputs=[activation_block])
    #     model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    #     loss={'activation_block': 'binary_crossentropy', 'survival_block': 'mean_squared_error'}
    # assign weights and loss as dictionaries
    # functional-api-guide
    # loss_weights define the ratio of how much I care about optimizing each one
    
    if num_outputs == 3:
        model = Model(inputs=inputs, outputs=[activation_block,tumortype_block,survival_block])
        # model.load_weights(weights_dir / f"model_weights_{num_outputs}_outputs.h5", by_name=True)

        model.compile(optimizer=optimizer, 
                    loss={'activation_block': weighted_dice_coefficient_loss, 'survival_block': 'mean_squared_error', 'tumortype_block': 'binary_crossentropy'}, 
                    loss_weights={'activation_block': 1., 'survival_block': 0.2, 'tumortype_block': 0.2},
                    metrics={'activation_block': ['accuracy', weighted_dice_coefficient, dice_coefficient], 'survival_block': ['accuracy', 'mae'], 'tumortype_block': ['accuracy']})

    if num_outputs == 2:
        model = Model(inputs=inputs, outputs=[activation_block,tumortype_block])
        # model.load_weights(weights_dir / f"model_weights_{num_outputs}_outputs.h5", by_name=True) # the by_name=True allows you to use a different architecture and bring in the weights from the matching layers 

        model.compile(optimizer=optimizer, 
                    loss={'activation_block': weighted_dice_coefficient_loss, 'tumortype_block': 'binary_crossentropy'}, 
                    loss_weights={'activation_block': 1., 'tumortype_block': 0.2},
                    metrics={'activation_block': ['accuracy',weighted_dice_coefficient, dice_coefficient], 'tumortype_block': ['accuracy']})
    
    if num_outputs == 1:
        model = Model(inputs=inputs, outputs=[activation_block])
        # model.load_weights(weights_dir / f"model_weights_{num_outputs}_outputs.h5", by_name=True) # by_name=True allows you to use a different architecture and bring in the weights from the matching layers 

        model.compile(optimizer=optimizer, 
                    loss={'activation_block': weighted_dice_coefficient_loss}, 
                    loss_weights={'activation_block': 1.},
                    metrics={'activation_block': ['accuracy',weighted_dice_coefficient, dice_coefficient]})

    print(model.summary(line_length=150)) # add the parameter that allows me to show everything instead of cutting it off 
        
    return model



    # model.compile(optimizer=RMSprop(lr=5e-4), 
    #             loss={'activation_block': weighted_dice_coefficient_loss}, 
    #             loss_weights={'activation_block': 1.},
    #             metrics={'activation_block': ['accuracy',weighted_dice_coefficient, dice_coefficient]})

    # model.summary(line_length=150) # add the parameter that allows me to show everything instead of cutting it off

    # model.save_weights("./weights/model_3_weights.h5")
    # print("Saved model to disk")