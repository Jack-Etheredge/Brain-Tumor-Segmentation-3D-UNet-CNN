# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% [markdown]
# ### Load pickles for prediction and sample dictionary:

#%%
import pickle
import os
import numpy as np

import copy
import nibabel as nib

from tifffile import imsave
from tifffile import imsave
from libtiff import TIFF

from skimage.io._plugins import freeimage_plugin as fi


#%%
data_path = "/Volumes/EtheredgeJ_External_Personal_2/05_3DCNN/weights/"


#%%
prediction = pickle.load( open( data_path+"predictions_2_multi_gpu_pred.pkl", "rb" ) )


#%%
partition = pickle.load( open( data_path+"partition.pkl", "rb" ) )

#%% [markdown]
# ### Saving a pickle for each prediction:

#%%
for i in range(len(prediction)):
    ID = partition['holdout'][i]
    pickle.dump(prediction[i] open("./data/"+ID+"_prediction.pkl", "wb"))

#%% [markdown]
# ### Making ground truth .tiff files: 
# - open the pickle file with the cropped image and cropped segmentation
# - add channel 0 of the image (the flair) to the segmentation to get the ground truth
# - add channel 0 of the image (the flair) to the prediction to get the ground truth

#%%
def make_256(a):
    a = a.astype(float)
    a *= 255.0/a.max()  # convert to 8-bit pixel values
    a = np.rollaxis(a, 0, 2)
    a = a.astype('uint8')    
    return(a)


#%%


for i in range(len(prediction)):
    ID = partition['holdout'][i]
    pred_seg = pickle.load( open( "./data/"+ID+"_prediction.pkl", "rb" ) )
    gt_seg = pickle.load( open( "./data/"+ID+"_seg_mask_3ch.pkl", "rb" ) )
    images = pickle.load( open( "./data/"+ID+"_images.pkl", "rb" ) )

    print("current image:", i, ID)

    flair, b, c, d = np.split(images, 4, axis=0)
    pred_seg_mask_1, pred_seg_mask_2, pred_seg_mask_3 = np.split(pred_seg, 3, axis=0)
    gt_seg_mask_1, gt_seg_mask_2, gt_seg_mask_3 = np.split(gt_seg, 3, axis=0)

    imsave("./prediction_tiffs/"+ID+"flair.tif", make_256(flair), 'imagej')
    imsave("./prediction_tiffs/"+ID+"pred_seg_mask_1.tif", make_256(pred_seg_mask_1), 'imagej')
    imsave("./prediction_tiffs/"+ID+"pred_seg_mask_2.tif", make_256(pred_seg_mask_2), 'imagej')
    imsave("./prediction_tiffs/"+ID+"pred_seg_mask_3.tif", make_256(pred_seg_mask_3), 'imagej')
    imsave("./prediction_tiffs/"+ID+"gt_seg_mask_1.tif", make_256(gt_seg_mask_1), 'imagej')
    imsave("./prediction_tiffs/"+ID+"gt_seg_mask_2.tif", make_256(gt_seg_mask_2), 'imagej')
    imsave("./prediction_tiffs/"+ID+"gt_seg_mask_3.tif", make_256(gt_seg_mask_3), 'imagej')


#%%
partition['holdout'][0]


#%%
# import copy
# import nibabel as nib

# from libtiff import TIFF

# from skimage.io._plugins import freeimage_plugin as fi

# # import javabridge
# # import bioformats
# # javabridge.start_vm(class_path=bioformats.JARS)

# # your program goes here


# # ID = testIDlist[i]
# # for i in range(len(testIDlist)):

#%% [markdown]
# ### Adding predictions:

# import javabridge
# import bioformats
# javabridge.start_vm(class_path=bioformats.JARS)

# your program goes here


# ID = testIDlist[i]
# for i in range(len(testIDlist)):
for i in range(2):

    print("current image:", i)

    ID = testIDlist[i]
    img1 = './data/' + ID + '_flair.nii.gz'
    img2 = './data/' + ID + '_t1.nii.gz'
    img3 = './data/' + ID + '_t1ce.nii.gz'
    img4 = './data/' + ID + '_t2.nii.gz'
    img5 = './data/' + ID + '_seg.nii.gz'

    newimage = nib.concat_images([img1, img2, img3, img4, img5])
    cropped = crop_img(newimage)
    img_array = np.array(cropped.dataobj)
    z = np.rollaxis(img_array, 3, 0)

    padded_image = np.zeros((5, 160, 192, 160))
    padded_image[:z.shape[0], :z.shape[1], :z.shape[2], :z.shape[3]] = z

    a, b, c, d, seg_mask = np.split(padded_image, 5, axis=0)

    images = np.concatenate([a, b, c, d], axis=0)

    # print("images shape:", images.shape, "images values:", np.unique(images.astype(int)))

    # split the channels:
    # seg_mask_1 = copy.deepcopy(seg_mask.astype(int))
    seg_mask_1 = np.zeros((1, 160, 192, 160))
    seg_mask_1[seg_mask.astype(int) > 0] = 1
    seg_mask_2 = np.zeros((1, 160, 192, 160))
    seg_mask_2[seg_mask.astype(int) > 1] = 1
    seg_mask_3 = np.zeros((1, 160, 192, 160))
    seg_mask_3[seg_mask.astype(int) > 2] = 1
    seg_mask_3ch = np.concatenate(
        [seg_mask_1, seg_mask_2, seg_mask_3], axis=0).astype(int)

    # def scale_image(image_array):
    #     image_array = image_array.astype(float)
    #     image_array *= 255.0/image_array.max() # convert to 8-bit pixel values
    #     image_array = image_array.astype(int)
    #     return image_array

    # img_array_list = [a,seg_mask_1,seg_mask_2,seg_mask_3]
    # for img_array in img_array_list:
    #     img_array = scale_image(img_array)

    a = a.astype(float)
    a *= 255.0/a.max()  # convert to 8-bit pixel values
    a = np.rollaxis(a, 0, 2) # cxyz -> xycz for imagej
    a = np.rollaxis(a, 0, 3) # switching x and z
    a = a.astype('uint8')
#     print("unique flair values:", np.unique(a))

    seg_mask_1 = seg_mask_1.astype(float)
    seg_mask_1 *= 255.0/seg_mask_1.max()  # convert to 8-bit pixel values
    seg_mask_1 = np.rollaxis(seg_mask_1, 0, 2) # cxyz -> xycz for imagej
    seg_mask_1 = np.rollaxis(seg_mask_1, 0, 3) # switching x and z
    seg_mask_1 = seg_mask_1.astype('uint8')
#     print("unique segment mask values:", np.unique(seg_mask_1))

    seg_mask_2 = seg_mask_2.astype(float)
    seg_mask_2 *= 255.0/seg_mask_2.max()  # convert to 8-bit pixel values
    seg_mask_2 = np.rollaxis(seg_mask_2, 0, 2) # cxyz -> xycz for imagej
    seg_mask_2 = np.rollaxis(seg_mask_2, 0, 3) # switching x and z
    seg_mask_2 = seg_mask_2.astype('uint8')

    seg_mask_3 = seg_mask_3.astype(float)
    seg_mask_3 *= 255.0/seg_mask_3.max()  # convert to 8-bit pixel values
    seg_mask_3 = np.rollaxis(seg_mask_3, 0, 2) # cxyz -> xycz for imagej
    seg_mask_3 = np.rollaxis(seg_mask_3, 0, 3) # switching x and z
    seg_mask_3 = seg_mask_3.astype('uint8')

#     ground_truth = np.concatenate(
#         [a, seg_mask_1, seg_mask_2, seg_mask_3], axis=0).astype('uint8')

#     print("unique flair + segment mask values:", np.unique(ground_truth))
    # shape.ground_truth
    # flairimg = flairimg.astype(float)
    # flairimg *= 255.0/flairimg.max() # convert to 8-bit pixel values
    # flairimg = flairimg.astype(int)
    # print(np.unique(flairimg))
#     print("final image shape:", ground_truth.shape)
#     imsave("./channel_split/"+testIDlist[i]+"ground_truth.tif", ground_truth, 'imagej')
    imsave("./channel_split/"+testIDlist[i]+"_flair.tif", a, 'imagej')
    imsave("./channel_split/"+testIDlist[i]+"_ground_truth_1.tif", seg_mask_1, 'imagej')
    imsave("./channel_split/"+testIDlist[i]+"_ground_truth_2.tif", seg_mask_2, 'imagej')
    imsave("./channel_split/"+testIDlist[i]+"_ground_truth_3.tif", seg_mask_3, 'imagej')

    imarray = pickle.load(open( "./channel_split/prediction_"+str(i)+".pkl", "rb" ) )

    prediction_thresh = copy.deepcopy(imarray)
    prediction_thresh[prediction_thresh < 0.5] = 0.
    prediction_thresh[prediction_thresh >= 0.5] = 1.
    prediction_thresh = prediction_thresh
    print(np.unique(prediction_thresh))
    prediction_thresh *= 255.0/prediction_thresh.max() # convert to 8-bit pixel values
    prediction_thresh = prediction_thresh.astype('uint8')
    prediction_thresh = np.rollaxis(prediction_thresh, 1, 3) # switching x and z; c will be taken care of in split
    print(np.unique(prediction_thresh))
    print(prediction_thresh.shape)

    pred1, pred2, pred3 = np.split(prediction_thresh, 3, axis=0)

    imsave("./channel_split/"+testIDlist[i]+"_predicted_1.tif", pred1, 'imagej')
    imsave("./channel_split/"+testIDlist[i]+"_predicted_2.tif", pred2, 'imagej')
    imsave("./channel_split/"+testIDlist[i]+"_predicted_3.tif", pred3, 'imagej')

    # print("images shape:", images.shape, "images values:", np.unique(images.astype(int)))

    # split the channels:
    # seg_mask_1 = copy.deepcopy(seg_mask.astype(int))


    #     seg_mask_3ch = np.concatenate(
#         [seg_mask_1, seg_mask_2, seg_mask_3], axis=0).astype(int)

#     imarray *= 255.0/imarray.max()

#     imsave("./channel_split/"+testIDlist[i]+"ground_truth_1.tif", seg_mask_1, 'imagej')
#     imsave("./channel_split/"+testIDlist[i]+"ground_truth_2.tif", seg_mask_2, 'imagej')
#     imsave("./channel_split/"+testIDlist[i]+"ground_truth_3.tif", seg_mask_3, 'imagej')

#%% [markdown]
# # Probably remove this stuff upon cleanup:
#%% [markdown]
# ### Write images to .tif:

#%%
from tifffile import imsave
for i in range(len(prediction)):
    imarray = pickle.load(open( "./channel_split/prediction_"+str(i)+".pkl", "rb" ) )
    
    # open the pickle file with the cropped image and cropped segmentation
    # add channel 0 of the image (the flair) to the segmentation to get the ground truth
    # add channel 0 of the image (the flair) to the prediction to get the ground truth

    imarray[]
    imarray *= 255.0/imarray.max()
    print(np.unique())
    
    imsave("./channel_split/"+testIDlist[i]+"prediction.tif", imarray)

    # make ground truth 
    ID = testIDlist[i]
    img1 = './data/' + ID + '_flair.nii.gz'
    img2 = './data/' + ID + '_t1.nii.gz'
    img3 = './data/' + ID + '_t1ce.nii.gz'
    img4 = './data/' + ID + '_t2.nii.gz'
    img5 = './data/' + ID + '_seg.nii.gz'

    newimage = nib.concat_images([img1, img2, img3, img4, img5])
    cropped = crop_img(newimage)         
    img_array = np.array(cropped.dataobj)
    z = np.rollaxis(img_array, 3, 0)

    padded_image = np.zeros((5,160,192,160))
    padded_image[:z.shape[0],:z.shape[1],:z.shape[2],:z.shape[3]] = z

    a,b,c,d,seg_mask = np.split(padded_image, 5, axis=0)

    images = np.concatenate([a,b,c,d], axis=0)
    imsave("./channel_split/"+testIDlist[i]+"ground_truth.tif", images)
    

#%% [markdown]
# ### Testing:

#%%
import pickle
import numpy as np
import copy
import nibabel as nib

i = 0

imarray = pickle.load(open( "./channel_split/prediction_"+str(i)+".pkl", "rb" ) )
# threshold the channels (for prediction):
prediction_thresh = copy.deepcopy(imarray)
prediction_thresh[prediction_thresh < 0.5] = 0.
prediction_thresh[prediction_thresh >= 0.5] = 1.
prediction_thresh = prediction_thresh
print(np.unique(prediction_thresh))
prediction_thresh *= 255.0/prediction_thresh.max() # convert to 8-bit pixel values
prediction_thresh = prediction_thresh.astype(int)
print(np.unique(prediction_thresh))
print(prediction_thresh.shape)

ID = testIDlist[i]
img1 = './data/' + ID + '_flair.nii.gz'
flairimg = nib.load(img1)
flairimg = np.array(flairimg.dataobj)
flairimg = np.expand_dims(flairimg, axis=0)
flairimg = np.rollaxis(flairimg, 3, 0)
print(np.unique(flairimg))
flairimg = flairimg.astype(float)
flairimg *= 255.0/flairimg.max() # convert to 8-bit pixel values
flairimg = flairimg.astype(int)
print(np.unique(flairimg))
print(flairimg.shape)

#%% [markdown]
# ### Imports:

#%%
from functools import partial

from keras.layers import Input, LeakyReLU, Add, UpSampling3D, Activation, SpatialDropout3D
from keras.engine import Model
from keras.optimizers import Adam

import numpy as np
from keras import backend as K
from keras.engine import Input, Model
from keras.layers import Conv3D, MaxPooling3D, UpSampling3D, Activation, BatchNormalization, PReLU, Deconvolution3D
from keras.optimizers import Adam

# K.set_image_dim_ordering('th')
K.set_image_dim_ordering('tf')
K.set_image_data_format('channels_first')

try:
    from keras.engine import merge
except ImportError:
    from keras.layers.merge import concatenate

#%% [markdown]
# ### Calculating metrics:

#%%
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


dice_coef = dice_coefficient
dice_coef_loss = dice_coefficient_loss


#%%
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
    layer = Conv3D(n_filters, kernel, padding=padding, strides=strides)(input_layer)
    if batch_normalization:
        layer = BatchNormalization(axis=1)(layer)
    elif instance_normalization:
        try:
            from keras_contrib.layers.normalization import InstanceNormalization
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


def isensee2017_model(input_shape=(4, 128, 128, 128), n_base_filters=16, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=4, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid"):
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

    activation_block = Activation(activation_name)(output_layer)

    model = Model(inputs=inputs, outputs=activation_block)
    model.compile(optimizer=optimizer(lr=initial_learning_rate), loss=loss_function)
    return model


#%%
# change the number of labels?

model = isensee2017_model(input_shape=(4, 160, 192, 160), n_base_filters=6, depth=5, dropout_rate=0.3,
                      n_segmentation_levels=3, n_labels=3, optimizer=Adam, initial_learning_rate=5e-4,
                      loss_function=weighted_dice_coefficient_loss, activation_name="sigmoid")
model.summary()

#%% [markdown]
# ## Make the labels and test train dictionaries:

#%%
# from glob import glob
# paths = glob('/Users/etheredgej/Desktop/MICCAI_BraTS17_Data_Training/train/HGG/*/')
# print(paths)

import os
HGG_dir_list = next(os.walk('./HGG/'))[1]
print(len(HGG_dir_list))
LGG_dir_list = next(os.walk('./LGG/'))[1]
print(len(LGG_dir_list))


#%%
completelist = HGG_dir_list + LGG_dir_list
# print(completelist[0:4])
np.random.shuffle(completelist) # shuffles in place
# print(completelist[0:4])

partition={}

holdout_percentage=0.15
partition['holdout']=completelist[0:int(len(completelist)*holdout_percentage)]
trainlist=completelist[int(len(completelist)*holdout_percentage):len(completelist)]

train_percentage=0.7
partition['train']=trainlist[0:int(len(trainlist)*train_percentage)]
partition['test']=trainlist[int(len(trainlist)*train_percentage):len(trainlist)]


labels={}
# HGG=0
# LGG=1
for directory in HGG_dir_list:
    labels[directory]=0
for directory in LGG_dir_list:
    labels[directory]=1
    
print(len(partition['holdout']))
print(len(partition['train']))
print(len(partition['test']))

#%% [markdown]
# ### crop_img function:

#%%
import numpy as np
from nilearn.image.image import check_niimg
from nilearn.image.image import _crop_img_to as crop_img_to


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

#%% [markdown]
# ### Data generator:

#%%
# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

import numpy as np
import keras
import nibabel as nib

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=2, dim=(240,240,155), n_channels=4,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size, 3, *self.dim))

        # Generate data
        # Decode and load the data
        for i, ID in enumerate(list_IDs_temp):

            img1 = './data/' + ID + '_flair.nii.gz'
            img2 = './data/' + ID + '_t1.nii.gz'
            img3 = './data/' + ID + '_t1ce.nii.gz'
            img4 = './data/' + ID + '_t2.nii.gz'
            img5 = './data/' + ID + '_seg.nii.gz'

            newimage = nib.concat_images([img1, img2, img3, img4, img5])
            cropped = crop_img(newimage)         
            img_array = np.array(cropped.dataobj)
            z = np.rollaxis(img_array, 3, 0)

            padded_image = np.zeros((5,160,192,160))
            padded_image[:z.shape[0],:z.shape[1],:z.shape[2],:z.shape[3]] = z

            a,b,c,d,seg_mask = np.split(padded_image, 5, axis=0)

            images = np.concatenate([a,b,c,d], axis=0)

            # print("images shape:", images.shape, "images values:", np.unique(images.astype(int)))

            # split the channels:
            # seg_mask_1 = copy.deepcopy(seg_mask.astype(int))
            seg_mask_1 = np.zeros((1,160,192,160))
            seg_mask_1[seg_mask.astype(int) == 1] = 1
            seg_mask_2 = np.zeros((1,160,192,160))
            seg_mask_2[seg_mask.astype(int) == 2] = 1
            seg_mask_3 = np.zeros((1,160,192,160))
            seg_mask_3[seg_mask.astype(int) == 4] = 1
            seg_mask_3ch = np.concatenate([seg_mask_1,seg_mask_2,seg_mask_3], axis=0).astype(int)
            
            # 1) the "enhancing tumor" (ET), 2) the "tumor core" (TC), and 3) the "whole tumor" (WT) 
            # The ET is described by areas that show hyper-intensity in T1Gd when compared to T1, but also when compared to “healthy” white matter in T1Gd. The TC describes the bulk of the tumor, which is what is typically resected. The TC entails the ET, as well as the necrotic (fluid-filled) and the non-enhancing (solid) parts of the tumor. The appearance of the necrotic (NCR) and the non-enhancing (NET) tumor core is typically hypo-intense in T1-Gd when compared to T1. The WT describes the complete extent of the disease, as it entails the TC and the peritumoral edema (ED), which is typically depicted by hyper-intense signal in FLAIR.
            # The labels in the provided data are: 
            # 1 for NCR & NET (necrotic (NCR) and the non-enhancing (NET) tumor core) = TC ("tumor core")
            # 2 for ED ("peritumoral edema")
            # 4 for ET ("enhancing tumor")
            # 0 for everything else

            X[i,] = images
            y[i,] = seg_mask_3ch
            
#         return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return X, y


#%%
import numpy as np

from keras.models import Sequential
# from my_classes import DataGenerator

# Parameters
params = {'dim': (160,192,160),
          'batch_size': 2,
          'n_classes': 3,
          'n_channels': 4,
          'shuffle': True}

# # Datasets
# partition = # IDs
# labels = # Labels

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['test'], labels, **params)

# # Design model
# model = Sequential()
# [...] # Architecture
# model.compile()

# Train model on dataset
# model.fit_generator(generator=training_generator,
#                     validation_data=validation_generator,
#                     use_multiprocessing=True,
#                     workers=4)

cb_1=keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')
cb_2=keras.callbacks.ModelCheckpoint(filepath=weights.{epoch:02d}-{val_loss:.2f}.hdf5, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

results = model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                   epochs=2, 
                   nb_worker=4,
                   callbacks=[cb_1,cb_2])
print("Test-Accuracy:", np.mean(results.history["val_acc"]))


# model.fit_generator(
#         generator=training_generator,
#         samples_per_epoch=2,
#         steps_per_epoch=2000,
#         epochs=50,
#         validation_data=validation_generator,
#         nb_worker=1,
#         validation_steps=800)


#%%
# score = model.evaluate(x_test, y_test, verbose=0)

params = {'dim': (160,192,160),
          'batch_size': 2,
          'n_classes': 3,
          'n_channels': 4,
          'shuffle': False}

# Turned shuffle off so that we can match the values in the dictionary to the predictions. 
# This way we can compare the predictions side-by-side with the ground truth.

validation_generator = DataGenerator(partition['holdout'], labels, **params)

prediction = model.predict_generator(generator=validation_generator)
# print('Test loss:', score[0])
# print('Test accuracy:', score[1])
prediction


#%%
# # sanity check on the predictions:
# len(prediction)
# prediction[0].shape
# np.unique(prediction)


#%%
# ! mkdir to_categorical_try
# ! mkdir channel_split


#%%
# import pickle

# pickle.dump( partition, open( "./channel_split/partition.pkl", "wb" ) ) # this has the test/train ID matches

# # # access the test list:
# # testIDlist = partition['test']
# # testIDlist


#%%
# for i in range(len(prediction)):
#     pickle.dump( prediction[i], open( "./channel_split/prediction_"+str(i)+".pkl", "wb" ) )


#%%
# # https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model
# # pickle.dump( model, open( "model.pkl", "wb" ) )
# model.save_weights('./channel_split/my_model_weights.h5')


#%%
import pickle

partition = pickle.load(open( "./channel_split/partition.pkl", "rb" ) ) # this has the test/train ID matches

# access the test list:
testIDlist = partition['test']
# testIDlist


