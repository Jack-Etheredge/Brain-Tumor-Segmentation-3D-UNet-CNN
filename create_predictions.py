import pickle
import numpy as np
import copy
import nibabel as nib
from tifffile import imsave
from unet_utils import crop_img, weights_dir, data_dir, pred_dir, DataGenerator, create_model
from keras.models import load_model
from keras import backend as K
from keras.engine import Model
K.tensorflow_backend.set_image_dim_ordering('tf')
K.set_image_data_format('channels_first')

# doesn't look like I kept using these:
# from libtiff import TIFF
# from skimage.io._plugins import freeimage_plugin as fi

def predict_unet(num_outputs, load_weights_filepath):
    train_val_test_dict = pickle.load(open( "train_val_test_dict.pkl", "rb" ) ) # this has the test/train ID matches

#     pickle.dump( results.history, open( weights_dir / f"history_{num_outputs}_pred.pkl", "wb" ) )

    # This isn't an ideal way to do things. 
    # I need to find a simpler way around the issue of having custom layers (ValueError: Unknown layer: InstanceNormalization)
    model = create_model(input_shape=(4, 160, 192, 160),
        n_base_filters=12,
        depth=5,
        dropout_rate=0.3,
        n_segmentation_levels=3,
        n_labels=3,
        num_outputs=num_outputs,
        optimizer='adam',
        learning_rate=1e-2,
        activation_name="sigmoid")

    model.load_weights(load_weights_filepath, by_name=True) # by_name=True allows you to use a different architecture and bring in the weights from the matching layers 
    
    # Turned shuffle off so that we can match the values in the dictionary to the predictions. 
    # This way we can compare the predictions side-by-side with the ground truth.

    params = {'dim': (160,192,160),
        'batch_size': 1,
        'n_classes': 3,
        'n_channels': 4,
        'shuffle': False,
        'num_outputs': num_outputs}
    validation_generator = DataGenerator(train_val_test_dict['test'], **params)
 
#     # load model
#     model = load_model(model_path, custom_objects={'InstanceNormalization':unet_utils.InstanceNormalization})
#     keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

    predictions = model.predict_generator(generator=validation_generator)

    for i, prediction in enumerate(predictions):
        pickle.dump( prediction, open( pred_dir / f"predictions_{num_outputs}_pred_{i}.pkl", "wb" ) )


def create_tiffs_from_predictions(num_outputs):
    train_val_test_dict = pickle.load(open( "train_val_test_dict.pkl", "rb" ) ) # this has the test/train ID matches

    # access the test list:
    testIDlist = train_val_test_dict['test']

    # ID = testIDlist[i]
    for i in range(len(testIDlist)):
#     for i in range(2):

        print("current image:", i)

        ID = testIDlist[i]
        img1 = data_dir / f'{ID}_flair.nii.gz'
        img2 = data_dir / f'{ID}_t1.nii.gz'
        img3 = data_dir / f'{ID}_t1ce.nii.gz'
        img4 = data_dir / f'{ID}_t2.nii.gz'
        img5 = data_dir / f'{ID}_seg.nii.gz'

        img_list = [str(x) for x in [img1, img2, img3, img4, img5]]
        newimage = nib.concat_images(img_list)
        # # nibabel uses .lower on the filepath, which requires the filepath to be string, not the posixpath type from pathlib
        # newimage = nib.concat_images([img1, img2, img3, img4, img5])
        
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
        #     [a, seg_mask_1, seg_mask_2, seg_mask_3], axis=0).astype('uint8')

        #     print("unique flair + segment mask values:", np.unique(ground_truth))
        # shape.ground_truth
        # flairimg = flairimg.astype(float)
        # flairimg *= 255.0/flairimg.max() # convert to 8-bit pixel values
        # flairimg = flairimg.astype(int)
        # print(np.unique(flairimg))
        #     print("final image shape:", ground_truth.shape)
        #     imsave("./channel_split/"+testIDlist[i]+"ground_truth.tif", ground_truth, 'imagej')
        imsave(pred_dir / f"{ID}_flair.tif", a, 'imagej')
        imsave(pred_dir / f"{ID}_ground_truth_1.tif", seg_mask_1, 'imagej')
        imsave(pred_dir / f"{ID}_ground_truth_2.tif", seg_mask_2, 'imagej')
        imsave(pred_dir / f"{ID}_ground_truth_3.tif", seg_mask_3, 'imagej')

        imarray = pickle.load(open( pred_dir / f"predictions_{num_outputs}_pred_{i}.pkl", "rb" ) )

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

        imsave(pred_dir / f"{ID}_predicted_1.tif", pred1, 'imagej')
        imsave(pred_dir / f"{ID}_predicted_2.tif", pred2, 'imagej')
        imsave(pred_dir / f"{ID}_predicted_3.tif", pred3, 'imagej')

        # print("images shape:", images.shape, "images values:", np.unique(images.astype(int)))

        # split the channels:
        # seg_mask_1 = copy.deepcopy(seg_mask.astype(int))


        #     seg_mask_3ch = np.concatenate(
        #     [seg_mask_1, seg_mask_2, seg_mask_3], axis=0).astype(int)

        #     imarray *= 255.0/imarray.max()

        #     imsave("./channel_split/"+testIDlist[i]+"ground_truth_1.tif", seg_mask_1, 'imagej')
        #     imsave("./channel_split/"+testIDlist[i]+"ground_truth_2.tif", seg_mask_2, 'imagej')
        #     imsave("./channel_split/"+testIDlist[i]+"ground_truth_3.tif", seg_mask_3, 'imagej')

if __name__ == "__main__":
        num_outputs = 1
        predict_unet(num_outputs, './weights/model_weights_3_outputs.h5')
        create_tiffs_from_predictions(num_outputs)