#%%
import pickle
import numpy as np
import nibabel as nib
import pandas as pd
import os
from unet_utils import crop_img, data_dir

def main():
    """
    This function requires that you've first moved the images from inside each folder within HGG and LGG to a separate directory called "data".
    Combines the HGG and LGG data and creates a dictionary of their labels called "tumor_type_dict".
    Creates a train, test, holdout split and stores the train, test, and holdout assignments in a dictionary called "train_val_test_dict".
    """
    # ### Make the labels and test train dictionaries:

    # make tumor type dictionary:
    tumor_type_dict = {}

    HGG_dir_list = next(os.walk('./HGG/'))[1]
    # print(len(HGG_dir_list))
    LGG_dir_list = next(os.walk('./LGG/'))[1]
    # print(len(LGG_dir_list))

    for patientID in HGG_dir_list+LGG_dir_list:
    #     print(patientID)
        if patientID in HGG_dir_list:
    #         tumor_type_dict[patientID] = "HGG"
            tumor_type_dict[patientID] = 0
        elif patientID in LGG_dir_list:
    #         tumor_type_dict[patientID] = "LGG"
            tumor_type_dict[patientID] = 1

    # print(len(tumor_type_dict))

    completelist = HGG_dir_list + LGG_dir_list

    for patientID in completelist:
        print(tumor_type_dict[patientID])
    # tumor_type_dict[(HGG_dir_list+LGG_dir_list)[0]]

    
    # print(completelist[0:4])
    np.random.shuffle(completelist) # shuffles in place
    # print(completelist[0:4])

    train_val_test_dict={}

    holdout_percentage=0.15
    train_val_test_dict['test']=completelist[0:int(len(completelist)*holdout_percentage)]
    trainlist=completelist[int(len(completelist)*holdout_percentage):len(completelist)]

    train_percentage=0.7
    train_val_test_dict['train']=trainlist[0:int(len(trainlist)*train_percentage)]
    train_val_test_dict['val']=trainlist[int(len(trainlist)*train_percentage):len(trainlist)]

    pickle.dump( tumor_type_dict, open( "tumor_type_dict.pkl", "wb" ) )
    pickle.dump( train_val_test_dict, open( "train_val_test_dict.pkl", "wb" ) )

    for i, ID in enumerate(completelist):

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

    #     X[i,] = images
    #     y1[i,] = seg_mask_3ch
        pickle.dump( images, open( data_dir / f"{ID}_images.pkl", "wb" ) )
        pickle.dump( seg_mask_3ch, open( data_dir / f"{ID}_seg_mask_3ch.pkl", "wb" ) )
        print("Saving", i+1, "of", len(completelist))

if __name__ == "__main__":
    main()

#%%
