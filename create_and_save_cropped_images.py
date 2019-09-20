#%%
import pickle
import numpy as np
import nibabel as nib
import os
from unet_utils import crop_img

def main():
    """
    Combines the HGG and LGG data and creates a dictionary of their labels.
    This function requires that you've first moved the images from inside each folder within HGG and LGG to a separate directory called "data".
    """
    #%% [markdown]
    # ### Make the labels and test train dictionaries:

    #%%
    # from glob import glob
    # paths = glob('/Users/etheredgej/Desktop/MICCAI_BraTS17_Data_Training/train/HGG/*/')
    # print(paths)

    # first move the 

    HGG_dir_list = next(os.walk('./HGG/'))[1]
    # print(len(HGG_dir_list))
    LGG_dir_list = next(os.walk('./LGG/'))[1]
    # print(len(LGG_dir_list))

    #%% [markdown]
    # ### Dictionary for all samples:

    #%%
    completelist = HGG_dir_list + LGG_dir_list


    #%%
    # completelist = HGG_dir_list + LGG_dir_list

    # completelist = list(survival_data.Brats17ID.copy())

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
        
    # print(len(partition['holdout']))
    # print(len(partition['train']))
    # print(len(partition['test']))

    pickle.dump( labels, open( "./data/labels.pkl", "wb" ) )
    pickle.dump( partition, open( "./data/partition.pkl", "wb" ) )

    for i, ID in enumerate(completelist):

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

    #     X[i,] = images
    #     y1[i,] = seg_mask_3ch
        pickle.dump( images, open( "./data/%s_images.pkl"%(ID), "wb" ) )
        pickle.dump( seg_mask_3ch, open( "./data/%s_seg_mask_3ch.pkl"%(ID), "wb" ) )
        print("Saving", i+1, "of", len(completelist))

if __name__ == "__main__":
    main()