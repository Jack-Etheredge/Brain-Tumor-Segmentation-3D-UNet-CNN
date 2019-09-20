#%% [markdown]
# ### Make the labels and test train dictionaries:

#%%
# from glob import glob
# paths = glob('/Users/etheredgej/Desktop/MICCAI_BraTS17_Data_Training/train/HGG/*/')
# print(paths)

import os
HGG_dir_list = next(os.walk('./HGG/'))[1]
print(len(HGG_dir_list))
LGG_dir_list = next(os.walk('./LGG/'))[1]
print(len(LGG_dir_list))

#%% [markdown]
# ### Dictionary for all samples:

#%%
completelist = HGG_dir_list + LGG_dir_list


#%%
completelist = HGG_dir_list + LGG_dir_list

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
    
print(len(partition['holdout']))
print(len(partition['train']))
print(len(partition['test']))

#%% [markdown]
# ### Dictionary for the samples with survival data:

#%%
# completelist = HGG_dir_list + LGG_dir_list

completelist = list(survival_data.Brats17ID.copy())

# print(completelist[0:4])
np.random.shuffle(completelist) # shuffles in place
# print(completelist[0:4])

subpartition={}

holdout_percentage=0.15
subpartition['holdout']=completelist[0:int(len(completelist)*holdout_percentage)]
trainlist=completelist[int(len(completelist)*holdout_percentage):len(completelist)]

train_percentage=0.7
subpartition['train']=trainlist[0:int(len(trainlist)*train_percentage)]
subpartition['test']=trainlist[int(len(trainlist)*train_percentage):len(trainlist)]


labels={}
# HGG=0
# LGG=1
for directory in HGG_dir_list:
    labels[directory]=0
for directory in LGG_dir_list:
    labels[directory]=1
    
print(len(subpartition['holdout']))
print(len(subpartition['train']))
print(len(subpartition['test']))


#%%
# len(completelist)
# len(survival_data.Brats17ID)
# len(set.intersection(set(completelist),set(survival_data.Brats17ID)))