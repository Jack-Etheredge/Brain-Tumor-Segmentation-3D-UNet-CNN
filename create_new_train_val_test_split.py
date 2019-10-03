#%%
import pickle

def create_new_train_val_test_split(save=False, holdout_percentage=0.15, train_percentage=0.7):
    """
    This function requires that you've first moved the images from inside each folder within HGG and LGG to a separate directory called "data".
    Creates a train, test, holdout split and stores the train, test, and holdout assignments in a dictionary called "train_val_test_dict".
    Optionally, overwrites the original train_val_test_dict.pkl file, which will be done if run directly as a script.
    """

    tumor_type_dict = pickle.load(open( "tumor_type_dict.pkl", "rb" ) )
    completelist = list(tumor_type_dict.keys())

    np.random.shuffle(completelist) # shuffles in place

    train_val_test_dict={}

    holdout_percentage=holdout_percentage
    train_val_test_dict['test']=completelist[0:int(len(completelist)*holdout_percentage)]
    trainlist=completelist[int(len(completelist)*holdout_percentage):len(completelist)]

    train_percentage=train_percentage
    train_val_test_dict['train']=trainlist[0:int(len(trainlist)*train_percentage)]
    train_val_test_dict['val']=trainlist[int(len(trainlist)*train_percentage):len(trainlist)]

    if save==True:
        pickle.dump( train_val_test_dict, open( "train_val_test_dict.pkl", "wb" ) )

    return train_val_test_dict

if __name__ == "__main__":
    create_new_train_val_test_split(save=False)