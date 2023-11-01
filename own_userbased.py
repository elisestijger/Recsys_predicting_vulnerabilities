import numpy as np
from lenskit import crossfold as xf
from collections import defaultdict
from lenskit_modules import do_recommendations
from lenskit import batch, topn, util, topn
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit.algorithms import Recommender, als, user_knn as knn_user
import pandas as pd
import numpy as np

def __user_based_crossfold_validation_split():

    # generate splits of equal size
    splits = np.array_split(data, num_folds)
    print(type(splits))
    # splits = the data -> every user is equally distributed over the folds
    # print('should be different numbers for different amount of items',splits)
    # go through each split
    for i in range(len(splits)):
        # the split denoted by i is the test set, so all other splits are the train set
        print(type(splits[i + 1:]))
        train = pd.concat(splits[:i] + splits[i + 1:], axis=0, ignore_index=False)
        # the test data is simply the index we are currently observing
        test = splits[i]
        # print("test", splits[i])
        # append the indices to the dictionary // append all the indices of these train and test data 
        fold_indices[i]["train"] = np.append(fold_indices[i]["train"], train.index  )  #.index)
        fold_indices[i]["validation"] = np.append(fold_indices[i]["validation"], test.index ) #  .index)

    # test_folds = user_based_validation_split(interaction_sequence, 5)

    # print(fold_indices)
    return fold_indices


def own_userbased(data: pd.DataFrame, num_folds: int = 5, random_state=42) -> dict:
    # Define the number of folds and create an empty fold_indices dictionary
    fold_indices = {i: {"train": np.array([]), "validation": np.array([])} for i in range(num_folds)}
    unique_users = data['user_id'].unique()

    for user, items in data.groupby("user_id"):
        fold_indices = __user_based_crossfold_validation_split(fold_indices=fold_indices,
                                                                   data=items,
                                                                   num_folds=num_folds)

        # print(data[user])

    print(type(data))

    # generate splits of size of unique users. 
    splits = np.array_split(data, num_folds)
    
    print(type(splits))
    for i in range(len(splits)):

        # the split denoted by i is the test set, so all other splits are the train set
        train = pd.concat(splits[:i] + splits[i + 1:], axis=0, ignore_index=False)
        # the test data is simply the index we are currently observing
        test = splits[i]
        # print("test", splits[i])
        # append the indices to the dictionary // append all the indices of these train and test data 
        fold_indices[i]["train"] = np.append(fold_indices[i]["train"], train.index  )  #.index)
        fold_indices[i]["validation"] = np.append(fold_indices[i]["validation"], test.index ) #  .index)






    # # Split users into groups of 5 for training and 1 for validation
    # user_groups = [unique_users[i:i+5] for i in range(0, len(unique_users), 6)]
    # print(user_groups)
    # # print(user_groups)
    # # Create the folds
    # for fold_index, user_group in enumerate(user_groups):
    #     for user in user_group:
    #         # user_group = list(user_group)
    #         # user_groups = list(user_groups)
    #         # print(user_group)
    #         for i in range(len(user_group)):
    #             print(user_group[i], user_group[i + 1:])
    #             # the split denoted by i is the test set, so all other splits are the train set
    #             train = pd.concat(user_group[:i] + user_group[i + 1:], axis=0, ignore_index=False)
    #             # the test data is simply the index we are currently observing
    #             test = user_group[i]

    #         fold_indices[i]["train"] = np.append(fold_indices[i]["train"], train.index.index  )  
    #         fold_indices[i]["validation"] = np.append(fold_indices[i]["validation"], test.index ) 


    return fold_indices

