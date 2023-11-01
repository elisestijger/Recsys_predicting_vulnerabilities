import pandas as pd 
import numpy as np
from lenskit.crossfold import partition_rows


def row_based_validation_split(data: pd.DataFrame, num_folds: int = 1, frac: float = 0.25, random_state=42) -> dict:
 
    # initialize a dictionary with the indices of the train and validation split for the given data
    fold_indices = {i: {"train": np.array([]), "validation": np.array([])} for i in
                    range(num_folds)}
    # if num_folds < 2, we use a holdout validation split
    if num_folds < 2:
        fold_indices = __holdout_validation_split(fold_indices=fold_indices,
                                                  data=data,
                                                  frac=frac,
                                                  random_state=random_state)
    # if num_folds > 1, we use a cross validation split
    else:
        fold_indices = __row_based_k_fold_validation_split(fold_indices=fold_indices,
                                                           data=data,
                                                           num_folds=num_folds,
                                                           random_state=random_state)
    return fold_indices


def user_based_validation_split(data: pd.DataFrame, num_folds: int = 1, frac: float = 0.25, random_state=42) -> dict:

    # initialize a dictionary with the indices of the train and validation split for the given data
    fold_indices = {i: {"train": np.array([]), "validation": np.array([])} for i in
                    range(num_folds)}

    # group by users and then sample from each user
    for user, items in data.groupby("Name_user"):
        # if num_folds < 2, we use a holdout validation split
        if num_folds < 2:
            fold_indices = __holdout_validation_split(fold_indices=fold_indices,
                                                      data=items,
                                                      random_state=random_state,
                                                      frac=frac)
        # if num_folds > 1, we use a cross validation split
        else:
            fold_indices = __user_based_crossfold_validation_split(fold_indices=fold_indices,
                                                                   data=items,
                                                                   num_folds=num_folds)

    return fold_indices


def __holdout_validation_split(fold_indices: dict, data: pd.DataFrame, frac: float, random_state=42):

    # sample the validation set
    validation = data.sample(frac=frac, random_state=random_state)
    # get the train set by dropping the validation set
    train = data.drop(validation.index)
    # append the indices of the train and validation set to the dictionary
    fold_indices[0]['train'] = np.append(fold_indices[0]["train"], train.index)
    fold_indices[0]['validation'] = np.append(fold_indices[0]["validation"], validation.index)
    # return the dictionary
    return fold_indices


def __row_based_k_fold_validation_split(fold_indices: dict, data: pd.DataFrame, num_folds: int, random_state):

    # generate the indices of the train and validation split for the given data
    for i, splits in enumerate(partition_rows(data, partitions=num_folds, rng_spec=random_state)):
        fold_indices[i]['train'] = np.append(fold_indices[i]["train"], splits[0].index)
        fold_indices[i]['validation'] = np.append(fold_indices[i]["train"], splits[1].index)
    return fold_indices


def __user_based_crossfold_validation_split(fold_indices, data, num_folds) -> dict:

    # generate splits of equal size
    splits = np.array_split(data, num_folds)
    for i in range(len(splits)):
        # the split denoted by i is the test set, so all other splits are the train set
        train = pd.concat(splits[:i] + splits[i + 1:], axis=0, ignore_index=False)
        # the test data is simply the index we are currently observing
        test = splits[i]
        fold_indices[i]["train"] = np.append(fold_indices[i]["train"], train.index  )  
        fold_indices[i]["validation"] = np.append(fold_indices[i]["validation"], test.index ) 
    return fold_indices
