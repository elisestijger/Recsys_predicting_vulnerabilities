import pandas as pd
import numpy as np


# def validation_split(data: pd.DataFrame, strategie: str = 'user_based', num_folds: int = 1,
#                      frac: float = 0.25, random_state=42) -> dict:

#     # decide which validation split strategie to use
#     if strategie == 'user_based':
#         return user_based_validation_split(data=data, num_folds=num_folds, frac=frac, random_state=random_state)
#     elif strategie == 'row_based':
#         return row_based_validation_split(data=data, num_folds=num_folds, frac=frac, random_state=random_state)
#     else:
#         raise ValueError(f"Unknown validation split strategie: {strategie}")


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
    for user, items in data.groupby("user"):
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
    # go through each split
    for i in range(len(splits)):
        # the split denoted by i is the test set, so all other splits are the train set
        train = pd.concat(splits[:i] + splits[i + 1:], axis=0, ignore_index=False)
        # the test data is simply the index we are currently observing
        test = splits[i]
        # append the indices to the dictionary
        fold_indices[i]["train"] = np.append(fold_indices[i]["train"], train.index)
        fold_indices[i]["validation"] = np.append(fold_indices[i]["validation"], test.index)

    return fold_indices

# def cross_validation(data, n):
#     #n = folds 
#     sample_size = len(data) // n

#     # Get all unique products
#     unique_products = np.unique(data[0:1000]['Name_user'].to_numpy())
#     unique_items = np.unique(data[0:1000]['Name_item'].to_numpy())

#     # TEST with small dataset
#     unique_products = unique_products[0:10]
#     unique_items = unique_items[0:10]

#     data = data[0:1000]
#     folds = [pd.DataFrame(columns=data.columns) for _ in range(n)]
#     sample_size = len(data) // n

#     # Iterate over the unique products and add rows to folds
#     for index in unique_products:
#         product = index

#         # Iterate over the folds and add random rows where product from unique products is present. Also delete this row from df so no duplicate rows in the folds
#         for i, fold in enumerate(folds):
#             if len(fold) < sample_size:
#                 add_fold = data.query('Name_user == @product') #.sample(n=1)
#                 if len(add_fold) > 0:
#                     add_fold = add_fold.sample(n=1)
#                 fold = fold.append(add_fold)
#                 folds[i] = fold
#                 data = data.drop(add_fold.index)
#     for index in unique_items:
#         item = index

#         for i, fold in enumerate(folds):
#             if len(fold) < sample_size:
#                 add_fold = data.query('Name_item == @item') #.sample(n=1)
#                 if len(add_fold) > 0:
#                     add_fold = add_fold.sample(n=1)
#                 fold = fold.append(add_fold)
#                 folds[i] = fold
#                 data = data.drop(add_fold.index)

#     # Fill the rest of rows in the folds randomly
#     for i, fold in enumerate(folds):
#         remaining_rows = sample_size - len(fold)
#         if remaining_rows > 0:
#             filled_df = data.sample(n=remaining_rows, replace=False)
#             data = data.drop(filled_df.index)
#             fold = fold.append(filled_df, ignore_index=True)
#             folds[i] = fold

#     return folds 


# def cross_validation(data2, n):
#     data = data2[0:100]
#     # n_folds = n
#     # unique_users = data[0:100]['Name_user'].unique()
#     # unique_items = data[0:100]['Name_item'].unique()

#     # # Initialize empty folds
#     # folds = [pd.DataFrame(columns=data.columns) for _ in range(n_folds)]

#     n_folds = n
#     unique_users = data['Name_user'].unique()
#     unique_items = data['Name_item'].unique()

#     n_unique_users = len(unique_users)
#     n_unique_items = len(unique_items)

#     # Calculate the number of rows per fold for users and items
#     rows_per_fold_users = n_unique_users // n_folds
#     rows_per_fold_items = n_unique_items // n_folds

#     # Initialize empty folds
#     folds = [pd.DataFrame(columns=data.columns) for _ in range(n_folds)]

#     # Distribute users evenly to each fold
#     for i in range(n_folds):
#         fold_users = unique_users[i * rows_per_fold_users : (i + 1) * rows_per_fold_users]
#         fold_data_users = data[data['Name_user'].isin(fold_users)]
#         folds[i] = pd.concat([folds[i], fold_data_users])

#     # Distribute items evenly to each fold
#     for i in range(n_folds):
#         fold_items = unique_items[i * rows_per_fold_items : (i + 1) * rows_per_fold_items]
#         fold_data_items = data[data['Name_item'].isin(fold_items)]
#         folds[i] = pd.concat([folds[i], fold_data_items])

#     # Optional: Shuffle the rows within each fold for randomness
#     for i in range(n_folds):
#         folds[i] = folds[i].sample(frac=1).reset_index(drop=True)

#     # Access the folds as folds[0], folds[1], ..., folds[4]

#     # Calculate the number of duplicate users and items in each fold
#     duplicate_users_count = [len(folds[i]['Name_user'].duplicated()) for i in range(n_folds)]
#     duplicate_items_count = [len(folds[i]['Name_item'].duplicated()) for i in range(n_folds)]

#     print("Duplicate Users Count in Each Fold:", duplicate_users_count)
#     print("Duplicate Items Count in Each Fold:", duplicate_items_count)


#     return folds
