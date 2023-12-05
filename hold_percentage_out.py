import numpy as np
from lenskit import crossfold as xf
from collections import defaultdict
from lenskit_modules import do_recommendations
from lenskit import batch, topn, util, topn
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit.algorithms import Recommender, als, user_knn as knn_user
import pandas as pd
import numpy as np


def hold_k_perc_out(data, frac):
    unique_users = data[0:100]['Name_user'].unique()
    print("len",len(unique_users))

    fold_indices = {i: {"train": np.array([]), "validation": np.array([])} for i in
                    range(len(unique_users))}
    fold_indices = hold_k_perc_out2(fold_indices, data[0:100], frac)

    return fold_indices

def hold_k_perc_out2(fold_indices: dict, data,  frac: float, random_state=42):         #fold_indices: dict, data: pd.DataFrame, frac: float, random_state=42
    unique_users = data['Name_user'].unique()

    # For each unique user, hold out all their items
    for user in unique_users:
        user_data = data[data['Name_user'] == user]
        # print(len(unique_users))
        
        # Sample the validation set based on frac
        validation = user_data.sample(frac=frac, random_state=random_state)
        
        # Append the selected rows (items) to the validation set
        fold_indices[0]['validation'] = np.append(fold_indices[0]["validation"], validation.index)
        
        # Exclude the selected rows (items) from the training set
        train = data.drop(validation.index)
        
        # Append the indices of the train set to the dictionary
        fold_indices[0]['train'] = np.append(fold_indices[0]["train"], train.index)
    
    return fold_indices

