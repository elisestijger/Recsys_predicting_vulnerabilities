import pandas as pd 
import numpy as np
from lenskit_modules import do_recommendations
from lenskit.crossfold import partition_rows


def random_cross_fold(cve_all_r_filtered, n):
    # Data for cross validation 
    copy_cve_all_r_filtered = cve_all_r_filtered

    # Calculate the sample size for each fold
    sample_size = len(copy_cve_all_r_filtered) // n

    # Get all unique products
    # unique_products = np.unique(copy_cve_all_r_filtered['product'].to_numpy())

    # Create n empty folds
    folds = [pd.DataFrame(columns=copy_cve_all_r_filtered.columns) for _ in range(n)]
    # sample_size = len(copy_cve_all_r_filtered) // n

    # Fill the rows in the folds randomly/ first randomly spread the rows of the df 
    copy_cve_all_r_filtered = copy_cve_all_r_filtered.sample(frac=1).reset_index(drop=True)
    for i, fold in enumerate(folds):
            filled_df = copy_cve_all_r_filtered.sample(n=sample_size, replace=False)
            copy_cve_all_r_filtered = copy_cve_all_r_filtered.drop(filled_df.index)
            fold = fold.append(filled_df, ignore_index=True)
            folds[i] = fold


    # Assuming you have a list of 5 dataframes named 'dataframes'
    dataframes_dict = {}

    # Create an empty dataframe to store the concatenated "train" dataframes
    # all_train_df = pd.DataFrame()

    for i, df in enumerate(folds):
        validation_df = df
        train_dfs = [dataframe for j, dataframe in enumerate(folds) if j != i]

        # Concatenate all "train" dataframes into a single dataframe
        train_concatenated = pd.concat(train_dfs)

        dataframes_dict[i] = {'train': train_concatenated, 'validation': validation_df}

    # 'all_train_df' will now contain the concatenated "train" dataframes


    return dataframes_dict 

# def start_random_cross(interaction_sequence, n, m):       # m = recom

#     # Make the folds with user based cross validation with n folds
#     test_folds = random_cross(interaction_sequence, n)

#     results = do_recommendations(test_folds, n=m)
#     itemitem_outcome = results[0]
#     results_ngcd = results[1]
#     outcome_ngcd =  results[2]
    
#     return results, itemitem_outcome, results_ngcd, outcome_ngcd


# def __row_based_k_fold_validation_split(fold_indices: dict, data: pd.DataFrame, num_folds: int, random_state):
#     splits = np.array_split(data, num_folds)

#     # generate the indices of the train and validation split for the given data
#     for i, splits in enumerate(partition_rows(data, partitions=num_folds, rng_spec=random_state)):
#         train = pd.concat(splits[:i] + splits[i + 1:], axis=0, ignore_index=False)
#         test = splits[i]

#         fold_indices[i]["train"] = np.append(fold_indices[i]["train"], train.index  )  
#         fold_indices[i]["validation"] = np.append(fold_indices[i]["validation"], test.index ) 
#     return fold_indices


