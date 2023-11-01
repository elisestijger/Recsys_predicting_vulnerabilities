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


# def hold_k_perc_out(data,  m ):
#     for user, items in data.groupby("Name_user"):
#         train_data =  data[data['Name_user'] != user]
#         test_data = items


#     # save these folds in a dict 
#     filled_dict = {}

#     # # Iterate through each fold in the existing dictionary
#     for fold, data in test_data.items():
#         filled_fold = {}
#         for key, indices in data.items():
#             # Use the indices to retrieve values from the DataFrame
#             values = data.iloc[indices]
#             filled_fold[key] = values
#         filled_dict[fold] = filled_fold

#     # Do item item and user user knn and return the results 
#     results = do_recommendations(filled_dict, n=m)
#     itemitem_outcome = results[0]
#     results_ngcd = results[1]
#     outcome_ngcd =  results[2]
    
#     # return results, itemitem_outcome, results_ngcd, outcome_ngcd


#     for user, items in data.groupby("Name_user"):
#             train_data =  data[data['Name_user'] != user]
#             test_data = items

#     algo_ii = knn.ItemItem(10, feedback='implicit')  #20
#     algo_uu = knn_user.UserUser(10, feedback='implicit')
#     # algo_als = als.BiasedMF(50)

#     def eval(aname, algo, train, test, n):
#         fittable = util.clone(algo)
#         fittable = Recommender.adapt(fittable)
#         fittable.fit(train)
#         users = test.user.unique()
#         recs = batch.recommend(fittable, users, n)
#         recs['Algorithm'] = aname
#         return recs
    

#     all_recs = []
#     test_data = []

#     data = data[['user_id', 'item_id', 'rating']]
#     data = data.rename(columns={'user_id': 'user', 'item_id': 'item'} )
    
    
#     for train, test in xf.partition_users(data[['user', 'item', 'rating']], m, xf.SampleFrac(0.2)):
#         test_data.append(test)
#         all_recs.append(eval('ItemItem', algo_ii, train, test, n))
#         all_recs.append(eval('UserUser', algo_uu, train, test, n))

#         # all_recs.append(eval('ALS', algo_als, train, test))


#     all_recs = pd.concat(all_recs, ignore_index=True)
#     all_recs.head()

#     test_data = pd.concat(test_data, ignore_index=True)
#     test_data.head()

#     rla = topn.RecListAnalysis()
#     rla.add_metric(topn.ndcg)
#     results = rla.compute(all_recs, test_data)
#     results.head()
#     outcome = results.groupby('Algorithm').ndcg.mean()


#     return all_recs, results, outcome
