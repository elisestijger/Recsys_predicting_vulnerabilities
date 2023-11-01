import os
import pickle
import sys
from time import gmtime
from time import strftime
import numpy as np
from informative_diverse import InformativeClusterDiverseSampler
import random
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit.algorithms import Recommender, als, user_knn as knn_user
from lenskit import batch, topn, util, topn
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import multiprocessing
from sklearn.metrics.pairwise import cosine_similarity
from multiprocessing import Process
import random
import joblib
from lenskit.algorithms import Recommender, als, item_knn as knn


def activelearning(folds, unique_items):  
    # first random : pick train data -> remove one user. Then pick randomly one item / check rating and add to interaction sequence
# STARTS HERE   
    dict_recommendations_all = {}
    list_recs_without_user = []
    users_removed = ['426', '1674', '1512', '3357', '1674']
    p = 0 
    for i in folds:
        dict_recommendations = {}
        model1 =  knn.ItemItem(10, feedback='implicit')  
        model2 =  knn.ItemItem(10, feedback='implicit')  

        test = folds[i]['validation'][['item_id', 'user_id', 'rating']]
        train = folds[i]['train'][['item_id', 'user_id', 'rating']]
        
        test.rename(columns={'item_id': 'item', 'user_id': 'user'}, inplace=True)
        train.rename(columns={'item_id': 'item', 'user_id': 'user'}, inplace=True)
        user_counts = train['user'].value_counts()
        most_common_users = user_counts.nlargest(10)
        most_common_users_list = most_common_users.index.tolist()

# remove one user for train (before data)
        user_to_remove = users_removed[p]     #random.choice(most_common_users_list)
        p = p + 1
        deleted_rows = train[train['user'] == user_to_remove]
        train_without_user = train[train['user'] != user_to_remove]
        
        training_size = len(train_without_user)
        model1 = Recommender.adapt(model1)
        model1.fit(train_without_user)

        users = test.user.unique()
        recs_without_user = batch.recommend(model1, users, 5)
        list_recs_without_user.append(recs_without_user)

        key_size = 'trainingsize' 
        dict_recommendations[key_size] = training_size
        key = 'user_removed'  
        dict_recommendations[key] = user_to_remove
        key = 'recommendation original' 
        dict_recommendations[key] = recs_without_user
        
        unique_items2 = unique_items
        selected_items = []


#  RANDOM PER ADD 

        # train_without_user2 = train_without_user
        # list_recommendations_per_add = []
        # for j in range(0,20):        
        #     random_index = random.randint(0, len(unique_items2) - 1)
        #     random_item = unique_items2[random_index]
        #     if random_item in list(deleted_rows['item']):
        #         train_without_user2 = train_without_user2.append(pd.Series([random_item, user_to_remove,1.0], index=train_without_user.columns), ignore_index=True)
        #     model2 = Recommender.adapt(model2)
        #     model2.fit(train_without_user2)
        #     users = test.user.unique()
        #     recs_without_user2 = batch.recommend(model2, users, 5)
        #     list_recommendations_per_add.append(recs_without_user2)
        
        # key = 'recommendation after active learning' 
        # dict_recommendations[key] = list_recommendations_per_add

        # key = 'test' 
        # dict_recommendations[key] = test

# RANDOM NORMAL / VOOR ETEN 

        # train_without_user2 = train_without_user
        # for j in range(0,20):        
        #     random_index = random.randint(0, len(unique_items2) - 1)
        #     random_item = unique_items2[random_index]
        #     if random_item in list(deleted_rows['item']):
        #          train_without_user2 = train_without_user2.append(pd.Series([random_item, user_to_remove,1.0], index=train_without_user.columns), ignore_index=True)
        
        # model2 = Recommender.adapt(model2)
        # model2.fit(train_without_user2)
        # users = test.user.unique()
        # recs_without_user2 = batch.recommend(model2, users, 5)
        
        # key = 'recommendation after active learning' 
        # dict_recommendations[key] = recs_without_user2

        # key = 'test' 
        # dict_recommendations[key] = test

# SAMPLE 40 ITEMS ALL ITEMS CONSIDERED -> PADD  -> joblib.dump(NAME, 'sample_allitems_40added_padd.pkl')
        
        list_with_items = samplingmethod(train_without_user, model2, deleted_rows, unique_items, test)
        sorted_keys = [key for key, value in sorted(list_with_items.items(), key=lambda item: item[1], reverse=True)]
        train_without_user2 = train_without_user
        list_recommendations_per_add = []

        for k in sorted_keys[:40]:    
            if k in list(deleted_rows['item']):
                train_without_user2 = train_without_user2.append(pd.Series([k, user_to_remove,1.0], index=train_without_user.columns), ignore_index=True)
            model2 = Recommender.adapt(model2)
            model2.fit(train_without_user2)
            users = test.user.unique()
            recs_without_user2 = batch.recommend(model2, users, 5)
                
            list_recommendations_per_add.append(recs_without_user2)
            
        key = 'recommendation after active learning' 
        dict_recommendations[key] = list_recommendations_per_add

        key = 'test' 
        dict_recommendations[key] = test

# SAMPLE 20 ITEMS ALL ITEMS CONSIDERED NOW -> PADD -> joblib.dump(NAME, 'sample_allitems_20added_padd.pkl')

        # list_with_items = samplingmethod(train_without_user, model2, deleted_rows, unique_items, test)
        # sorted_keys = [key for key, value in sorted(list_with_items.items(), key=lambda item: item[1], reverse=True)]
        # train_without_user2 = train_without_user
        # list_recommendations_per_add = []

        # for k in sorted_keys[:20]:    
        #     if k in list(deleted_rows['item']):
        #         train_without_user2 = train_without_user2.append(pd.Series([k, user_to_remove,1.0], index=train_without_user.columns), ignore_index=True)
        #     model2 = Recommender.adapt(model2)
        #     model2.fit(train_without_user2)
        #     users = test.user.unique()
        #     recs_without_user2 = batch.recommend(model2, users, 5)
                
        #     list_recommendations_per_add.append(recs_without_user2)
            
        # key = 'recommendation after active learning' 
        # dict_recommendations[key] = list_recommendations_per_add

        # key = 'test' 
        # dict_recommendations[key] = test


# # 4 BACTCHES SAMPLE ALL ITEMS CONSIDERED NOW -> PADD -> joblib.dump(NAME, 'sample_4batches_allitems_padd.pkl')

#         list_recommendations = []
#         for b in range(4):
#             # list_random_items = []
#             # unique_items3 = unique_items2
#             list_with_items = samplingmethod(train_without_user, model2, deleted_rows, unique_items2, test)
#             sorted_keys = [key for key, value in sorted(list_with_items.items(), key=lambda item: item[1], reverse=True)]
#             for k in sorted_keys[:5]:    
#                 # delete this item from the unique_items2
#                 unique_items2 = [x for x in unique_items2 if x != k]

#                 if k in list(deleted_rows['item']):
#                     train_without_user2 = train_without_user2.append(pd.Series([k, user_to_remove,1.0], index=train_without_user.columns), ignore_index=True)
                                
#                 model2 = Recommender.adapt(model2)
#                 model2.fit(train_without_user2)
#                 users = test.user.unique()
#                 recs_without_user2 = batch.recommend(model2, users, 5)

#                 list_recommendations.append(recs_without_user2)             # add recoms to list  times after adding 5 items. 

#         key = 'recommendation after active learning' 
#         dict_recommendations[key] = list_recommendations

#         key = 'test' 
#         dict_recommendations[key] = test

# 10 BACTCHES SAMPLE ALL ITEMS CONSIDERED NOW -> PADD -> joblib.dump(NAME, 'sample_10batches_allitems_padd.pkl')

        # list_recommendations = []
        # for b in range(10):
        #     list_with_items = samplingmethod(train_without_user, model2, deleted_rows, unique_items2, test)
        #     sorted_keys = [key for key, value in sorted(list_with_items.items(), key=lambda item: item[1], reverse=True)]
        #     for k in sorted_keys[:2]:    
        #         # delete this item from the unique_items2
        #         unique_items2 = [x for x in unique_items2 if x != k]

        #         if k in list(deleted_rows['item']):
        #             train_without_user2 = train_without_user2.append(pd.Series([k, user_to_remove,1.0], index=train_without_user.columns), ignore_index=True)
                                
        #         model2 = Recommender.adapt(model2)
        #         model2.fit(train_without_user2)
        #         users = test.user.unique()
        #         recs_without_user2 = batch.recommend(model2, users, 5)

        #         list_recommendations.append(recs_without_user2)             # add recoms to list  times after adding 5 items. 

        # key = 'recommendation after active learning' 
        # dict_recommendations[key] = list_recommendations

        # key = 'test' 
        # dict_recommendations[key] = test


        key2 = 'fold' + str(i)
        dict_recommendations_all[key2] = dict_recommendations

    return dict_recommendations_all 

def samplingmethod(data, model, deleted_rows,unique_items,test):

    count_per_item = {}       

    for i in unique_items:
        train_with = data
        train_with = data.append(pd.Series([i, deleted_rows['user'].iloc[0], 1.0], index=data.columns), ignore_index=True)
        model = Recommender.adapt(model)
        model.fit(train_with)
        users = test.user.unique()
        recs_with_user = batch.recommend(model, users, 5)

        test_data = test
        merged_df = pd.merge(recs_with_user, test_data, on='user', suffixes=('_df1', '_df2'))
        matching_items = merged_df[merged_df['item_df1'] == merged_df['item_df2']]

        count_per_item[i] = len(matching_items)


    return count_per_item





