import pandas as pd
import numpy as np 
import random
from array import array
from basic import Popular
from lenskit import batch, topn, util, topn


def do_pop(data, n):
    unique_users = data['user_id'].unique()
    items = data['item_id']
    most_frequent_value = items.value_counts().idxmax()
    all_recommendations = []
    for i in unique_users:  # i is a string 
        recommendations = []
        m = n 
        while m > 0:
            recommendations.append(most_frequent_value)
            m = m - 1
        all_recommendations.append(recommendations)
    return all_recommendations

# def pop_single_users(i, unique_items):
#     unique_items, counts = np.unique(unique_items, return_counts=True)
#     most_frequent_item = unique_items[np.argmax(counts)]
#     print(unique_items)
#     # print(most_frequent_item)
#     return most_frequent_item

def test_pop(data, n):
    data = data[['user_id', 'item_id', 'rating']]
    data = data.rename(columns={'user_id': 'user', 'item_id': 'item'} )
    model = Popular()
    model.fit(data)
    users = data.user.unique()
    recs = batch.recommend(model, users, n)
    
    return recs