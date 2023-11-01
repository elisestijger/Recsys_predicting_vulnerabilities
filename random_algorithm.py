import pandas as pd
import numpy as np 
import random
from array import array
from lenskit.algorithms.basic import Random
from lenskit import batch, topn, util, topn


def do_random(data, n):
    unique_users = data['user_id'].unique()
    unique_items = data['item_id'].unique()
    all_recommendations = []
    for i in unique_users:  # i is a string 
        recommendations = []
        m = n 
        while m > 0:
            recommendations.append(random_single_users(i, unique_items))
            m = m - 1
        all_recommendations.append(recommendations)
    return all_recommendations

def random_single_users(i, unique_items):
    return random.choice(unique_items)


def test_random(data, n):
    data = data[['user_id', 'item_id', 'rating']]
    data = data.rename(columns={'user_id': 'user', 'item_id': 'item'} )
    model = Random()
    model.fit(data)
    users = data.user.unique()
    recs = batch.recommend(model, users, n)
    
    return recs