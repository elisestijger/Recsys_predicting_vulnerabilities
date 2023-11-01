import numpy as np
from lenskit import crossfold as xf
from collections import defaultdict
from lenskit_modules import do_recommendations
from lenskit import batch, topn, util, topn
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit.algorithms import Recommender, als, user_knn as knn_user
import pandas as pd
import numpy as np

def start_cross_lenskit(data, n,m):
    algo_ii = knn.ItemItem(10, feedback='implicit')  #20
    algo_uu = knn_user.UserUser(10, feedback='implicit')

    def eval(aname, algo, train, test, n):
        fittable = util.clone(algo)
        fittable = Recommender.adapt(fittable)
        fittable.fit(train)
        users = test.user.unique()
        recs = batch.recommend(fittable, users, n)
        recs['Algorithm'] = aname
        return recs
    
    all_recs = []
    test_data = []

    data = data[['user_id', 'item_id', 'rating']]
    data = data.rename(columns={'user_id': 'user', 'item_id': 'item'} )
    
    
    for train, test in xf.partition_users(data[['user', 'item', 'rating']], m, xf.SampleFrac(0.2)):
        test_data.append(test)
        all_recs.append(eval('ItemItem', algo_ii, train, test, n))
        all_recs.append(eval('UserUser', algo_uu, train, test, n))


    all_recs = pd.concat(all_recs, ignore_index=True)
    all_recs.head()

    test_data = pd.concat(test_data, ignore_index=True)
    test_data.head()

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(all_recs, test_data)
    results.head()
    outcome = results.groupby('Algorithm').ndcg.mean()


    return all_recs, results, outcome
