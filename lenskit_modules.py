from lenskit import batch, topn, util, topn
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit.algorithms import Recommender, als, user_knn as knn_user
from lenskit.algorithms import svd
from lenskit.algorithms import funksvd
# from lenskit.algorithms.basic import PopScore 
# from lenskit.algorithms import bias
# from lenskit.algorithms.mf_common import MFPredictor
# from lenskit.algorithms.als import BiasedMF, ImplicitMF
# from lenskit.algorithms.svd import BiasedSVD
# from lenskit.algorithms.basic import Bias, Popular, TopN
import pandas as pd
import numpy as np

def do_recommendations(data, n):
    # BEST OUTCOMES:
    algo_ii = knn.ItemItem(10, feedback='implicit')  #20
    algo_uu = knn_user.UserUser(10, feedback='implicit')

    # NOT REALLY GOOD OUTCOMES:
    # algo_als = als.BiasedMF(50)
    # algo_implicitmf = als.ImplicitMF(50)
    # algo_biasedsvd = svd.BiasedSVD(50)
    # algo_funksvd = funksvd.FunkSVD(50)
    # algo_tf = PopScore()   #gives weird outcomes
    # algo_bias = Bias()    #gives weird outcomes
    # algo_MFPredictor = MFPredictor()
    # Assuming you have already defined the `all_recs` and `test_data` lists as shown in your code

    def eval(aname, algo, train, test, n):
        fittable = util.clone(algo)
        fittable = Recommender.adapt(fittable)
        # Can also implement a topn recommender / or with PlackettLuce stochastic generator 
        # pred = item_knn.ItemItem(20, feedback='implicit')
        # select = UnratedItemCandidateSelector()
        # topn = TopN(pred, select)
        # pred.fit(ratings)
        # select.fit(ratings)
        fittable.fit(train)
        users = test.user.unique()
        recs = batch.recommend(fittable, users, n)
        recs['Algorithm'] = aname
        return recs

    all_recs = []
    test_data = []

    for i in data:                          #for train, test in xf.partition_users(ratings[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)):
        train = data[i]['train']
        test = data[i]['validation']

        # rename columns so it can be processed in the models 
        if 'user_id' in train.columns and 'item_id' in train.columns:
            train = train[['user_id', 'item_id', 'rating']]
            train = train.rename(columns={'user_id': 'user', 'item_id': 'item'} )
            test = test[['user_id', 'item_id', 'rating']]
            test = test.rename(columns={'user_id': 'user', 'item_id': 'item'} )

        test_data.append(test)
        all_recs.append(eval('ItemItem', algo_ii, train, test, n))
        all_recs.append(eval('UserUser', algo_uu, train, test, n))
        # all_recs.append(eval('ALS', algo_als, train, test, n))
        # all_recs.append(eval('tf', algo_tf, train, test, n))
        # all_recs.append(eval('MFPredictor', algo_MFPredictor, train, test, n))
        # all_recs.append(eval('bias', algo_bias, train, test, n))
        # all_recs.append(eval('implicitmf', algo_implicitmf, train, test, n))
        # all_recs.append(eval('biasedsvd', algo_biasedsvd, train, test, n))
        # all_recs.append(eval('funksvd', algo_funksvd, train, test, n))


    all_recs = pd.concat(all_recs, ignore_index=True)
    all_recs.head()
    # all_recs.to_csv('/Users/elisestijger/Desktop/all_recs.csv')

    test_data = pd.concat(test_data, ignore_index=True)
    test_data.head()

    rla = topn.RecListAnalysis()
    rla.add_metric(topn.ndcg)
    results = rla.compute(all_recs, test_data)
    results.head()
    outcome = results.groupby('Algorithm').ndcg.mean()


    return all_recs, results, outcome