from lenskit import batch, topn, util, topn
from lenskit import crossfold as xf
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit.algorithms import Recommender, als, user_knn as knn_user
from lenskit.algorithms import svd
from lenskit.algorithms import funksvd
from lenskit.algorithms.basic import Random
from basic import Popular
from lenskit.algorithms.basic import PopScore 
from lenskit.algorithms import bias
from lenskit.algorithms.als import BiasedMF, ImplicitMF
from lenskit.algorithms.svd import BiasedSVD
from lenskit.algorithms.basic import Bias, Popular, TopN, AllItemsCandidateSelector
import pandas as pd
import numpy as np


def make_recommendations(data, n):
    algo_ii = knn.ItemItem(10, feedback='implicit')  
    algo_uu = knn_user.UserUser(10, feedback='implicit')
    algo_random = Random()
    algo_pop = Popular()
    algo_pop2 = Popular(selector=AllItemsCandidateSelector())

    # NOT REALLY GOOD OUTCOMES:
# UNCOMMENT
    # algo_als = als.BiasedMF(50)
    # algo_implicitmf = als.ImplicitMF(50)
    # algo_biasedsvd = svd.BiasedSVD(50)
    # algo_funksvd = funksvd.FunkSVD(50)


    algo_tf = PopScore()   #gives weird outcomes
    algo_bias = Bias()    #gives weird outcomes
    algo_MFPredictor = MFPredictor()

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
        # print(recs)
        return recs, fittable

    all_recs = []
    # all_recs_per_fold = []
    # all_test_per_fold = []
    fold_data = {}
    # fold_data["Recommendations"] = []

    test_data = []

    for i in data:                          #for train, test in xf.partition_users(ratings[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)):
        train = data[i]['train']
        test = data[i]['validation']

        all_recs2 = []

        # rename columns so it can be processed in the models 
        if 'user_id' in train.columns and 'item_id' in train.columns:
            train = train[['user_id', 'item_id', 'rating']]
            train = train.rename(columns={'user_id': 'user', 'item_id': 'item'} )
            test = test[['user_id', 'item_id', 'rating']]
            test = test.rename(columns={'user_id': 'user', 'item_id': 'item'} )

        test_data.append(test)
        recommendations_itemitem0 = eval('ItemItem', algo_ii, train, test, n)
        recommendations_itemitem = recommendations_itemitem0[0]
        recommendations_itemitem_model = recommendations_itemitem0[1]
        recommendations_useruser = eval('UserUser', algo_uu, train, test, n)
        recommendations_random = eval('algo_random', algo_random, train, test, n)
        recommendations_pop = eval('algo_pop_unseen', algo_pop, train, test, n)
        recommendations_als = eval('algo_als', algo_als, train, test, n)
        recommendations_implicitmf = eval('algo_implicitmf', algo_implicitmf, train, test, n)
        recommendations_biasedsvd = eval('algo_biasedsvd', algo_biasedsvd, train, test, n)
        recommendations_funksvd = eval('algo_funksvd', algo_funksvd, train, test, n)
        recommendations_pop2 = eval('algo_pop_seen', algo_pop2, train, test, n)



        # recommendations_both = 
        all_recs.append(recommendations_itemitem)
        all_recs.append(recommendations_useruser)
        all_recs.append(recommendations_random)
        all_recs.append(recommendations_pop)
        all_recs.append(recommendations_als)
        all_recs.append(recommendations_implicitmf)
        all_recs.append(recommendations_biasedsvd)
        all_recs.append(recommendations_funksvd)
        all_recs.append(recommendations_pop2)



        all_recs2.append(recommendations_itemitem)
        all_recs2.append(recommendations_useruser)
        all_recs2.append(recommendations_random)
        all_recs2.append(recommendations_pop)
        all_recs2.append(recommendations_als)
        all_recs2.append(recommendations_implicitmf)
        all_recs2.append(recommendations_biasedsvd)
        all_recs2.append(recommendations_funksvd)
        all_recs2.append(recommendations_pop2)


        all_recs_per_fold.append(all_recs)
        all_test_per_fold.append(test)
        all_recs.append(eval('ALS', algo_als, train, test, n))
        # all_recs.append(eval('tf', algo_tf, train, test, n))
        # all_recs.append(eval('MFPredictor', algo_MFPredictor, train, test, n))
        # all_recs.append(eval('bias', algo_bias, train, test, n))
        all_recs.append(eval('implicitmf', algo_implicitmf, train, test, n))
        all_recs.append(eval('biasedsvd', algo_biasedsvd, train, test, n))
        all_recs.append(eval('funksvd', algo_funksvd, train, test, n))
        all_recs.append(eval('POP', algo_pop, train, test, n))
        all_recs.append(eval('POP_seen', algo_pop2, train, test, n))


        # all_recs.append(eval('POP', algo_pop, train, test, n))
        # all_recs.append(eval('POP_seen', algo_pop2, train, test, n))

        # all_recs2.append(eval('tf', algo_tf, train, test, n))
        # all_recs2.append(eval('MFPredictor', algo_MFPredictor, train, test, n))
        # all_recs2.append(eval('bias', algo_bias, train, test, n))
        all_recs2.append(eval('implicitmf', algo_implicitmf, train, test, n))
        all_recs2.append(eval('biasedsvd', algo_biasedsvd, train, test, n))
        all_recs2.append(eval('funksvd', algo_funksvd, train, test, n))
        all_recs2.append(eval('POP', algo_pop, train, test, n))
        all_recs2.append(eval('POP_seen', algo_pop2, train, test, n))

        fold_data[i] = {
        'Recommendations': all_recs2,
        'Test_Data': test       # test data per fold ! 
    }

    # get recs per fold + validation data 
    all_recs = pd.concat(all_recs, ignore_index=True)
    all_recs.head()

    test_data = pd.concat(test_data, ignore_index=True)
    test_data.head()

    # rla = topn.RecListAnalysis()
    # rla.add_metric(topn.ndcg)
    # results = rla.compute(all_recs, test_data)
    # results.head()
    # outcome = results.groupby('Algorithm').ndcg.mean()

    # print(fold_data)
    return all_recs, test_data, fold_data, recommendations_itemitem_model          #, results, outcome
