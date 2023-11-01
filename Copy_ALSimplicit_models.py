import numpy as np
from lenskit import crossfold as xf
from collections import defaultdict
from lenskit_modules import do_recommendations
from lenskit import batch, topn, util, topn
from lenskit.algorithms import Recommender, als, item_knn as knn
from lenskit.algorithms import Recommender, als, user_knn as knn_user
import pandas as pd
import numpy as np
import implicit
from scipy.sparse import csr_matrix
import scipy.sparse as sparse
from matrix import make_matix2
from implicit.nearest_neighbours import bm25_weight



def start_test(data, n):
    algo_als = implicit.als.AlternatingLeastSquares() #factors=20, regularization=0.1, iterations=20)
    algo_nn =  implicit.nearest_neighbours.CosineRecommender()
    # algo_base = implicit.recommender_base.RecommenderBase()
    algo_baseysian = implicit.bpr.BayesianPersonalizedRanking(factors=20, regularization=0.1, iterations=20)
    algo_logistic = implicit.lmf.LogisticMatrixFactorization()

    def eval(aname, algo, train, test, n):
        model = algo
        sparse_matrix = sparse.csr_matrix((train['rating'].astype(float), (train['item'].astype(int), train['user'].astype(int))))
        sparse_matrix = sparse_matrix.T.tocsr()

        alpha_val = 40
        data_conf = (sparse_matrix * alpha_val).astype('double')
        # fittable.fit(data_conf)
        model.fit(data_conf)

        user_ids = sparse_matrix.nonzero()[0]

         # Get unique user IDs
        unique_user_ids = np.unique(user_ids) 
        ids, scores = model.recommend(unique_user_ids, sparse_matrix[unique_user_ids], N=n)
        data = {'item': np.concatenate(ids), 'user': np.repeat(unique_user_ids, [len(items) for items in ids]), 'rating': 1.0}
        df = pd.DataFrame(data)
        df['user'] = df['user'].astype(str)
        df['item'] = df['item'].astype(str)
        df['Algorithm'] = aname
        return df #recs
    
    all_recs = []
    test_data = []
    fold_data = {}

    for i in data:                          #for train, test in xf.partition_users(ratings[['user', 'item', 'rating']], 5, xf.SampleFrac(0.2)):
        # print(data)
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
        recommendations_nn = eval('nn', algo_nn, train, test, n)
        recommendations_als = eval('als', algo_als, train, test, n)
        # recommendations_base = eval('base', algo_base, train, test, n)
        recommendations_baseysian = eval('baseyesian', algo_baseysian, train, test, n)
        recommendations_logistic = eval('logistic', algo_logistic, train, test, n)


        all_recs.append(recommendations_nn)
        all_recs2.append(recommendations_nn)
        all_recs.append(recommendations_als)
        all_recs2.append(recommendations_als)
        # all_recs.append(recommendations_base)
        # all_recs2.append(recommendations_base)
        all_recs.append(recommendations_baseysian)
        all_recs2.append(recommendations_baseysian)
        all_recs.append(recommendations_logistic)
        all_recs2.append(recommendations_logistic)

        fold_data[i] = {
        'Recommendations': all_recs2,
        'Test_Data': test
    }

    # get recs per fold + validation data 
    all_recs = pd.concat(all_recs, ignore_index=True)
    all_recs.head()
    # all_recs.to_csv('/Users/elisestijger/Desktop/all_recs.csv')

    test_data = pd.concat(test_data, ignore_index=True)
    test_data.head()

    return all_recs, test_data, fold_data 


