import implicit
import numpy as np
import scipy.sparse as sparse
# from implicit.datasets.lastfm import get_lastfm
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix
import pandas as pd


def implicitBayesian(matrix, realmatrix):
    sparse_matrix = sparse.csr_matrix((matrix['rating'].astype(float), (matrix['item_id'].astype(int), matrix['user_id'].astype(int))))

    model = implicit.bpr.BayesianPersonalizedRanking(factors=20, regularization=0.1, iterations=20)
    alpha_val = 40
    data_conf = (sparse_matrix * alpha_val).astype('double')
    model.fit(data_conf)

    # Get the row indices (user IDs)
    user_ids = sparse_matrix.nonzero()[0]

    # Get unique user IDs
    unique_user_ids = np.unique(user_ids)    # print(sparse_matrix.indices)

    ids, scores = model.recommend(unique_user_ids, sparse_matrix[unique_user_ids], N=5)

    # Initialize lists to store DataFrame columns
    user_id_column = []
    recommended_items_column = []
    appears_in_matrix_column = []

    # Iterate through user IDs and recommended item IDs
    for item_id, recommended_items in enumerate(ids):
        # print("user_id", user_id)
        # print(recommended_items, "recom items")
        for user_id in recommended_items:
            # print('itemid', item_id)
            # Check if the user-item combination appears in the csr_matrix
            # appears_in_matrix = realmatrix.iloc[user_id, item_id] > 0
            appears_in_matrix = realmatrix.iloc[user_id, item_id] > 0

            # Append data to columns
            user_id_column.append(user_id)
            recommended_items_column.append(item_id)
            appears_in_matrix_column.append(appears_in_matrix)

    # Create a DataFrame
    df = pd.DataFrame({
        'user_id': user_id_column,
        'recommended_item_id': recommended_items_column,
        'appears_in_matrix': appears_in_matrix_column
    })

    return df
