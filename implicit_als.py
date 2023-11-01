import implicit
import numpy as np
import scipy.sparse as sparse
# from implicit.datasets.lastfm import get_lastfm
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix
import pandas as pd


# Is working but really bad predictions....
def ALS(matrix, realmatrix):
    # sparse_matrix1 = bm25_weight(matrix, K1=100, B=0.8)
    # sparse_matrix = sparse_matrix1.T.tocsr()
    # sparse_matrix = matrix

    # sparse_matrix = csr_matrix(matrix)
# GOOD? :
    # sparse_matrix = sparse.csr_matrix(matrix)
    # model = implicit.als.AlternatingLeastSquares(factors=64)
    # model.fit(sparse_matrix)

    # recommendations = model.recommend(userid, sparse_matrix[userid])

    # print(matrix)
    sparse_matrix = sparse.csr_matrix((matrix['rating'].astype(float), (matrix['item_id'].astype(int), matrix['user_id'].astype(int))))

     # Get the rows and columns for our new matrix
    # users = list(np.sort(matrix.user_id.unique()))
    # artists = list(np.sort(matrix.item_id.unique()))
    # rows = matrix.user_id.astype(int)
    # cols = matrix.item_id.astype(int)
    # sparse_matrix = sparse.csr_matrix((matrix['rating'].astype(float), (rows, cols)), shape=(len(users), len(artists)))

    # traindata = sparse_matrix.iloc[3:] 
    # testdata = sparse_matrix.iloc[0:3] 

    # rating , itemid, userid
    #Building the model
    model = implicit.als.AlternatingLeastSquares()
    alpha_val = 40
    data_conf = (sparse_matrix * alpha_val).astype('double')
    model.fit(data_conf)


    # Get the row indices (user IDs)
    user_ids = sparse_matrix.nonzero()[0]

    # print("userids", user_ids)
    # Get unique user IDs

    unique_user_ids = np.unique(user_ids)    # print(sparse_matrix.indices)

    ids, scores = model.recommend(unique_user_ids, sparse_matrix[unique_user_ids], N=5)

    # print(ids)
#BEGIN
    # Initialize lists to store DataFrame columns
    user_id_column = []
    recommended_items_column = []
    appears_in_matrix_column = []

    # Iterate through user IDs and recommended item IDs
    for item_id, recommended_items in enumerate(ids):
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
    # return ids, scores