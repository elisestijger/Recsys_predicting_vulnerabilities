import implicit
import numpy as np
import scipy.sparse as sparse
# from implicit.datasets.lastfm import get_lastfm
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix
import pandas as pd


# Is working but really bad predictions....
def implicitlogistic(matrix, realmatrix):
    # sparsematrix 
    sparse_matrix = sparse.csr_matrix((matrix['rating'].astype(float), (matrix['item_id'].astype(int), matrix['user_id'].astype(int))))

    # sparse_matrix = csr_matrix(sparse_matrix2)
    model = implicit.lmf.LogisticMatrixFactorization()
    model.fit(sparse_matrix)

    # Get the row indices (user IDs)
    user_ids = sparse_matrix.nonzero()[0]

    # Get unique user IDs
    unique_user_ids = np.unique(user_ids)    # print(sparse_matrix.indices)

    ids, scores = model.recommend(unique_user_ids, sparse_matrix[unique_user_ids], N=20)

#BEGIN
    # Initialize lists to store DataFrame columns
    user_id_column = []
    recommended_items_column = []
    appears_in_matrix_column = []

    # Iterate through user IDs and recommended item IDs
    for item_id, recommended_items in enumerate(ids):
        for user_id in recommended_items:
            appears_in_matrix = realmatrix.iloc[user_id, item_id] > 0
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
