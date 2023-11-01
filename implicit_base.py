import implicit
import numpy as np
import scipy.sparse as sparse
# from implicit.datasets.lastfm import get_lastfm
from implicit.nearest_neighbours import bm25_weight
from scipy.sparse import csr_matrix
import pandas as pd


# Is working but really bad predictions....
def implicitbase(matrix):
    sparse_matrix1 = bm25_weight(matrix, K1=100, B=0.8)
    sparse_matrix = sparse_matrix1.T.tocsr()

    # sparse_matrix = csr_matrix(sparse_matrix2)
    model = implicit.recommender_base.RecommenderBase()
    model.fit(sparse_matrix)
    # recommendations = model.recommend(userid, sparse_matrix[userid])

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
    for user_id, recommended_items in enumerate(ids):
        # print("user_id", user_id)
        # print(recommended_items, "recom items")
        for item_id in recommended_items:
            # print('itemid', item_id)
            # Check if the user-item combination appears in the csr_matrix
            appears_in_matrix = sparse_matrix[user_id, item_id] > 0

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

    print(df)
    print("COUNT",len(df[df['appears_in_matrix'] == 'TRUE']))


    # df.to_csv('/Users/elisestijger/Desktop/implicitrecommenDF.csv')


# END 
    # print(sparse_matrix[user_ids])
    #class 'numpy.ndarray = ids // OUTCOMES DIFFER 

    # df_outcomes = pd.DataFrame({"artist": ids, "score": scores, "already_liked": np.in1d(ids, sparse_matrix[user_ids])})

    # related = model.similar_items(item_id)


    # # Recommend items for a user
    # user_id = 3  # Replace with the user you want to recommend to
    # recommendations = model.recommend(user_id, sparse_matrix)

    # # Get item IDs from recommendations
    # recommended_item_ids = [item_id for item_id, _ in recommendations]

    return sparse_matrix
    # return ids, scores