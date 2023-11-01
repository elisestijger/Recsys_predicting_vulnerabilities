from cross_validation_userbased import user_based_validation_split
from lenskit_modules import do_recommendations

def start_user_based(interaction_sequence, n, m):       # m = recom

    # Make the folds with user based cross validation with n folds
    test_folds = user_based_validation_split(interaction_sequence, n)

    # save these folds in a dict 
    filled_dict = {}

    # # Iterate through each fold in the existing dictionary // Getting the real data with the indices 
    for fold, data in test_folds.items():
        filled_fold = {}
        for key, indices in data.items():
            # Use the indices to retrieve values from the DataFrame
            values = interaction_sequence.iloc[indices]
            filled_fold[key] = values
        filled_dict[fold] = filled_fold

    # Do item item and user user knn and return the results 
    results = do_recommendations(filled_dict, n=m)
    itemitem_outcome = results[0]
    results_ngcd = results[1]
    outcome_ngcd =  results[2]
    
    return results, itemitem_outcome, results_ngcd, outcome_ngcd, filled_dict
