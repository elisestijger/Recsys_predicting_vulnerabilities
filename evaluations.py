from lenskit.metrics import predict
import pandas as pd 
import numpy as np
from lenskit import batch, topn, util
# from topn_reclist import RecListAnalysis
# from topn import ndcg
from average_precision import apk, mapk
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import ndcg_score, dcg_score
import recmetrics 
from ndcg import ndcg , idcg , dcg
from metrics import prediction_coverage, novelty, personalization, novelty


def eval(predictions, train_data, eval):
    # get all algorithms:
    lists_algorithm = []
    for i in  range(0, len(predictions)):
        # print(predictions[i]['Recommendations'])
        for j in predictions[i]['Recommendations']:
            for m in j['Algorithm']:
                if  m not in lists_algorithm:
                    lists_algorithm.append(m)    

    # # this list contains the lists filtered on the type of algorithm
    filtered_lists = []
    for k in lists_algorithm:
        filtered_list = []
        for i in range(0, len(predictions)):
            filtered = [df for df in predictions[i]['Recommendations'] if k in df['Algorithm'].values]
            filtered_list.append(filtered)
        filtered_lists.append(filtered_list)

    if eval == 'correct counts':
        all_outcomes =  counting(filtered_lists, predictions)
    
    if eval == 'ndcg':
        all_outcomes =  ndcg(filtered_lists, predictions)
        
    if eval == 'precision':
        all_outcomes = precision(filtered_lists, predictions)

    # if eval == 'recall':
    #     all_outcomes = recall(filtered_lists, predictions)

    if eval == 'hit':
        all_outcomes = hit(filtered_lists, predictions)

    if eval == 'recip_rank':
        all_outcomes = recip_rank(filtered_lists, predictions)
    
    # if eval == 'f1measure':
    #     all_outcomes = f1measure(filtered_lists, predictions)

    if eval == 'apk':
        all_outcomes = map_apk(filtered_lists, predictions)

    if eval == 'mapk':
        all_outcomes = map_mapk(filtered_lists, predictions)

    if eval == 'coverage':
            all_outcomes = coverage(filtered_lists, predictions)

    if eval == 'personalization':
            all_outcomes = do_personalization(filtered_lists)

    if eval == 'novelty':
            all_outcomes = do_novelty(filtered_lists, train_data)

    return all_outcomes

# counts amount of 
def counting(filtered_lists, predictions):
        all_outcomes = []
        dict_per_alg = {}
        k = 0
        for p in filtered_lists:
            outcomes_per_fold = {}
            outcomes = []
            name_alg = "count for algorithm " + p[0][0]['Algorithm'][k]
            total_count = 0
            for s in range(0,len(p)):
                count = 0
                test_data = predictions[s]['Test_Data']
                predicted_recoms = p[s][0] 
                merged_df = pd.merge(predicted_recoms, test_data, on='user', suffixes=('_df1', '_df2'))
                matching_items = merged_df[merged_df['item_df1'] == merged_df['item_df2']]
                outcomes.append(matching_items)
                count = count + len(matching_items)
                key = "fold " + str(s+1)
                outcomes_per_fold[key] = count
                total_count = total_count + count
            all_outcomes.append(outcomes)
            dict_per_alg[name_alg] = outcomes_per_fold
            k = k + 1
        return dict_per_alg

def ndcg(filtered_lists, predictions):
    dict_all_outcomes = {}
    dict_all_outcomes_all = {}
    for p in filtered_lists:
        name_alg =  "ndcg for "+ p[0][0]['Algorithm'][0]
        dict_folds_outcomes = {}
        dict_folds_outcomes_all = {}
        for n in range(0, len(p)):
                user_groups = p[n][0].groupby('user')
                dict_all_users = {}
                ndcg_total = 0
                total_users = 0
                for user, group in user_groups:
                    filtered_df = predictions[n]['Test_Data'].loc[predictions[n]['Test_Data']['user'] == user]
                    relevances_actual = [1 if x in list(filtered_df['item']) else 0 for x in list(group['item'])]
                    ideal =  idcg(relevances_actual)   # send all relevance scores of items this user could have (test_data)
                    true = dcg(relevances_actual)   
                    if true == 0.0 and ideal == 0.0:
                        ndcg = 0.0
                    else:
                        ndcg = true / ideal
                    dict_all_users[user]= ndcg
                    ndcg_total = ndcg_total + ndcg
                    total_users = total_users + 1

                key = 'fold' + str(n+1)
                dict_folds_outcomes_all[key] = dict_all_users
                dict_folds_outcomes[key] = ndcg_total / total_users

        dict_all_outcomes[name_alg] = dict_folds_outcomes
        dict_all_outcomes_all[name_alg] = dict_folds_outcomes

    return dict_all_outcomes

def precision(filtered_lists, predictions):
    dict_all_outcomes = {}
    all_results = []
    for p in filtered_lists:
        name_alg =  "precision for "+ p[0][0]['Algorithm'][0]
        dict_folds_outcomes = {}
        for s in range(0,len(p)):
            test_data = predictions[s]['Test_Data']
            predicted_recoms = p[s][0]    
            rla = topn.RecListAnalysis(group_cols=['user'])
            rla.add_metric(topn.precision)
            results = rla.compute(predicted_recoms, test_data, include_missing=True)
            results.head()
            all_results.append(results)
            key = 'fold' + str(s+1)
            dict_folds_outcomes[key] = results.precision.mean()

        dict_all_outcomes[name_alg] = dict_folds_outcomes

    return dict_all_outcomes
    
# def recall(filtered_lists, predictions):
#     dict_all_outcomes = {}
#     # all_outcomes = []
#     all_results = []
#     for p in filtered_lists:
#         name_alg =  "recall for "+ p[0][0]['Algorithm'][0]
#         dict_folds_outcomes = {}
#         for s in range(0,len(p)):
#             test_data = predictions[s]['Test_Data']
#             predicted_recoms = p[s][0]    
#             rla = topn.RecListAnalysis(group_cols=['user'])
#             rla.add_metric(topn.recall)
#             results = rla.compute(predicted_recoms, test_data, include_missing=True)
#             results.head()
#             all_results.append(results)
#             # outcome = results.groupby('Algorithm').recall.mean()
#             # all_outcomes.append(outcome)
#             key = 'fold' + str(s+1)
#             dict_folds_outcomes[key] = results.recall.mean()

#         dict_all_outcomes[name_alg] = dict_folds_outcomes

    
#     return dict_all_outcomes
    
def hit(filtered_lists, predictions):
    dict_all_outcomes = {}
    # all_outcomes = []
    all_results = []
    for p in filtered_lists:
        name_alg =  "hit for "+ p[0][0]['Algorithm'][0]
        dict_folds_outcomes = {}
        for s in range(0,len(p)):
            test_data = predictions[s]['Test_Data']
            predicted_recoms = p[s][0]    
            rla = topn.RecListAnalysis(group_cols=['user'])
            rla.add_metric(topn.hit)
            results = rla.compute(predicted_recoms, test_data, include_missing=True)
            results.head()
            all_results.append(results)
            # outcome = results.groupby('Algorithm').hit.mean()
            # all_outcomes.append(outcome)
            key = 'fold' + str(s+1)
            dict_folds_outcomes[key] = results.hit.mean()

        dict_all_outcomes[name_alg] = dict_folds_outcomes

    return dict_all_outcomes
    
def recip_rank(filtered_lists, predictions):
    dict_all_outcomes = {}
    all_results = []
    for p in filtered_lists:
        name_alg =  "recip_rank for"+ p[0][0]['Algorithm'][0]
        dict_folds_outcomes = {}
        for s in range(0,len(p)):
            test_data = predictions[s]['Test_Data']
            predicted_recoms = p[s][0]    
            rla = topn.RecListAnalysis()
            rla.add_metric(topn.recip_rank)
            results = rla.compute(predicted_recoms, test_data)
            results.head()
            all_results.append(results)
            key = 'fold' + str(s+1)
            dict_folds_outcomes[key] = results.recip_rank.mean()

        dict_all_outcomes[name_alg] = dict_folds_outcomes

    return dict_all_outcomes

# def f1measure(filtered_lists, predictions):
#     precision_data_df = precision(filtered_lists, predictions)[0]
#     recall_data_df = recall(filtered_lists, predictions)[0]
#     # this is for the dataframe object
#     for j in range(0,len(precision_data_df)):
#         copydf = precision_data_df.copy()
#         copydf[j]['recall'] = recall_data_df[j]['recall']
#         copydf[j]['f1measure'] =  float('nan')
#         for n in range(0, len(precision_data_df[j]['precision'])):
#             precision_value_df = precision_data_df[j]['precision'][n]
#             recall_value_df = recall_data_df[j]['recall'][n]
#             # print(copydf)
#             if precision_value_df == 0 or recall_value_df == 0:
#                 copydf[j]['f1measure'][n] = 0
#             else:
#                 f1_score_df = (2 * (precision_value_df * recall_value_df) / (precision_value_df + recall_value_df))
#                 copydf[j]['f1measure'][n] = f1_score_df


#     precision_data = precision(filtered_lists, predictions)[1]
#     recall_data = recall(filtered_lists, predictions)[1]
#     copy_df = precision_data.copy()
#     f1_scores = []
#     # evaluation[1][0] = pd.Series(evaluation[1][0], name='f1measure')

#     # This is for the series object
#     for i in range(0,len(precision_data)):
#         precision_value =  precision_data[i].iloc[0]
#         recall_value = recall_data[i].iloc[0]
#         if precision_value == 0 or recall_value == 0:
#                 copy_df[i].name = 'f1measure'
#                 copy_df[i].iloc[0] = 0
#                 f1_scores.append(0)  # Handle the case where precision or recall is zero to avoid division by zero
#         else:
#             f1_score = (2 * (precision_value * recall_value) / (precision_value + recall_value))
#             copy_df[i].name = 'f1measure'
#             copy_df[i].iloc[0] = f1_score
#             f1_scores.append(f1_score)
  
#     return copydf ,copy_df

# I do have this but it is not really usefull/ meaningful in my case. 
def map_apk(filtered_lists, predictions):
    dict_all_outcomes = {}
    for j in filtered_lists:                    
        name_alg =  "apk for "+ j[0][0]['Algorithm'][0]
        dict_folds_outcomes = {}
        for n in range(0, len(j)):
            user_groups = j[n][0].groupby('user')
            dict_all_users = {}
            total_apk_all_users = 0 
            for user, group in user_groups:
                filtered_df = predictions[n]['Test_Data'].loc[predictions[n]['Test_Data']['user'] == user]
                map_score = apk(list(filtered_df['item']), list(group['item']), 5)
                dict_all_users[user]= map_score
                total_apk_all_users = total_apk_all_users + map_score

            key = 'fold' + str(n+1)
            dict_folds_outcomes[key] = dict_all_users

        dict_all_outcomes[name_alg] = dict_folds_outcomes

    return dict_all_outcomes #map_scores , fill_in_data_df
    
def map_mapk(filtered_lists, predictions):
    dict_all_outcomes = {}
    for j in filtered_lists:
        name_alg =  "mapk for "+ j[0][0]['Algorithm'][0]
        dict_folds_outcomes = {}
        for n in range(0, len(j)):
            user_groups = j[n][0].groupby('user')
            # dict_all_users = {}
            # total_apk_all_users = 0 
            list_actual = []
            list_predicted = []
            for user, group in user_groups:
                # make list of lists 

                filtered_df = predictions[n]['Test_Data'].loc[predictions[n]['Test_Data']['user'] == user]
                list_actual.append(list(filtered_df['item']))
                list_predicted.append(list(group['item']))
                
                # map_score = apk(list(filtered_df['item']), list(group['item']), 5)
                # dict_all_users[user]= map_score
                # total_apk_all_users = total_apk_all_users + map_score

            map_score = mapk(list_actual, list_predicted, 5)

            # list_predicted = list(j[n][0]['item'])
            # list_actual = list(predictions[n]['Test_Data']['item'])
            # map_score = mapk(list_actual, list_predicted, 5)
            key = 'fold' + str(n+1)
            dict_folds_outcomes[key] = map_score

        dict_all_outcomes[name_alg] = dict_folds_outcomes

    return dict_all_outcomes

def coverage(filtered_lists, predictions):
    dict_all_outcomes = {}

    # make all test data:
    all_test_data = []
    for l in range(0, len(predictions)):        # print for every fold the test_data in a list 
        test_data = list(predictions[l]['Test_Data']['item'])
        all_test_data.append(test_data)

    for j in range(0, len(filtered_lists)):
        name_alg =  "coverage for "+ filtered_lists[j][0][0]['Algorithm'][0]
        dict_folds_outcomes = {}
        for n in range(0, len(filtered_lists[j])):
            # list_predicted = list(filtered_lists[j][n][0]['item'])
            user_item_lists = filtered_lists[j][n][0].groupby('user')['item'].apply(list).reset_index()
            list_items_p_user = user_item_lists['item'].tolist()                                    
            unique_values = list(set(value for sublist in all_test_data for value in sublist))      
            outcome_coverage = prediction_coverage(list_items_p_user,unique_values)
            key = 'fold' + str(n+1)
            dict_folds_outcomes[key] = outcome_coverage
 
        dict_all_outcomes[name_alg] = dict_folds_outcomes

    return  dict_all_outcomes

# Calculates (cosine) similarity across all users predictions
def do_personalization(filtered_lists):
    dict_all_outcomes = {}
    for j in range(0, len(filtered_lists)):
        name_alg =  "personalization for "+ filtered_lists[j][0][0]['Algorithm'][0]
        dict_folds_outcomes = {}
        for n in range(0, len(filtered_lists[j])):
            user_item_lists = filtered_lists[j][n][0].groupby('user')['item'].apply(list).reset_index()
            list_items_p_user = user_item_lists['item'].tolist()
            outcome_personalization = personalization(list_items_p_user)
            key = 'fold' + str(n+1)
            dict_folds_outcomes[key] = outcome_personalization

        dict_all_outcomes[name_alg] = dict_folds_outcomes

    return  dict_all_outcomes

# how "new" are the recommendations to the user 
def do_novelty(filtered_lists, train_data):
    dict_all_outcomes = {}
    item_counts = train_data['item_id'].value_counts().to_dict()      
    num_users = len(train_data['user_id'].unique())           

    for j in range(0, len(filtered_lists)):
        name_alg =  "novelty for "+ filtered_lists[j][0][0]['Algorithm'][0]
        dict_folds_outcomes = {}
        for n in range(0, len(filtered_lists[j])):
            user_item_lists = filtered_lists[j][n][0].groupby('user')['item'].apply(list).reset_index()
            list_items_p_user = user_item_lists['item'].tolist()
            outcome_personalization = novelty(list_items_p_user, item_counts, num_users, len(list_items_p_user[0]))

            key = 'fold' + str(n+1)
            dict_folds_outcomes[key] = outcome_personalization

        dict_all_outcomes[name_alg] = dict_folds_outcomes
      
    return dict_all_outcomes # dict_outcomes , mean_dict_outcomes 

