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


def eval_active(predictions , eval):
    # get all algorithms:
   
    if eval == 'correct counts':
        all_outcomes =  counting(predictions)

    if eval == 'correct counts2':
        all_outcomes =  counting2(predictions)
    
    if eval == 'ndcg':
        all_outcomes =  ndcg(predictions)
        
    if eval == 'ndcg2':
        all_outcomes =  ndcg2(predictions)

    if eval == 'precision':
        all_outcomes = precision(predictions)

    if eval == 'precision2':
        all_outcomes = precision2(predictions)

    if eval == 'hit':
        all_outcomes = hit(predictions)

    if eval == 'hit2':
        all_outcomes = hit2(predictions)

    if eval == 'recip_rank':
        all_outcomes = recip_rank(predictions)

    if eval == 'recip_rank2':
        all_outcomes = recip_rank2(predictions)

    if eval == 'mapk':
        all_outcomes = map_mapk(predictions)


    if eval == 'mapk2':
        all_outcomes = map_mapk2(predictions)

    return all_outcomes

# def add_together(dict1, dict2):


# predictions = dict with folds 
def counting(predictions):

    #iterate through folds 
    # s will be 'fold0' etc?
    # dict_all_folds = {}
    dict_before = {}
    dict_after = {}

    dict_per_alg = {
        "correct counts before active learning": dict_before,
        "correct counts after active learning": dict_after
    }
    j = 0
    for s in predictions:
        # dict_fold_outcomes = {}
        test_data = predictions[s]['test']
        original = predictions[s]['recommendation original']
        after = predictions[s]['recommendation after active learning'] 
        key_fold = 'fold ' +  str(j)

        # count for original 
        merged_df = pd.merge(original, test_data, on='user', suffixes=('_df1', '_df2'))
        matching_items = merged_df[merged_df['item_df1'] == merged_df['item_df2']]
        key_original = 'correct counts before sampling active learning'
        # dict_fold_outcomes[key_original] = len(matching_items)
        dict_before[key_fold] = len(matching_items)

        # count for after 
        merged_df = pd.merge(after, test_data, on='user', suffixes=('_df1', '_df2'))
        matching_items = merged_df[merged_df['item_df1'] == merged_df['item_df2']]
        key_after = 'correct counts after active learning'
        # dict_fold_outcomes[key_after] = len(matching_items)
        # dict_all_folds[key_fold] = dict_fold_outcomes
        dict_after[key_fold] = len(matching_items)
        # print(dict_fold_outcomes)

        j = j + 1  
    
    return dict_per_alg

def counting2(predictions):
    keyvalue = 0
    dict_before = {}
    dict_after = {}
    trainsizes = []
    dict_per_alg = {
        "correct counts before active learning": dict_before,
        "correct counts after active learning": dict_after, 
        "trainsize": trainsizes

    }
    for s in predictions:
        train_size = predictions[s]['trainingsize']
        test_data = predictions[s]['test']
        original = predictions[s]['recommendation original']
        after = predictions[s]['recommendation after active learning']          # this is a list of dataframes
        keyvalue_correct = 'fold ' + str(keyvalue)
        list_all_counts = []
        merged_df = pd.merge(original, test_data, on='user', suffixes=('_df1', '_df2'))
        matching_items = merged_df[merged_df['item_df1'] == merged_df['item_df2']]
        list_all_counts.append(len(matching_items))
        dict_before[keyvalue_correct] = len(matching_items)

        for i in after:
            merged_df = pd.merge(i, test_data, on='user', suffixes=('_df1', '_df2'))
            matching_items = merged_df[merged_df['item_df1'] == merged_df['item_df2']]
            list_all_counts.append(len(matching_items))
        dict_after[keyvalue_correct] = list_all_counts
        trainsizes.append(train_size)

        keyvalue = keyvalue + 1

    return dict_per_alg

def ndcg(predictions):
    n = 0
    dict_before = {}
    dict_after = {}
    dict_per_alg = {

        "ndcg before active learning": dict_before,
        "ndcg after active learning": dict_after,
    }
    for s in predictions:
        key = 'fold' + str(n+1)
        test_data = predictions[s]['test']
        original = predictions[s]['recommendation original']
        after = predictions[s]['recommendation after active learning'] 

        user_groups = after.groupby('user')
        ndcg_total = 0
        total_users = 0
        for user, group in user_groups:
            filtered_df = test_data.loc[test_data['user'] == user]
            relevances_actual = [1 if x in list(filtered_df['item']) else 0 for x in list(group['item'])]
            
            ideal =  idcg(relevances_actual)   
            true = dcg(relevances_actual)   
            if true == 0.0 and ideal == 0.0:
                ndcg = 0.0
            else:
                ndcg = true / ideal
            ndcg_total = ndcg_total + ndcg
            total_users = total_users + 1
            dict_after[key] = ndcg_total / total_users

    # also do this for original:
        user_groups = original.groupby('user')
        ndcg_total2 = 0
        total_users2 = 0
        for user, group in user_groups:
            filtered_df = test_data.loc[test_data['user'] == user]
            relevances_actual = [1 if x in list(filtered_df['item']) else 0 for x in list(group['item'])]
            ideal =  idcg(relevances_actual)   
            true = dcg(relevances_actual)   
            if true == 0.0 and ideal == 0.0:
                ndcg = 0.0
            else:
                ndcg = true / ideal
            ndcg_total2 = ndcg_total2 + ndcg
            total_users2 = total_users2 + 1
            dict_before[key] = ndcg_total2 / total_users2
        n = n + 1 
       
    return dict_per_alg

def ndcg2(predictions):
    n = 0
    dict_before = {}
    dict_after = {}
    trainsizes = []
    dict_per_alg = {

        "ndcg before active learning": dict_before,
        "ndcg after active learning": dict_after,
        "trainsize" : trainsizes
    }
    # print(predictions)
    for s in predictions:
        key = 'fold' + str(n+1)
        # dict_original_after = {}
        test_data = predictions[s]['test']
        original = predictions[s]['recommendation original']
        after = predictions[s]['recommendation after active learning'] 
        trainsize = predictions[s]['trainingsize']
        trainsizes.append(trainsize)
    # do this for original:
        user_groups = original.groupby('user')
        ndcg_total2 = 0
        total_users2 = 0
        for user, group in user_groups:
            filtered_df = test_data.loc[test_data['user'] == user]
            relevances_actual = [1 if x in list(filtered_df['item']) else 0 for x in list(group['item'])]
            ideal =  idcg(relevances_actual)   # send all relevance scores of items this user could have (test_data)
            true = dcg(relevances_actual)   
            if true == 0.0 and ideal == 0.0:
                ndcg = 0.0
            else:
                ndcg = true / ideal
            ndcg_total2 = ndcg_total2 + ndcg
            total_users2 = total_users2 + 1
            dict_before[key] = ndcg_total2 / total_users2

    # do this for after
        list_all_ndcg = []
        for k in after:
            user_groups = k.groupby('user')
            ndcg_total = 0
            total_users = 0
            for user, group in user_groups:
                filtered_df = test_data.loc[test_data['user'] == user]
                relevances_actual = [1 if x in list(filtered_df['item']) else 0 for x in list(group['item'])]
                
                ideal =  idcg(relevances_actual)   # send all relevance scores of items this user could have (test_data)
                true = dcg(relevances_actual)   
                if true == 0.0 and ideal == 0.0:
                    ndcg = 0.0
                else:
                    ndcg = true / ideal
                ndcg_total = ndcg_total + ndcg
                total_users = total_users + 1

            list_all_ndcg.append(ndcg_total / total_users)
            dict_after[key] = list_all_ndcg

        n = n + 1 
       
    return dict_per_alg

def precision(predictions):
    dict_before = {}
    dict_after = {}
    dict_per_alg = {
        "precision before active learning": dict_before,
        "precision after active learning": dict_after
    }
    j = 0
    for s in predictions:
        key = 'fold' + str(j+1)
        test_data = predictions[s]['test']
        original = predictions[s]['recommendation original']
        rla = topn.RecListAnalysis(group_cols=['user'])
        rla.add_metric(topn.precision)
        results = rla.compute(original, test_data, include_missing=True)
        results.head()
        dict_before[key] = results.precision.mean()

    # also do this for after active learning:
        after = predictions[s]['recommendation after active learning'] 
        rla = topn.RecListAnalysis(group_cols=['user'])
        rla.add_metric(topn.precision)
        results = rla.compute(after, test_data, include_missing=True)
        results.head()
        dict_after[key] = results.precision.mean()

        j = j + 1

    return dict_per_alg

def precision2(predictions):
    dict_before = {}
    dict_after = {}
    trainsizes = []

    dict_per_alg = {
        "precision before active learning": dict_before,
        "precision after active learning": dict_after,
        "trainsize":  trainsizes
    }
    j = 0
    for s in predictions:
        trainsize = predictions[s]['trainingsize']
        trainsizes.append(trainsize)
        key = 'fold' + str(j+1)
        test_data = predictions[s]['test']
        original = predictions[s]['recommendation original']
        rla = topn.RecListAnalysis(group_cols=['user'])
        rla.add_metric(topn.precision)
        results = rla.compute(original, test_data, include_missing=True)
        results.head()
        dict_before[key] = results.precision.mean()

    # also do this for after active learning:
        after = predictions[s]['recommendation after active learning'] 
        list_precision = []
        for l in after:
            rla = topn.RecListAnalysis(group_cols=['user'])
            rla.add_metric(topn.precision)
            results = rla.compute(l, test_data, include_missing=True)
            results.head()
            list_precision.append(results.precision.mean())
        dict_after[key] = list_precision

        j = j + 1

    return dict_per_alg
  
    
def hit(predictions):
    dict_before = {}
    dict_after = {}
    dict_per_alg = {
        "hit before active learning": dict_before,
        "hit after active learning": dict_after
    }
    j = 0
    for s in predictions:
        key = 'fold' + str(j+1)
        test_data = predictions[s]['test']
        original = predictions[s]['recommendation original']
        rla = topn.RecListAnalysis(group_cols=['user'])
        rla.add_metric(topn.hit)
        results = rla.compute(original, test_data, include_missing=True)
        results.head()
        dict_before[key] = results.hit.mean()

    # also do this for after
        after = predictions[s]['recommendation after active learning'] 
        rla = topn.RecListAnalysis(group_cols=['user'])
        rla.add_metric(topn.hit)
        results = rla.compute(after, test_data, include_missing=True)
        results.head()
        dict_after[key] = results.hit.mean()

        j = j + 1 

    return dict_per_alg

def hit2(predictions):
    dict_before = {}
    dict_after = {}
    trainsizes = []
    dict_per_alg = {
        "hit before active learning": dict_before,
        "hit after active learning": dict_after,
        "trainsize": trainsizes
    }
    j = 0
    for s in predictions:
        key = 'fold' + str(j+1)
        test_data = predictions[s]['test']
        original = predictions[s]['recommendation original']
        trainsize = predictions[s]['trainingsize']
        trainsizes.append(trainsize)        
        rla = topn.RecListAnalysis(group_cols=['user'])
        rla.add_metric(topn.hit)
        results = rla.compute(original, test_data, include_missing=True)
        results.head()
        dict_before[key] = results.hit.mean()

    # also do this for after
        after = predictions[s]['recommendation after active learning'] 
        list_hit = []
        for l in after:
            rla = topn.RecListAnalysis(group_cols=['user'])
            rla.add_metric(topn.hit)
            results = rla.compute(l, test_data, include_missing=True)
            results.head()
            list_hit.append(results.hit.mean())
        dict_after[key] = list_hit

        j = j + 1 

    return dict_per_alg

    
def recip_rank(predictions):

    dict_before = {}
    dict_after = {}
    dict_per_alg = {
        "recip_rank before active learning": dict_before,
        "recip_rank after active learning": dict_after
    }
    j = 0
    for s in predictions:
        key = 'fold' + str(j+1)
        test_data = predictions[s]['test']
        original = predictions[s]['recommendation original']
        rla = topn.RecListAnalysis(group_cols=['user'])
        rla.add_metric(topn.recip_rank)
        results = rla.compute(original, test_data)
        results.head()
        dict_before[key] = results.recip_rank.mean()

    # also do this for after
        after = predictions[s]['recommendation after active learning'] 
        rla = topn.RecListAnalysis(group_cols=['user'])
        rla.add_metric(topn.recip_rank)
        results = rla.compute(after, test_data)
        results.head()
        dict_after[key] = results.recip_rank.mean()

        j = j + 1 

    return dict_per_alg

    
def recip_rank2(predictions):
    dict_before = {}
    dict_after = {}
    trainsizes = []
    dict_per_alg = {
        "recip_rank before active learning": dict_before,
        "recip_rank after active learning": dict_after, 
        "trainsize": trainsizes
    }
    j = 0
    for s in predictions:
        trainsize = predictions[s]['trainingsize']
        trainsizes.append(trainsize)        
        key = 'fold' + str(j+1)
        test_data = predictions[s]['test']
        original = predictions[s]['recommendation original']
        rla = topn.RecListAnalysis(group_cols=['user'])
        rla.add_metric(topn.recip_rank)
        results = rla.compute(original, test_data)
        results.head()
        dict_before[key] = results.recip_rank.mean()

    # also do this for after
        after = predictions[s]['recommendation after active learning'] 
        list_recip = []
        for l in after:
            rla = topn.RecListAnalysis(group_cols=['user'])
            rla.add_metric(topn.recip_rank)
            results = rla.compute(l, test_data, include_missing=True)
            results.head()
            list_recip.append(results.recip_rank.mean())
        dict_after[key] = list_recip
        j = j + 1 

    return dict_per_alg


def map_mapk( predictions):

    dict_before = {}
    dict_after = {}
    dict_per_alg = {
        "mapk before active learning": dict_before,
        "mapk after active learning": dict_after, 
    }
    j = 0
    for s in predictions:
        # trainsize = predictions[s]['trainingsize']
        # trainsizes.append(trainsize)        
        key = 'fold' + str(j+1)
        test_data = predictions[s]['test']
        original = predictions[s]['recommendation original']
        after = predictions[s]['recommendation after active learning'] 

    # for original: 
        user_groups = original.groupby('user')
        list_actual = []
        list_predicted = []
        for user, group in user_groups:
            filtered_df = test_data.loc[test_data['user'] == user]
            list_actual.append(list(filtered_df['item']))
            list_predicted.append(list(group['item']))

        
        map_score = mapk(list_actual, list_predicted, 5)
        dict_before[key] = map_score

    # for after:
        user_groups = after.groupby('user')
        list_actual_after = []
        list_predicted_after = []
        for user, group in user_groups:
            filtered_df = test_data.loc[test_data['user'] == user]
            list_actual_after.append(list(filtered_df['item']))
            list_predicted_after.append(list(group['item']))

        
        map_score_after = mapk(list_actual_after, list_predicted_after, 5)
        dict_after[key] = map_score_after
        j = j + 1 

    return dict_per_alg

def map_mapk2( predictions):

    dict_before = {}
    dict_after = {}
    trainsizes = []
    dict_per_alg = {
        "mapk before active learning": dict_before,
        "mapk after active learning": dict_after, 
        "trainsize": trainsizes 
    }
    j = 0

    for s in predictions:
        trainsize = predictions[s]['trainingsize']
        trainsizes.append(trainsize)     
        key = 'fold' + str(j+1)
        test_data = predictions[s]['test']
        original = predictions[s]['recommendation original']
        after = predictions[s]['recommendation after active learning'] 

    # for original: 
        user_groups = original.groupby('user')
        list_actual = []
        list_predicted = []
        for user, group in user_groups:
            filtered_df = test_data.loc[test_data['user'] == user]
            list_actual.append(list(filtered_df['item']))
            list_predicted.append(list(group['item']))

        
        map_score = mapk(list_actual, list_predicted, 5)
        dict_before[key] = map_score

    # for after:
        after = predictions[s]['recommendation after active learning'] 
        mapk_scores = []
        for l in after:
            user_groups = l.groupby('user')
            list_actual_after = []
            list_predicted_after = []
            for user, group in user_groups:
                filtered_df = test_data.loc[test_data['user'] == user]
                list_actual_after.append(list(filtered_df['item']))
                list_predicted_after.append(list(group['item']))

            map_score_after = mapk(list_actual_after, list_predicted_after, 5)
            mapk_scores.append(map_score_after)
        dict_after[key] = mapk_scores

        j = j + 1 

    return dict_per_alg