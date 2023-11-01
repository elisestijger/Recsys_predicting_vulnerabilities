#!/usr/bin/env python3
import sys
import argparse
import joblib
import numpy as np
import pandas as pd
from load_data import grab_data, get_data
from substract_cwe import extract_cwe
from matrix import make_matrix, save_matrix, load_matrix
from pruning import prune
from interaction_sequence import make_interaction_sequence 
from metadata_features import calculate_meta_features
import csv
# from implicit_als import ALS
# from implicit_Bayesian import implicitBayesian
# from implicit_nn import implicitNN_cosine, implicitNN_tfidf, implicitNN_bm25
# from random_algorithm import do_random, test_random
# from popularity_algorithm import do_pop, test_pop
from Copy_ALSimplicit_models import start_test
from evaluations import eval
from REALrandom_cross_val import random_cross_fold
from REALlenskit_modules import make_recommendations
from graphs import make_graphs, make_graphs_together, learningcurve,learningcurve_together, make_boxplots, boxplots2, make_bars
import matplotlib.pyplot as plt
from active_learning import activelearning
from lenskit.algorithms import Recommender, als, user_knn as knn_user
from lenskit.algorithms import Recommender, als, user_knn as knn_user
import random 
from test_activel import * 
from ndcg import ndcg , idcg , dcg
from evaluations_active_learning import eval_active
from Statistical_hypothesis import pairedt

def main():
    # default
    year1 = 2003
    year2 = 2023
    # parge arguments
    parser = argparse.ArgumentParser(
                    prog='SoftwareVulnerbilityPredicter',
                    description='This program will get/modify cyber security data and will predict vulnerabilities in software products',
                    epilog='paser will parse -year1 and -year2 to retrieve data from these years')
    parser.add_argument('-year1')     
    parser.add_argument('-year2')   
    args = parser.parse_args()
    
    if args.year1 != None:
        year1 = int(args.year1)
    if args.year2 != None:
        year2 = int(args.year2)

    # -------------------------------------
    # load data 

    # cve_all = get_data(year1,year2)
    # # print(cve_all)

    # cve_all = grab_data()

    # # # change data 
    # cwe_all = extract_cwe(cve_all)
    
    # # # pruning
    # cwe_all_pruned = prune(cwe_all)

    # cwe_all_pruned = joblib.load('cwe_all_pruned.pkl')

    # # # matrix 
    # matrix = make_matrix(cwe_all_pruned)

    # matrix = load_matrix()
    # matrix = joblib.load('matrix.pkl')
    # # # interaction sequence

    # interaction_sequence = make_interaction_sequence(matrix)[0]
    # user_mapping = make_interaction_sequence(matrix)[1]
    # item_mapping = make_interaction_sequence(matrix)[2]

    # joblib.dump(interaction_sequence, 'interaction_sequence_correct.pkl')
    interaction_sequence = joblib.load('interaction_sequence_correct.pkl')
    # pd.DataFrame(interaction_sequence)
    # cwe_filtered = joblib.load('cwe_all_pruned.pkl')

   # -------------------------------------


# random cross validation (only making the folds: df with n folds each fold has a train and a validation part ):
    
    # folds = random_cross_fold(interaction_sequence, 5)

    # joblib.dump(folds, 'next_folds.pkl')

    folds = joblib.load('next_folds.pkl')

    # subset_folds = joblib.load('subset_folds.pkl')

# lenskit 

    # predicted_recommendations_lenskit = make_recommendations(folds, 5)           

    # fold_data = predicted_recommendations_lenskit[2]

    # fold_data_lenskit = joblib.load('fold_data_all_algorithms_newPOP_both_pops.pkl')


# implicit

    # predicted_recommendations_implicit = start_test(folds, 5)       
    
    # fold_data2 = predicted_recommendations_implicit[2]

    # fold_data_implicit = joblib.load('folddata_1implicit_all.pkl')


# evaluations   // CHANGE PARAMETER FOR EVALUATION METRIC! 

    # evaluation_lenskit = eval(fold_data_lenskit, interaction_sequence, 'count')
    # evaluation_implicit = eval(fold_data_implicit, interaction_sequence, 'count')

# graphs

    # graph_lenskit = make_graphs(evaluation_lenskit)
    # graph_implicit = make_graphs(evaluation_implicit)

    # graph_lenskit.show()
    # graph_implicit.show()


#  bar plots for multiple recommender systems
    # fold_data_lenskit = joblib.load('fold_data_all_algorithms_newPOP_both_pops.pkl')

    # variable_names = ['correct counts', 'precision', 'ndcg', 'hit', 'recip_rank', 'mapk']
    # for var_name in variable_names:
    #     testingnewgraph = eval(fold_data_lenskit, interaction_sequence, var_name)
    #     graph_lenskit = make_graphs(testingnewgraph, var_name)
    #     graph_lenskit.show()


# bar plots active learning: 

    # activel = joblib.load('better_sampling_sizes_user.pkl')
    # activel2 = joblib.load('sample_sizes_user.pkl')


    # # # get last recommendations from batch sampling active learning
    # # for key, sub_dict in activel.items():
    # #     # Get the last dataframe from the list
    # #     dataframe_list = sub_dict['recommendation after active learning']
        
    # #     # Extract the last dataframe (if the list is not empty)
    # #     if dataframe_list:  # Ensure the list is not empty
    # #         last_dataframe = dataframe_list[-1]  # Get the last dataframe
            
    # #         # Update the dictionary value with the last dataframe only
    # #         sub_dict['recommendation after active learning'] = last_dataframe
        
        
    # variable_names = ['correct counts', 'precision','ndcg', 'hit', 'recip_rank', 'mapk']

    # for var_name in variable_names:
    #     # testingnewgraph = eval_active(activel, var_name)
    #     testingnewgraph2 = eval_active(activel2, var_name)
    #     print(testingnewgraph2)
    #     make_bars(testingnewgraph2, var_name, testingnewgraph)


# hypothesis testing, paired t test:

    activel = joblib.load('random_sizes_user.pkl')
    activel2 = joblib.load('sample_sizes_user.pkl')

    variable_names = ['correct counts', 'precision','ndcg', 'hit', 'recip_rank', 'mapk']

    for var_name in variable_names:
        testingnewgraph = eval_active(activel, var_name)
        testingnewgraph2 = eval_active(activel2, var_name)
    
        values_after_active_learningBEFORE = list(testingnewgraph[str(var_name)+ ' before active learning'].values())
        values_after_active_learningSAMPLING = list(testingnewgraph2[str(var_name)+ ' after active learning'].values())
        print(values_after_active_learningBEFORE)
        # pairedt(values_after_active_learningBEFORE, values_after_active_learningSAMPLING )


# learning curves:

    # active_learningcurce = joblib.load('random_padd_sizes_user.pkl')
    # active_learningcurce3 =  joblib.load('sample_padd_sizes_user.pkl')
    # variable_names = ['correct counts', 'precision', 'ndcg', 'hit', 'recip_rank', 'mapk']

    # for var_name in variable_names:
    #     testingnewgraph = eval_active(active_learningcurce, var_name+"2")
    #     testingnewgraph3 = eval_active(active_learningcurce3, var_name+"2")
    #     learningcurve_together([testingnewgraph, testingnewgraph3] , var_name)
    #     print(testingnewgraph)

    # active_learningcurce = joblib.load('sample_sizes_user_items40.pkl')
    # active_learningcurce3 =  joblib.load('sample_sizes_user.pkl')
    # active_learningcurcerandom = joblib.load('random_sizes_user.pkl')

  
    # # # # get the 4 sets of recommendations into 20 sets 
    # # for key, sub_dict in active_learningcurcebetter.items():
    # # # Get the last dataframe from the list
    # #     dataframe_list = sub_dict['recommendation after active learning']
        
    # #     # Duplicate each dataframe in the list five times and maintain order
    # #     duplicated_dataframes = [df for df in dataframe_list for _ in range(5)]
        
    # #     # Update the dictionary value with the duplicated dataframes list
    # #     sub_dict['recommendation after active learning'] = duplicated_dataframes


# # SAMPLE 40 ITEMS ALL ITEMS CONSIDERED -> PADD  -> joblib.dump(NAME, 'sample_allitems_40added_padd.pkl')

    # activelearnign = activelearning(folds, interaction_sequence['item_id'].unique())
    
    # joblib.dump(activelearnign, 'sample_allitems_40added_padd.pkl')

# SAMPLE 20 ITEMS ALL ITEMS CONSIDERED NOW -> PADD -> joblib.dump(NAME, 'sample_allitems_20added_padd.pkl')

#     activelearnign = activelearning(folds, interaction_sequence['item_id'].unique())
    
#     joblib.dump(activelearnign, 'sample_allitems_20added_padd.pkl')
    
# # # 4 BACTCHES SAMPLE ALL ITEMS CONSIDERED NOW -> PADD -> joblib.dump(NAME, 'sample_4batches_allitems_padd.pkl')

#     activelearnign = activelearning(folds, interaction_sequence['item_id'].unique())
    
#     joblib.dump(activelearnign, 'sample_4batches_allitems_padd.pkl')

# # 10 BACTCHES SAMPLE ALL ITEMS CONSIDERED NOW -> PADD -> joblib.dump(NAME, 'sample_10batches_allitems_padd.pkl')

#     activelearnign = activelearning(folds, interaction_sequence['item_id'].unique())
    
#     joblib.dump(activelearnign, 'sample_10batches_allitems_padd.pkl')













# THIS IS WITH OTHER CROSS VALIDATION:

    # grouped = interaction_sequence.groupby('user_id') 

    # result_df = pd.DataFrame(columns=interaction_sequence.columns)

    # for _, group in grouped:
    #     # Delete the first two items in each group
    #     modified_group = group.iloc[2:]
    #     # Append the modified group to the result DataFrame
    #     result_df = pd.concat([result_df, modified_group])


    # outcome_implicit_models = ALS(result_df, matrix)

    # Cosine is het best. I also have: implicitNN_cosine , implicitNN_tfidf, implicitNN_bm25
    # outcome_implicit_nn = implicitNN_cosine(result_df, matrix)

    # # print(outcome_implicit_nn[outcome_implicit_nn['appears_in_matrix'] == True])

    # outcome_implicit_Bay = implicitBayesian(result_df, matrix)
    
    # # print(outcome_implicit_Bay[outcome_implicit_Bay['appears_in_matrix'] == True])

    # outcome_implicit_log = implicitlogistic(result_df, matrix)
    
    # print(outcome_implicit_log[outcome_implicit_log['appears_in_matrix'] == True])

    # folds = random_cross(interaction_sequence, 5)


  


# Python boilerplate.
if __name__ == '__main__':
    main()

        