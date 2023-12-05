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
from Copy_ALSimplicit_models import start_test
from evaluations import eval
from REALrandom_cross_val import random_cross_fold
from REALlenskit_modules import make_recommendations
from graphs import make_graphs, make_graphs_together, learningcurve,learningcurve_together, make_boxplots, boxplots2, make_bars, make_bars2
import matplotlib.pyplot as plt
from active_learning import activelearning
from lenskit.algorithms import Recommender, als, user_knn as knn_user
from lenskit.algorithms import Recommender, als, user_knn as knn_user
import random 
from test_activel import * 
from ndcg import ndcg , idcg , dcg
from evaluations_active_learning import eval_active
from Statistical_hypothesis import pairedt, wilcoxon_test, anovatest
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pylab
from scipy.stats import shapiro, ttest_rel
from scipy.stats import friedmanchisquare


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


    # following marked section is used to retrieve the data. This part can be skipped and the interaction_sequence_correct.pkl can be used for the code.
    # -------------------------------------

    # load data 

    # cve_all = get_data(year1,year2)
    # cve_all = grab_data()

    # # # change data 

    # cwe_all = extract_cwe(cve_all)
    
    # # # pruning

    # cwe_all_pruned = prune(cwe_all)


    # # # matrix
 
    # matrix = make_matrix(cwe_all_pruned)
    # matrix = load_matrix()
    # matrix = joblib.load('matrix.pkl')

    # # # interaction sequence

    # interaction_sequence = make_interaction_sequence(matrix)[0]
    # user_mapping = make_interaction_sequence(matrix)[1]
    # item_mapping = make_interaction_sequence(matrix)[2]

    interaction_sequence = joblib.load('interaction_sequence_correct.pkl')

   # -------------------------------------


# random cross validation --> only making the folds: df with n folds each fold has a train and a test part
    
    # folds = random_cross_fold(interaction_sequence, 5)

    folds = joblib.load('next_folds.pkl')


# Implement the recommender systems with Lenskit and make recommendations

    # predicted_recommendations_lenskit = make_recommendations(folds, 5)           

    # fold_data = predicted_recommendations_lenskit[2]

     fold_data_lenskit = joblib.load('fold_data_all_algorithms_newPOP_both_pops.pkl')


# evaluations   -> change 'precision' into the desired evaluation metric

    # evaluation_lenskit = eval(fold_data_lenskit, interaction_sequence, 'precision')



# graphs recommender systems 

    # graph_lenskit = make_graphs(evaluation_lenskit)

    # graph_lenskit.show()


#  bar plots for multiple recommender systems

    # variable_names = ['correct counts', 'precision', 'ndcg', 'hit', 'recip_rank', 'mapk']
    # for var_name in variable_names:
    #     testingnewgraph = eval(fold_data_lenskit, interaction_sequence, var_name)
    #     graph_lenskit = make_graphs(testingnewgraph, var_name)
    #     graph_lenskit.show()



# Active learning with various sampling techniques, uncomment the desired one and uncomment the same in the active_learning.py file:

# # SAMPLE 40 ITEMS -> 'sample_allitems_40added_padd.pkl'

    # activelearnign = activelearning(folds, interaction_sequence['item_id'].unique())
    
# SAMPLE 20 ITEMS --> 'sample_allitems_20added_padd.pkl'

#     activelearnign = activelearning(folds, interaction_sequence['item_id'].unique())
    
    
# # # 4 BACTCHES SAMPLE --> 'sample_4batches_allitems_padd.pkl'

#     activelearnign = activelearning(folds, interaction_sequence['item_id'].unique())
    

# # 10 BACTCHES SAMPLE  -> 'sample_10batches_allitems_padd.pkl'

#     activelearnign = activelearning(folds, interaction_sequence['item_id'].unique())
    

# bar plots active learning: 

    # activel0 = joblib.load('random_sizes_user.pkl')      
    # activel = joblib.load('sample_allitems_20added_padd.pkl')      
    # activel1 = joblib.load('sample_allitems_40added_padd.pkl')      
    # activel2 = joblib.load('sample_4batches_allitems_padd_20ITEMS.pkl')
    # activel3 = joblib.load('sample_10batches_allitems_padd_20ITEMS.pkl')
    # activel4 = joblib.load('sample_4batches_allitems_padd_40ITEMSNEW.pkl')
    # activel5 = joblib.load('sample_10batches_allitems_padd_40ITEMS.pkl')

    # format the data:

    # # get last recommendations from batch sampling active learning
    # for key, sub_dict in activel.items():
    #     # Get the last dataframe from the list
    #     dataframe_list = sub_dict['recommendation after active learning']
        
    #     # Extract the last dataframe (if the list is not empty)
    #     if dataframe_list:  # Ensure the list is not empty
    #         last_dataframe = dataframe_list[-1]  # Get the last dataframe
            
    #         # Update the dictionary value with the last dataframe only
    #         sub_dict['recommendation after active learning'] = last_dataframe

    # for key, sub_dict2 in activel2.items():
    #     # Get the last dataframe from the list
    #     dataframe_list2 = sub_dict2['recommendation after active learning']
        
    #     # Extract the last dataframe (if the list is not empty)
    #     if dataframe_list2:  # Ensure the list is not empty
    #         last_dataframe2 = dataframe_list2[-1]  # Get the last dataframe
            
    #         # Update the dictionary value with the last dataframe only
    #         sub_dict2['recommendation after active learning'] = last_dataframe2

    # for key, sub_dict1 in activel1.items():
    #     # Get the last dataframe from the list
    #     dataframe_list1 = sub_dict1['recommendation after active learning']
        
    #     # Extract the last dataframe (if the list is not empty)
    #     if dataframe_list1:  # Ensure the list is not empty
    #         last_dataframe1= dataframe_list1[-1]  # Get the last dataframe
            
    #         # Update the dictionary value with the last dataframe only
    #         sub_dict1['recommendation after active learning'] = last_dataframe1
    
    # for key, sub_dict3 in activel3.items():
    #     # Get the last dataframe from the list
    #     dataframe_list3 = sub_dict3['recommendation after active learning']
        
    #     # Extract the last dataframe (if the list is not empty)
    #     if dataframe_list3:  # Ensure the list is not empty
    #         last_dataframe3= dataframe_list3[-1]  # Get the last dataframe
            
    #         # Update the dictionary value with the last dataframe only
    #         sub_dict3['recommendation after active learning'] = last_dataframe3
           
    # for key, sub_dict4 in activel4.items():
    #     # Get the last dataframe from the list
    #     dataframe_list4 = sub_dict4['recommendation after active learning']
        
    #     # Extract the last dataframe (if the list is not empty)
    #     if dataframe_list4:  # Ensure the list is not empty
    #         last_dataframe4 = dataframe_list4[-1]  # Get the last dataframe
            
    #         # Update the dictionary value with the last dataframe only
    #         sub_dict4['recommendation after active learning'] = last_dataframe4
           
    # for key, sub_dict5 in activel5.items():
    #     # Get the last dataframe from the list
    #     dataframe_list5 = sub_dict5['recommendation after active learning']
        
    #     # Extract the last dataframe (if the list is not empty)
    #     if dataframe_list5:  # Ensure the list is not empty
    #         last_dataframe5 = dataframe_list5[-1]  # Get the last dataframe
            
    #         # Update the dictionary value with the last dataframe only
    #         sub_dict5['recommendation after active learning'] = last_dataframe5

    # variable_names = ['correct counts', 'precision','ndcg', 'hit', 'recip_rank', 'mapk']

    # for var_name in variable_names:
    #     testingnewgraph0 = eval_active(activel0, var_name)
    #     testingnewgraph = eval_active(activel, var_name)
    #     testingnewgraph1 = eval_active(activel1, var_name)
    #     testingnewgraph2 = eval_active(activel2, var_name)
    #     testingnewgraph3 = eval_active(activel3, var_name)
    #     testingnewgraph4 = eval_active(activel4, var_name)
    #     testingnewgraph5 = eval_active(activel5, var_name)
      
    # format data:
    #     testingnewgraph0 = {key: {f'fold {i}': value for i, value in enumerate(sub_dict.values())} for key, sub_dict in testingnewgraph0.items()}
    #     testingnewgraph = {key: {f'fold {i}': value for i, value in enumerate(sub_dict.values())} for key, sub_dict in testingnewgraph.items()}
    #     testingnewgraph1 = {key: {f'fold {i}': value for i, value in enumerate(sub_dict.values())} for key, sub_dict in testingnewgraph1.items()}
    #     testingnewgraph2 = {key: {f'fold {i}': value for i, value in enumerate(sub_dict.values())} for key, sub_dict in testingnewgraph2.items()}
    #     testingnewgraph3 = {key: {f'fold {i}': value for i, value in enumerate(sub_dict.values())} for key, sub_dict in testingnewgraph3.items()}
    #     testingnewgraph4 = {key: {f'fold {i}': value for i, value in enumerate(sub_dict.values())} for key, sub_dict in testingnewgraph4.items()}
    #     testingnewgraph5 = {key: {f'fold {i}': value for i, value in enumerate(sub_dict.values())} for key, sub_dict in testingnewgraph5.items()}


    # #     make_bars2(testingnewgraph, var_name, testingnewgraph1, testingnewgraph2,testingnewgraph4, testingnewgraph3, testingnewgraph5, testingnewgraph0 )


# Wilocoxon test
    # fill in the datasets you want to compare with the Wilcoxon test:

    activelsample40 = joblib.load('sample_10batches_allitems_padd_40ITEMS.pkl')
    activelsample20 = joblib.load('sample_allitems_40added_padd.pkl')


    # for key, sub_dict in activelsample40.items():
    #     # Get the last dataframe from the list
    #     dataframe_list = sub_dict['recommendation after active learning']
        
    #     # Extract the last dataframe (if the list is not empty)
    #     if dataframe_list:  # Ensure the list is not empty
    #         last_dataframe = dataframe_list[-1]  # Get the last dataframe
            
    #         # Update the dictionary value with the last dataframe only
    #         sub_dict['recommendation after active learning'] = last_dataframe
        
    # for key, sub_dict in activelsample20.items():
    #     # Get the last dataframe from the list
    #     dataframe_list = sub_dict['recommendation after active learning']
        
    #     # Extract the last dataframe (if the list is not empty)
    #     if dataframe_list:  # Ensure the list is not empty
    #         last_dataframe = dataframe_list[-1]  # Get the last dataframe
            
    #         # Update the dictionary value with the last dataframe only
    #         sub_dict['recommendation after active learning'] = last_dataframe
        

    # variable_names = ['correct counts', 'precision','ndcg', 'hit', 'recip_rank', 'mapk']

    # for var_name in variable_names:
    #     testingnewgraph20 = eval_active(activelsample20, var_name)  
    #     testingnewgraph40 = eval_active(activelsample40, var_name)


    #     values_after_active_learning20 = list(testingnewgraph20[str(var_name)+ ' before active learning'].values())
    #     values_after_active_learning40 = list(testingnewgraph40[str(var_name)+ ' after active learning'].values())

    #     values_after_active_learningBEFORE = list(testingnewgraph20[str(var_name)+ ' after active learning'].values())
    #     values_after_active_learningSAMPLING = list(testingnewgraph40[str(var_name)+ ' after active learning'].values())
        

    #     wilcoxon_test(values_after_active_learning20, values_after_active_learning40 )

# graphs with learning curves:

    # activel = joblib.load('sample_allitems_20added_padd.pkl')      
    # activel1 = joblib.load('sample_allitems_40added_padd.pkl')      
    # activel2 = joblib.load('sample_4batches_allitems_padd_20ITEMS.pkl')
    # activel3 = joblib.load('sample_10batches_allitems_padd_20ITEMS.pkl')
    # activel4 = joblib.load('sample_4batches_allitems_padd_40ITEMSNEW.pkl')
    # activel5 = joblib.load('sample_10batches_allitems_padd_40ITEMS.pkl')

    # # for key, sub_dict in activel2.items():
    # # # Get the last dataframe from the list
    # #     dataframe_list = sub_dict['recommendation after active learning']
        
    # #     # Duplicate each dataframe in the list five times and maintain order
    # #     duplicated_dataframes = [df for df in dataframe_list for _ in range(20)]
        
    # #     # Update the dictionary value with the duplicated dataframes list
    # #     sub_dict['recommendation after active learning'] = duplicated_dataframes

    # variable_names = ['correct counts', 'precision', 'ndcg', 'hit', 'recip_rank', 'mapk']

    # for var_name in variable_names:
    #     testingnewgraph = eval_active(activel, var_name+"2")
    #     testingnewgraph3 = eval_active(activel2, var_name+"2")

    #     # testingnewgraph = {key: {f'fold {i}': value for i, value in enumerate(sub_dict.values())} for key, sub_dict in testingnewgraph.items()}
    #     # testingnewgraph3 = {key: {f'fold {i}': value for i, value in enumerate(sub_dict.values())} for key, sub_dict in testingnewgraph3.items()}

    #     if var_name == 'correct counts':
    #         testingnewgraph3 = {
    #             key: {f'fold {int(k.split()[1])}': v for k, v in sub_dict.items()} if key.startswith(var_name) else sub_dict
    #             for key, sub_dict in testingnewgraph3.items()
    #         }
    #         testingnewgraph = {
    #             key: {f'fold {int(k.split()[1])}': v for k, v in sub_dict.items()} if key.startswith(var_name) else sub_dict
    #             for key, sub_dict in testingnewgraph.items()
    #         }
    #     else:
    #         testingnewgraph3 = {
    #             key: {
    #                 f'fold {int(k.split("fold")[-1]) - 1  if "fold" in k else int(k.split("fold ")[-1])}': v
    #                 for k, v in sub_dict.items()
    #             } if key.startswith(var_name) else sub_dict
    #             for key, sub_dict in testingnewgraph3.items()
    #         }
    #         testingnewgraph = {
    #             key: {
    #                 f'fold {int(k.split("fold")[-1]) - 1 if "fold" in k else int(k.split("fold ")[-1])}': v
    #                 for k, v in sub_dict.items()
    #             } if key.startswith(var_name) else sub_dict
    #             for key, sub_dict in testingnewgraph.items()
    #         }
        
    
    #     learningcurve_together([testingnewgraph3, testingnewgraph] , var_name)





# Python boilerplate.
if __name__ == '__main__':
    main()

        
