import os
import sys
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
import math
import statistics
from datetime import datetime

from surprise import SVD
from surprise import Dataset
from surprise import accuracy
from surprise import Reader
from surprise.model_selection import train_test_split
import csv


def testactivelearning(df, folds):
    # df = pd.read_csv('movielens-100k-dataset/ml-100k/udata.csv',sep=";", header=0, engine="python")
    user = pd.read_csv('movielens-100k-dataset/ml-100k/uuser.csv',sep=";", header=0, engine ="python")
    genre = pd.read_csv('movielens-100k-dataset/ml-100k/ugenre.csv',sep=";", header=0, engine = "python")

    # 16 gaps of data

    data = []
    for i in range(17):
        data.append((i*5000))
    data[0] = 1


    # A reader is still needed but only the rating_scale param is requiered.

    reader = Reader(rating_scale=(1, 5))

    # We'll use the famous SVD algorithm.

    algo = SVD()
    
    df['item'] = df['item'].astype(int)
    df['user'] = df['user'].astype(int)


    # Active learning 4 - Attention Based Popularity
    # make matrix from interaction sequence
    df_1 = df.pivot(index = 'user', columns ='item', values = 'rating')
    df_1 = df_1.reindex(sorted(df_1.columns), axis=1)

    # df_1.to_csv('/Users/elisestijger/Desktop/testdf2.csv')

    # print per user the row values from the matrix -> every user has a list with the ratings per item 
    df_numpy = df_1.values

    # print((df_numpy[0]))

    rmse_al = []
    rmse_ran = []

    tstart = datetime.now()

    numberCol = []
    # column = for every item the ratings per user > lenght = 3513
    for column in df_numpy.T:
        i = 0
        for eachValue in column:
            if eachValue == 1 or eachValue == 2 or eachValue == 3 or eachValue == 4 or eachValue == 5:
                i=i+1
        numberCol.append(i)

    #numberCol = per item how many users have rated this item 

    # every item has the number of users it 'appears in' 
    item = np.array(numberCol)

    item = pd.DataFrame(item, columns = ["item"])

    # # take the most popular item / order it 

    item = item.sort_values(by=["item"], ascending = False)

    # index values for all items 
    index = item.index.tolist()

    trainact = pd.DataFrame(columns=['user','item','rating'])

    for i in index:
        trainact = trainact.append(df.loc[df['item']==i],ignore_index = True)
        # print(trainact)


    # train act still has all the items and users. 
    trainact = trainact.drop_duplicates(keep ='first')

    # print(trainact[trainact['item'] == 5])
    # trainact = all the items that appear the most plus their users in a df

    tend = datetime.now()
    for i in folds:
        algo = SVD()
        algoran = SVD()

        # # randomly get 2000 rows from the df as testset 
        test = folds[i]['validation'][['item_id', 'user_id', 'rating']]
        train = folds[i]['train'][['item_id', 'user_id', 'rating']]
        
        test.rename(columns={'item_id': 'item', 'user_id': 'user'}, inplace=True)
        train.rename(columns={'item_id': 'item', 'user_id': 'user'}, inplace=True)

        train = Dataset.load_from_df(train[['user', 'item', 'rating']], reader).build_full_trainset()   
        test = Dataset.load_from_df(test[['user', 'item', 'rating']], reader).build_full_trainset()

        # # add these random rows to the most popular items in trainact 
        # trainact1 = pd.concat([test,trainact]).drop_duplicates(keep=False)
        # trainact1 = trainact1.head(i)

        # # use the whole dataset and test is random 
        # train = pd.concat([df,test]).drop_duplicates(keep=False)
        # train = train.sample(n = i)

        # trainsetact = Dataset.load_from_df(trainact1[['user', 'item', 'rating']], reader).build_full_trainset()
        # trainset = Dataset.load_from_df(train[['user', 'item', 'rating']],reader).build_full_trainset()
        # testset = Dataset.load_from_df(test[['user', 'item', 'rating']], reader).build_full_trainset().build_testset()

        algo.fit(train)
        predictions = algo.test(test)

        rmse_al.append(accuracy.rmse(predictions, verbose=False))

        algoran.fit(train)
        predictionsran = algoran.test(test)

    #     rmse_ran.append(accuracy.rmse(predictionsran, verbose=False))

    # print(rmse_al)
    # print(" Active learning in ms : ")
    # print(tend-tstart)
    # print(rmse_ran)

    # plt.plot(data,rmse_al,'r')
    # plt.plot(data,rmse_ran,'b')
    # plt.axis([0,80000,0.5,1.5])
    # plt.title('Attention Based Popularity & Random - RMSE')
    # plt.xlabel('# data in the trainset')
    # plt.ylabel('RMSE')
    # plt.legend(['Active Learning RMSE', 'Random RMSE'], loc='upper left')
    # plt.show()
