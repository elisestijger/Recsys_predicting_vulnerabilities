# HOW TO USE

The whole code can be executed with main.py. Please note this is not the final version and various implementation of the algorithms should be uncommented in the main.py file to run it. 
Comments in the main.py will guide you. 

## Implementing active learning
In the main.py file there are mulitple comments for implementing active learning, make sure to also uncomment the desired selection in the active_learning.py file. You can also use the pcikle files where the various active learning implementations are already used to make predictions with.

## Using pickle 
The .pkl files are stored data files used for the project. Different implementations of active learning are stored in different pickle files:

random 20 items = random_sizes_user.pkl AND random_padd_sizes_user.pkl
single batch 20 items  = sample_allitems_20added_padd
single batch 40 items = sample_allitems_40added_padd.pkl 
4 batch 20 items = sample_4batches_allitems_padd_20ITEMS.pkl
10 batch 20 items = sample_10batches_allitems_padd_20ITEMS.pkl
4 batch 40 items = sample_4batches_allitems_padd_40ITEMSNEW.pkl
10 batch 40 items = sample_10batches_allitems_padd_40ITEMS.pkl
