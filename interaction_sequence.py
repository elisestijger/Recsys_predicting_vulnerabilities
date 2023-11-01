import pandas as pd     #need version 1.4.0 ! 
import numpy as np

def make_interaction_sequence(matrixCWE):

    interaction_sequence1_all = make_interaction_sequence1_0(matrixCWE, 1)
    interaction_sequence1 = interaction_sequence1_all[0]        #make_interaction_sequence1_0(matrixCWE, 1)[0]
    # interaction_sequence0_all = make_interaction_sequence1_0(matrixCWE, 0)
    # interaction_sequence0 =  interaction_sequence0_all[0]               #make_interaction_sequence1_0(matrixCWE, 0)[0]

    # merge the interaction sequence from 1 and 0 
    Interaction_sequence_full = interaction_sequence1 #+ interaction_sequence0
    # Interaction_sequence_full = np.random.shuffle(Interaction_sequence_full)
    Interaction_sequence_full = pd.DataFrame(Interaction_sequence_full)

    # rename the columns to user_id, item_id and rating
    Interaction_sequence_full.rename(columns = {0:'user_id', 1:'item_id', 2:'rating'}, inplace = True)

    # do the mapping
    user_mapping = interaction_sequence1_all[1]
    item_mapping = interaction_sequence1_all[2]
    
    Interaction_sequence_full_mapping =  mapping(user_mapping,item_mapping,Interaction_sequence_full)

    return Interaction_sequence_full_mapping


def make_interaction_sequence1_0(matrixCWE, zero_or_one):
    rating_matrix = matrixCWE          
    interaction_sequence = []
    num_users, num_items = rating_matrix.shape
    product_mapping = {}
    user_mapping = {}
    user_index = 0
    users = []
    items = []

    #iterate through the matrix and make mappings for users and items -> PMF handles integers
    
    if zero_or_one == 1:
        for user_id, col in rating_matrix.iterrows():
            if user_id not in user_mapping:
                user_mapping[user_id] = user_index
                user_index += 1

            for item_id, rating in col.iteritems():
                if rating == 1:
                    item_index = rating_matrix.columns.get_loc(item_id)
                    if item_index not in product_mapping:
                        product_mapping[item_index]= item_id
                    
                    users.append(user_mapping[user_id])
                    items.append(item_index)
                    interaction_sequence.append((user_mapping[user_id],item_index))
    
        # Insert the value 1 for every tuple in the list and make the type of the first two values a string and the last one a float
        interaction_sequence_full = [(x[0], x[1], 1) for x in interaction_sequence]
        interaction_sequence_full = [(str(x[0]), str(x[1]), float(x[2])) for x in interaction_sequence_full]

    # if zero_or_one == 0:
    #     for user_id, col in rating_matrix.iterrows():
    #         if user_id not in user_mapping:
    #             user_mapping[user_id] = user_index
    #             user_index += 1

    #         for item_id, rating in col.iteritems():
    #             if rating == 0:
    #                 item_index = rating_matrix.columns.get_loc(item_id)

    #                 if item_index not in product_mapping:
    #                     product_mapping[item_index]= item_id

    #                 users.append(user_mapping[user_id])
    #                 items.append(item_index)
    #                 interaction_sequence.append((user_mapping[user_id],item_index))
        
    #     # Insert the value 0 for every tuple in the list and make the type of the first two values a string and the last one a float
    #     interaction_sequence_full = [(x[0], x[1], 0) for x in interaction_sequence]
    #     interaction_sequence_full = [(str(x[0]), str(x[1]), float(x[2])) for x in interaction_sequence_full]

    return interaction_sequence_full, user_mapping, product_mapping

def mapping(user_mapping,item_mapping, Interaction_sequence_full):
    # Convert user_mapping dictionary to a DataFrame
    user_mapping_df = pd.DataFrame(user_mapping.items(), columns=['Name', 'Id'])

    # Convert item_mapping dictionary to a DataFrame and change column sequence
    item_mapping_df = pd.DataFrame(item_mapping.items(), columns=['Id', 'Name'])
    item_mapping_df = item_mapping_df.iloc[:,[1,0]]

    # rename the column names
    item_mapping_df.rename(columns = {'Id':'Id_item'}, inplace = True)
    user_mapping_df.rename(columns = {'Id':'Id_user'}, inplace = True)

    item_mapping_df.rename(columns = {'Name':'Name_item'}, inplace = True)
    user_mapping_df.rename(columns = {'Name':'Name_user'}, inplace = True)

    user_mapping_df['Id_user'] = user_mapping_df['Id_user'].astype(str)
    item_mapping_df['Id_item'] = item_mapping_df['Id_item'].astype(str)

    
    Interaction_sequence_full2 = Interaction_sequence_full.copy()
    # merge the user_mapping and item_mapping together to get names of the ids in one df
    Interaction_sequence_full2 = Interaction_sequence_full2.join(user_mapping_df.set_index('Id_user'), on=['user_id'])
    Interaction_sequence_full2 = Interaction_sequence_full2.join(item_mapping_df.set_index('Id_item'), on=['item_id'])

    return Interaction_sequence_full2, user_mapping_df, item_mapping_df