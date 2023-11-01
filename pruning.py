import numpy as np
import pandas as pd     #need version 1.4.0 ! 

def prune(df):

    # while number of items and users is different from number of desired items and users, do the pruning
    while(len(count_items(df)[0]) != len(count_items(df)[2]) or len(count_users(df)[0]) != len(count_users(df)[2])):

        if len(count_items(df)[0]) != len(df):
            # make def to delete items 
            df = delete_items(df)
        
        if len(count_users(df)[0]) != len(df):
            #  make def to delete users 
            df = delete_users(df)

    return df

def count_items(df):
    #count items with more / 5 users, count items with less than 5 users and number of items in dataframe
    cwe_unique_items_count = df.groupby('cwe')['product'].nunique()
    cwe_unique_items_count = cwe_unique_items_count.sort_values(ascending=False)
    length_df = cwe_unique_items_count
    selected_items = cwe_unique_items_count[cwe_unique_items_count >= 5]
    items_to_delete = cwe_unique_items_count[cwe_unique_items_count < 5].index
    selected_items = selected_items.reset_index()
    selected_items = selected_items.rename(columns={'index': 'cwe'})

    return selected_items, items_to_delete, length_df

def count_users(df):
    #count users with more / 5 items, count users with less than 5 items and number of users in dataframe
    cwe_unique_products_count = df.groupby('product')['cwe'].nunique()
    cwe_unique_products_count = cwe_unique_products_count.sort_values(ascending=False)
    length_df = cwe_unique_products_count
    selected_users = cwe_unique_products_count[cwe_unique_products_count >= 5]
    users_to_delete = cwe_unique_products_count[cwe_unique_products_count < 5].index
    selected_users = selected_users.reset_index()
    selected_users = selected_users.rename(columns={'index': 'product'})
    
    return selected_users, users_to_delete, length_df

def delete_items(df):

    items_to_delete = count_items(df)[1]

    # Clean 'cwe' values by removing whitespace
    df['cwe'] = df['cwe'].str.strip()

    # Convert 'cwe_to_delete' values to the same format
    items_to_delete_cleaned = [cwe.strip() for cwe in items_to_delete]

    # Use boolean indexing to delete rows with specified CWEs
    new_df = df[~df['cwe'].isin(items_to_delete_cleaned)]

    return new_df 

def delete_users(df):
    
    users_to_delete = count_users(df)[1]

    # Clean 'cwe' values by removing whitespace
    df['product'] = df['product'].str.strip()

    # Convert 'cwe_to_delete' values to the same format
    users_to_delete_cleaned = [product.strip() for product in users_to_delete]

    # Use boolean indexing to delete rows with specified CWEs
    new_df = df[~df['product'].isin(users_to_delete_cleaned)]

    return new_df 