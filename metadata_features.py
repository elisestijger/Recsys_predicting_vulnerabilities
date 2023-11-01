import pandas as pd
import numpy as np
from collections import Counter
from scipy.stats import skew, kurtosis, entropy


class MetaFeatures:
    def __init__(self, data):
        self._calculate_meta_features(data)  # Calculate meta-features when the object is initialized

    def _calculate_meta_features(self, data):
        self.min_rating = data["rating"].min()
        self.max_rating = data["rating"].max()
        self.mean_rating = data["rating"].mean()
        self.median_rating = data["rating"].median()
        self.mode_rating = data["rating"].mode()[0]
        user_counter = Counter(data["Name_user"])
        item_counter = Counter(data["Name_item"])
        num_users = data["Name_user"].unique().size
        num_items = data["Name_item"].unique().size
        num_instances = len(data)
        density = (num_instances * 100) / (num_users * num_items)
        sparsity = 100 - density
        self.user_item_ratio = num_users / num_items
        self.item_user_ratio = num_items / num_users
        unique_ratings, rating_counts = np.unique(data["rating"], return_counts=True)
        increments = []
        for idx in range(len(unique_ratings) - 1):
            increments.append(abs(unique_ratings[idx + 1] - unique_ratings[idx]))
        self.highest_num_rating_by_single_user = user_counter.most_common()[0][1]
        self.lowest_num_rating_by_single_user = user_counter.most_common()[-1][1]
        self.highest_num_rating_on_single_item = item_counter.most_common()[0][1]
        self.lowest_num_rating_on_single_item = item_counter.most_common()[-1][1]
        self.mean_num_ratings_by_user = num_instances / num_users
        self.mean_num_ratings_on_item = num_instances / num_items
        self.rating_skew = skew(data["rating"])
        self.rating_kurtosis = kurtosis(data["rating"])
        self.ratings_standard_variation = data["rating"].std()
        self.rating_variance = data["rating"].var()
        self.rating_entropy = entropy(rating_counts) / np.log(len(rating_counts))
        self.num_possible_ratings = len(data["rating"].unique())
        self.rating_average_increment = sum(increments) / len(increments)

        self.meta_features_df = pd.DataFrame({
            'Feature': ['min_rating', 'max_rating', 'mean_rating',
                        'user_item_ratio', 'item_user_ratio', 'number_ratings_by_single_user',
                        'number_ratings_on_single_item',
                         'rating_skew', 'rating_kurtosis',
                     
                        'num_possible_ratings'],
            'Value': [self.min_rating, self.max_rating, self.mean_rating, self.user_item_ratio, self.item_user_ratio,
                      self.highest_num_rating_by_single_user, 
                      self.highest_num_rating_on_single_item, 
                      self.rating_skew, self.rating_kurtosis, self.num_possible_ratings]
        })

    def to_dataframe(self):
        return self.meta_features_df

def calculate_meta_features(data: pd.DataFrame) -> MetaFeatures:
    return MetaFeatures(data)


# meta_data = calculate_meta_features(all_combinations).to_dataframe

# Print the DataFrame
# print(meta_features_df)