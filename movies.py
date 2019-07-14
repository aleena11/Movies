import pandas as pd
import json
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.model_selection import train_test_split
from ast import literal_eval

# =============================================================================
# Processing mode. See below accepted values and usage. 
# =============================================================================

# uncomment next code line to get the recommended top 10 movies
# results in results_top10movies.csv
PROCESSING_MODE = 1

# uncomment next code line to get the item based collaborative filter
# results in results_item_based.csv
#PROCESSING_MODE = 2

# uncomment next code line to get the content based recommendation
# results in results_content_based.csv
#PROCESSING_MODE = 3

# uncomment next code line to get the mix recommending system
# results in results_mix_matrix.csv
#PROCESSING_MODE = 4


_movies = pd.DataFrame()
_ratings = pd.DataFrame()
_ratings_movies = pd.DataFrame()
_valid_ratings = pd.DataFrame()
_train_ratings = pd.DataFrame()
_test_ratings = pd.DataFrame()
_train_movies = pd.DataFrame()
_train_movies_with_id = pd.DataFrame()
_train_corr_matrix = pd.DataFrame()
_train_cosine_sim = pd.DataFrame()
_train_mix = pd.DataFrame()
_train_matrix = pd.DataFrame()
_recommender_system = pd.DataFrame()


def read_movies():
    global _movies
    
    _movies = pd.read_csv(
        'movies_metadata.csv', usecols = ['id', 'title','genres'])

def prepare_movies():
    global _movies
    
    _movies = _movies.dropna()
    _movies["id_num"] = pd.to_numeric(_movies.id, errors='coerce')
    _movies[['id_num']] = _movies[['id_num']].astype(np.int64)

def prepare_movies_genres():
    global _movies
    
    for index, row in _movies.iterrows():
        row["genres"] = json.loads(row["genres"].replace("\'", "\"")) 

def read_ratings():
    global _ratings
    
    _ratings = pd.read_csv('ratings.csv')

def prepare_ratings():
    global _ratings

    _ratings['r_date'] = \
        pd.to_datetime(_ratings['timestamp'], unit='s').dt.floor('d')
    _ratings['r_year'] = _ratings['r_date'].dt.year

def merge_ratings_movies():
    global _ratings_movies
    
    _ratings_movies = \
        _ratings.merge( \
            _movies, \
            how='inner', \
            left_on='movieId', \
            right_on='id_num') \
        .dropna()
    #keep only common records in movies and ratings
    _ratings_movies = \
        _ratings_movies[['userId', 'movieId', 'rating','genres','title', \
            'r_year','timestamp']]

def get_top10_movies():
    """
    Question 1 - Top 10 movies 
    
    Simple recommender
    Generalized recommendations to all users based on movie popularity. 
    Movies that are popular and critically acclaimed have a higher probability
    of being liked by the average audience.
    """
    
    ratings_2010 = _ratings_movies[_ratings_movies['r_year'] > 2010]  
    # Old reviews may not be relevant and since the files are too heavy,
    # Keep only the newest ratings.
    ratings_2010_grp_by_movie = \
        ratings_2010 \
        .groupby('movieId', as_index = False) \
        .agg({'timestamp':'size', 'rating':'mean'}) \
        .rename(columns = {'timestamp':'rating_count', 'rating':'rating_avg'})
    
    ratings_2010_grp_by_movie_over_95 = \
        ratings_2010_grp_by_movie[ \
            ratings_2010_grp_by_movie['rating_count'] > \
            ratings_2010_grp_by_movie['rating_count'].quantile(0.95)] 
    
    # popularity reason  
    top_10_ratings_2010_grp_by_movie_over_95 = \
    ratings_2010_grp_by_movie_over_95 \
        .merge(_movies, how='left', left_on = 'movieId', right_on = 'id_num') \
        .dropna() \
        .nlargest(10, 'rating_avg') \
        .reset_index(drop=True)
    
    top_10_ratings_2010_grp_by_movie_over_95 = \
        top_10_ratings_2010_grp_by_movie_over_95[ \
            ['id_num', 'title', 'rating_avg' ]]
    top_10_ratings_2010_grp_by_movie_over_95.columns = \
        ['Movie ID', 'Title', 'Rating']
    top_10_ratings_2010_grp_by_movie_over_95['Rating'] = \
        top_10_ratings_2010_grp_by_movie_over_95['Rating'] \
            .apply(lambda x: round(x, 1))
    top_10_ratings_2010_grp_by_movie_over_95.to_csv('results_top10movies.csv')
    

def  filter_data():
    """    
    Remove movies that contained any NA values in the data - 
    minor impact on the overall structure and distribution of the dataset.
    Study the reliability of ratings and movies: 
    a movie must have a certain number of ratings before the overall rating 
    becomes reliable. 
    If a movie is rated just 2 times, then it might be biased and 
    not give a good representation. In addition, low popularity movies 
    may not be good recommendations. 
    Likewise, set a similar threshold for each user, because 
    if a user only rates a few movies, these ratings may not be reliable.
    We see that the data has a weak normal distribution with the mean of around 3.5. 
    There are a few outliers, no filter applied on the "rating"
    Filter the data due to a limited computational capability and 
    the previous assumptions.
    """    
     
    global _valid_ratings
    
    users_stat = \
        _ratings_movies \
        .groupby('userId', as_index = False) \
        .agg({'timestamp':'size'}) \
        .rename(columns = {'timestamp':'rating_count'})
    users_stat_80perc = \
        users_stat[ \
            users_stat['rating_count'] > \
            users_stat['rating_count'].quantile(0.80)]
    active_ratings = \
        _ratings_movies[_ratings_movies['userId'] \
            .isin(users_stat_80perc['userId'])]
    
    rated_movies = \
        active_ratings \
            .groupby('movieId', as_index=False) \
            .agg({'timestamp':'size'}) \
            .rename(columns = {'timestamp':'rating_count'})
    movie_stat_over_98perc = \
        rated_movies[ \
            rated_movies['rating_count'] > \
            rated_movies.rating_count.quantile(0.98)]
    _valid_ratings = \
        active_ratings[active_ratings['movieId'] \
            .isin(movie_stat_over_98perc['movieId'])]

    _valid_ratings['genres'] = \
        _valid_ratings['genres'] \
            .fillna('[]') \
            .apply(literal_eval) \
            .apply(lambda x: \
                [i['name'] for i in x] if isinstance(x, list) else [])
    
def split_train_test():
    global _train_ratings, _test_ratings
    
    # Question 2 - Split the dataset training/test, 60/40
    _train_ratings, _test_ratings = \
        train_test_split(_valid_ratings, test_size = 0.4)

def prepare_train_data():
    global _train_movies, _train_movies_with_id
    
    _train_movies = _train_ratings[['movieId', 'genres']]
    _train_movies.sort_values(by='movieId', inplace = True)
    _train_movies['genres'] = \
        _train_movies['genres'].apply(lambda x: ' '.join(x))
    _train_movies = _train_movies.drop_duplicates()
    _train_movies_with_id = _train_movies.copy(deep = True)
    _train_movies = _train_movies.set_index(['movieId'])

def build_correlation_matrix():
    """
    Compute the similarity matrix
    """
    
    global _train_corr_matrix
    
    train_movie_ratings = _train_ratings.pivot_table(index = \
        'userId' , columns = 'movieId', values = 'rating')
    _train_corr_matrix = \
        train_movie_ratings.corr(method = 'pearson', min_periods = 100)

def build_similarity_matrix():
    global _train_cosine_sim
    
    count_vec = \
        CountVectorizer( \
            analyzer='word', tokenizer=lambda doc: doc, lowercase=False)
    count_matrix = count_vec.fit_transform(_train_movies['genres'])
    _train_cosine_sim = cosine_similarity(count_matrix, count_matrix)
    _train_cosine_sim = \
        pd.DataFrame( \
             data = _train_cosine_sim \
             , index = _train_movies_with_id['movieId'] \
             , columns = _train_movies_with_id['movieId']) 
    
def combine_correlation_similarity():
    global _train_cosine_sim, _train_mix
    
    _train_mix = _train_cosine_sim.mul(_train_corr_matrix, fill_value = 0)
   
def build_movie_recommendation(train_matrix):
    global _recommender_system
    
    _recommender_system = pd.DataFrame()

    for test_user in set(_test_ratings['userId']):
        test_user_ratings = \
            _test_ratings[_test_ratings['userId'] == test_user] \
                [['movieId','rating']].dropna()
       
        # Find similar movies
        test_user_mix = \
            train_matrix[list(test_user_ratings['movieId'])].dropna()
        
        # Remove the movies that are already rated
        for i in test_user_ratings['movieId']:
            if i in set(test_user_mix.index):
                test_user_mix = test_user_mix.drop(i, axis = 0)
                
        # Scale the similarities using a weighted average forlmula and
        # add the movies name and calculatedRating
        # sum of corr. coefficients
        sum_test_user_corr =  test_user_mix.sum(axis = 1, skipna = True)
        sum_test_user_corr = \
            pd.DataFrame(data = sum_test_user_corr, columns = ['total_coef'])
        sum_test_user_corr['indexs'] = sum_test_user_corr.index
            
        test_user_rating_arr = np.array(test_user_ratings['rating'])
        test_user_mix_adjusted = test_user_mix.mul(test_user_rating_arr)
        
        mult_test_user_corr = \
            test_user_mix_adjusted.sum(axis = 1, skipna = True)
        mult_test_user_corr = \
            mult_test_user_corr.sort_values(ascending = False)
        mult_test_user_corr_total_score = \
            pd.DataFrame(data = mult_test_user_corr, columns = ['total_score'])
        mult_test_user_corr_total_score['indexp'] = \
            mult_test_user_corr_total_score.index
       
        
        test_user_total_score = \
            mult_test_user_corr_total_score \
                .merge(_movies, how = 'left', \
                   left_on ='indexp', right_on ='id_num') \
                .dropna()
        test_user_total_score = \
            test_user_total_score \
                .merge(sum_test_user_corr, how = 'left', \
                   left_on ='indexp', right_on ='indexs') \
                .dropna()
        test_user_calc_rating = \
            test_user_total_score.total_score / \
            test_user_total_score.total_coef
        test_user_total_score['calculatedRating'] = test_user_calc_rating 
        test_user_total_score = \
            test_user_total_score[['title', 'calculatedRating', 'id_num']] \
                .nlargest(10, 'calculatedRating') \
                .reset_index(drop = True)
        test_user_total_score['userId'] = test_user
        
        # Add the user recommendation to the list    
        _recommender_system = _recommender_system.append(test_user_total_score)   
        
    _recommender_system.columns = ['Title','Rating','Movie ID', 'User ID']
    _recommender_system['Rating'] = \
        _recommender_system['Rating'].apply(lambda x: round(x, 1))
    
def recommend_movies():
    if PROCESSING_MODE == 1:
        read_movies()
        prepare_movies()
        
        read_ratings()
        prepare_ratings()
        
        merge_ratings_movies()
        
        get_top10_movies()
    elif PROCESSING_MODE == 2:
        """
        Recomendations made using an Item-based Collaborative Filtering and 
        Content Based REccomender System:
        Based on the assumption that people who agreed 
        in their assessment of some items in the past 
        are expected to come to the same consensus again in the future,
        especially if the items belong to the same genre.
        """
        
        read_movies()
        prepare_movies()
        
        read_ratings()
        prepare_ratings()
        
        merge_ratings_movies()
        
        filter_data()
        split_train_test()
        prepare_train_data()
        
        build_correlation_matrix()
        
        build_movie_recommendation(_train_corr_matrix)
        
        _recommender_system.to_csv('results_item_based.csv')
    elif PROCESSING_MODE == 3:
        """
        Content-based recommenders: suggest similar items based on
        a particular item - movie genre in our case. 
        If a person liked a particular movie, they will also like a movie 
        of the same genre / genre mix.
        To further improve the accuracy we should consider the inclusion of
        additional movie attributes such as language, release date, name 
        or parsing semantic textual data such as plot description.
        The quality of the recommender would be increased with the usage of
        better metadata
        """
        
        read_movies()
        prepare_movies()
        prepare_movies_genres()
        
        read_ratings()
        prepare_ratings()
        
        merge_ratings_movies()
        
        filter_data()
        split_train_test()
        prepare_train_data()
        
        build_similarity_matrix()
        
        build_movie_recommendation(_train_cosine_sim)
        
        _recommender_system.to_csv('results_content_based.csv')
    elif PROCESSING_MODE == 4:
        """
        Recomendations made using an Item-based Collaborative Filtering and
        Content Based REccomender System:
        Based on the assumption that people who agreed in their assessment 
        of some items in the past are expected to come to 
        the same consensus again in the future especially 
        if the items belong to the same genre.
        """
        
        read_movies()
        prepare_movies()
        prepare_movies_genres()
        
        read_ratings()
        prepare_ratings()
        
        merge_ratings_movies()
        
        filter_data()
        split_train_test()
        prepare_train_data()
        
        build_correlation_matrix()
        build_similarity_matrix()
        combine_correlation_similarity()
        
        build_movie_recommendation(_train_mix)

        _recommender_system.to_csv('results_mix_matrix.csv')


recommend_movies()