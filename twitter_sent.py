# all libraries required
import tweepy
import re
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import os
import toml
#suppress the unneccessary warnings
import warnings
warnings.filterwarnings('ignore')
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.corpus import stopwords
#nltk.download('averaged_perceptron_tagger')
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from textblob import Word, TextBlob

from wordcloud import WordCloud,STOPWORDS

from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load secrets from the TOML file
secrets = toml.load("secrets.toml")

# Initialize a global variable for users
users = None
bearer_token= secrets["twitter"]["bearer_token"]
# Authenticate with Twitter API v2 (Replace with your own bearer token)
client = tweepy.Client(bearer_token=bearer_token)

def get_user_data(username):
    global users
    try:
        # Get user data
        users = client.get_users(usernames=[username], user_fields='public_metrics')
        
        # Check if user data is available
        if not users.data:
            raise Exception("User not found.")

        user_id = users.data[0].id

        # Get user's tweets
        tweets = client.get_users_tweets(
            id=user_id,
            max_results=100,
            exclude="retweets",
            tweet_fields=['public_metrics', 'created_at', 'text']
        )

        return tweets

    except tweepy.TweepError as e:
        raise Exception(f"Error: {str(e)}")

#function to clean the tweets and load them into a DataFrame
def tweetsETL(tweets):
    stop_words = stopwords.words('english')
    custom_stopwords = ['RT', 'lol', 'dey','wey','im','go','get','na','us','sey', 'make we','don','wetin', 'amp']
    result = []

    #regex function to clean the tweet text from haashtags, mentions and links
    def cleanTweets(text):
        clean_text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
        return clean_text

    #further process
    def preprocess_tweets(tweet, custom_stopwords):
        processed_tweet = tweet
        processed_tweet = " ".join(word for word in processed_tweet.split() if word not in stop_words)
        processed_tweet = " ".join(word for word in processed_tweet.split() if word not in custom_stopwords)
        processed_tweet = " ".join(Word(word).lemmatize() for word in processed_tweet.split() if len(word) >= 2)
        return(processed_tweet)

    #function to unpack the tweets list into a dataframe
    for tweet in tweets.data:
            result.append({'id': tweet.id,
                           'text': tweet.text,
                           'clean_tweet' : preprocess_tweets(cleanTweets(tweet.text), custom_stopwords),
                           'created_at': tweet.created_at,
                           'source':tweet.source,
                           'retweets': tweet.public_metrics['retweet_count'],
                           'replies': tweet.public_metrics['reply_count'],
                           'likes': tweet.public_metrics['like_count'],
                           'quote_count': tweet.public_metrics['quote_count']
                      })

    df = pd.DataFrame(result)
    return df

#using a transformers model "bert" to perform the sentiment analysis on the clean_tweets column.
def sentimentAnalysis(df):
    tokenizer = AutoTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    model = AutoModelForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
    classifier = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    res = df['clean_tweet'].apply(lambda x: classifier(x[:512]))
    return res

## function to add the list resulting from the analysis to the original dataframe as score, sentiment and stars
#the sentiment is either negative, positive or neutral, and the number of stars go from 1 to 5
#1 being the most negative sentiment and 5 being the most positive
def sentimentToDf(df,res):
    tweets_stars = []
    tweets_scores = []
    tweets_sentiment = []
    #looping over the list of result to unpack it into the original tweets dataframe
    for i in range(res.size):
        tweets_stars.append(int(float(res[i][0]['label'].split()[0])))
        tweets_scores.append(res[i][0]['score'])
        if res[i][0]['label'] == '4 stars' or res[i][0]['label'] == '5 stars':
            tweets_sentiment.append('positive')
        elif res[i][0]['label'] == '1 star' or res[i][0]['label'] == '2 stars':
            tweets_sentiment.append('negative')
        else :
            tweets_sentiment.append('neutral')
    df['scores'] = tweets_scores
    df['sentiment'] = tweets_sentiment
    df['stars'] = tweets_stars
    return df

#fucntion to Create the wordclouds using data frpom a column of a dataframe
def creatWordCloud(df,clm_name):
    text = " ".join(line for line in df[clm_name])
    # Create the wordcloud object
    wordcloud = WordCloud(width=980, height=580, margin=0,collocations = False, background_color = 'white').generate(text)
    # Display the generated image:
    plt.figure(figsize=(12,5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.margins(x=0, y=0)
    plt.show()
    return plt


#creating a function to show the result of the sentiment analysis from the final df
# Creating a function to show the result of the sentiment analysis from the final df
def showReport(df):
    print(f'* The tweets show that the sentiment around is mainly {df.groupby(by="sentiment").id.count().sort_values(ascending=False).index[0] }')
    print(f'* This is how the overall sentiment and stars ratings breakdown on the {len(df)} total records we recovered : ')
    print(df.groupby(["stars"]).count()['id'])

    # Build the percentage of star count reviews by category pie chart.
    # Get unique star ratings from the DataFrame
    unique_star_ratings = sorted(df['stars'].unique())

    # Generate labels based on available star ratings
    labels = [f"{rating} star" for rating in unique_star_ratings]

    # Count the number of occurrences of each star rating
    star_counts = df['stars'].value_counts().sort_index()

    # Calculate the percentage of star count reviews by category
    star_perc = 100 * star_counts / len(df)
    # Define colors and explode (if needed)
    colors = ["red", "orange", "gold", 'turquoise', 'green'][:len(labels)]
    explode = [0.05] * len(labels)  # You can adjust the explode values as needed

    # Create the pie chart
    plt.pie(star_perc, labels=labels, colors=colors, explode=explode, autopct='%1.1f%%', shadow=True, startangle=150)
    plt.title("Percentage of Total tweets by star ratings")
    
    # Save the star_pie_chart.png
    plt.savefig("star_pie_chart.png")
    plt.close()  # Close the current figure

    print("_________________________________________________________________________________________________")
    
    # Build the sentiment reviews by category pie chart.
    sent_perc = 100 * df.groupby(["sentiment"]).count()['id'] / len(df)
    plt.pie(sent_perc,
            labels=["Negative", "Neutral", "Positive"],
            colors=["red", "gold", 'green'],
            explode=[0.05, 0.05, 0.05],
            autopct='%1.1f%%',
            shadow=True, startangle=150)
    plt.title("Percentage of total tweets by sentiment ")
    
    # Save the sentiment_pie_chart.png
    plt.savefig("sentiment_pie_chart.png")
    plt.close()  # Close the current figure

   
    

#function to creat word clouds for each tweet sentiment
def sentimentWordcloud(df):
    print("We generate Wordclouds for each sentiment to see the words that appear most often for each one :")
    print("_________________________________________________________________________________________________")
    print('Wordcloud for negative sentiment tweets : ')
    wordcloud_neg =creatWordCloud(df.query('sentiment == "negative"'),"clean_tweet")
    wordcloud_neg.savefig("wordcloud_negative.png")
    print('Wordcloud for neutral sentiment tweets : ')
    wordcloud_neu = creatWordCloud(df.query('sentiment == "neutral"'),"clean_tweet")
    wordcloud_neu.savefig("wordcloud_neutral.png")
    print('Wordcloud for positive sentiment tweets : ')
    wordcloud_pos = creatWordCloud(df.query('sentiment == "positive"'),"clean_tweet")
    wordcloud_pos.savefig("wordcloud_positive.png")

def calculate_sentiment_percentage(sentiment_column):
    sentiment_mapping = {
        'positive': 3,
        'neutral': 2,
        'negative': 1
    }

    sentiment_column_numeric = sentiment_column.map(sentiment_mapping)
    sentiment_sum = sentiment_column_numeric.sum()
    sentiment_percentage = sentiment_sum / 300

    return sentiment_percentage

#normalize with min max scale
def normalize(metric, min_value, max_value):
    return (metric - min_value) / (max_value - min_value)

def calculate_networking_score(metrics):
    # Define weights for each metric (you can adjust these)
    weights = {
        'followers_count': 0.5,
        'following_count': 0.3,
        'tweet_count': 0.2
    }

    # Normalize the metrics
    min_followers = 0  # Minimum followers_count
    max_followers = 1000000  # Maximum followers_count 
    normalized_metrics = {metric: normalize(value, min_followers, max_followers) for metric, value in metrics.items()}

    # Calculate the networking score by iterating over normalized_metrics
    networking_score = sum(weights[metric] * normalized_metrics[metric] for metric in normalized_metrics)

    return networking_score


def get_user_info(username):
    
    try:
        # Get user data
        users = client.get_users(usernames=[username], user_fields='public_metrics')
        
        # Check if user data is available
        if not users.data:
            raise Exception("User not found.")

        print(f'Generating networking score for {username} ...')
        for user in users.data:
            followers_count = user.public_metrics['followers_count']
            following_count = user.public_metrics['following_count']
            tweet_count = user.public_metrics['tweet_count']

            metrics = {
                'followers_count': followers_count,
                'following_count': following_count,
                'tweet_count': tweet_count
            }

            networking_score = calculate_networking_score(metrics)
            print(f"Networking Score for {username}: {networking_score}")

        return networking_score

    except tweepy.TweepError as e:
        raise Exception(f"Error: {str(e)}")


# creating call functions
def get_tweet(username):
    print ("This is taking the username provided and trying to get the user id and user information to generate the tweets from the username")
    print ('-------------------------------------------------')
    tweets = get_user_data(username)
    print ('retrieving tweets -- ')
    return tweets

def tweetTodf(tweets):
    print ("This is running the data into DataFrame from the Tweepy api result")
    print ('-------------------------------------------------')
    df = tweetsETL(tweets)
    print ('retrieving tweets -- ')
    return df

def sentiment(df):
    print('-----------------------------------------')
    print('sentiment analysis in progress.')
    print('this might take a minute ...')
    final_df = sentimentToDf(df,sentimentAnalysis(df))
    print('-----------------------------------------')
    print(final_df.head())
    print('-----------------------------------------')
    return final_df

def sentiment_score(final_df):
    print('-----------------------------------------')
    print('sentiment scoring in progress.')
    print('this might take a minute ...')
    result = calculate_sentiment_percentage(final_df['sentiment'])
    print('-----------------------------------------')
    print(f'The sentiment score of this user is --: {result}' )
    print('-----------------------------------------')
    return result

def finalReport(final_df):
    print('creating report ...')
    print(f'the report represents the sentiment around tweeter, stars represent the sentiment of a tweet \n from 1 being most negative to 5 being most positive ...')
    print("_________________________________________________________________________________________________\n")
    showReport(final_df)
    print("_________________________________________________________________________________________________\n")
    print("_________________________________________________________________________________________________\n")
    sentimentWordcloud(final_df)
    return final_df.sample(10)

def generate_networking_score(username):
    print("Generating the score")
    score = get_user_info(username)
    print('-----------------------------------------')
    print(f'The networking score of this user is --: {score}' )
    print('-----------------------------------------')
    return score