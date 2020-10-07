import pandas as pd
import numpy as np
import joblib
import re
from datetime import datetime, timedelta
import dateutil.parser
from collections import defaultdict
from string import punctuation
import urllib
from newsapi.newsapi_client import NewsApiClient
from newspaper import Article
from TwitterAPI import TwitterAPI, TwitterConnectionError, TwitterRequestError

import NewsAPI_Credentials as nc
import Twitter_Credentials as tc

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import gensim
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser

from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

import warnings
import os

import multiprocessing
cores = multiprocessing.cpu_count()

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FILE_DIR = './artifacts/'
if not os.path.exists(FILE_DIR):
    os.mkdir(FILE_DIR)


class TextCleaner:
    def __init__(self, df):
        self.df = df

    def sent_to_words(self):
        """
        This generator performs several cleaning actions to remove unwanted information such as number, hyperlinks,
        special characters.
        :return: clean texts
        """
        data = self.df.Text.values.tolist()
        # Remove HTML special entities(e.g. &amp;)
        data = [re.sub(r'&\w*;', '', sent) for sent in data]
        # Convert @username to AT_USER
        data = [re.sub(r'@[A-Za-z0-9_]+', '', sent) for sent in data]
        # Remove tickers
        data = [re.sub(r'\$\w*', '', sent) for sent in data]
        # Remove hyperlinks
        data = [re.sub(r'(http|htpps)?:[A-Za-z0-9.\-r\n]+', '', sent) for sent in data]
        data = [re.sub(r'www.[A-Za-z0-9.]+', '', sent) for sent in data]
        # Remove hashtags
        data = [re.sub(r'#\w*', '', sent) for sent in data]
        # Handle contraction
        contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
                               "couldn't": "could not", "couldn't've": "could not have", "didn't": "did not",
                               "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                               "hadn't've": "had not have", "hasn't": "has not", "haven't": "have not",
                               "isn't": "is not", "mayn't": "may not", "mightn't": "might not",
                               "mightn't've": "might not have", "mustn't": "must not", "musn't've": "must not have",
                               "needn't": "need not", "needn't've": "need not have", "oughtn't": "ought not",
                               "oughtn't've": "ought not have", "shan't": "shall not",
                               "shan't've": "shall not have", "shouldn't": "should not",
                               "shouldn't've": "should not have", "wasn't": "was not",
                               "weren't": "were not", "won't": "will not", "won't've": "will not have",
                               "wouldn't": "would not", "wouldn't've": "would not have"}
        if np.median([len(s.split()) for s in data]) < 100:
            expanded = re.compile(r'\b(' + '|'.join(contraction_mapping.keys()) + r')\b')
            data = [expanded.sub(lambda x: contraction_mapping[x.group()], sent) for sent in data]
        # Remove word with 2 or fewer letters
        data = [re.sub(r'\b\w{1,2}\b', '', sent) for sent in data]
        # Remove numbers
        data = [re.sub(r'[0-9]+', '', sent) for sent in data]
        # Replace date
        data = [re.sub(r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})+', '', sent) for sent in data]
        # Replace time
        data = [re.sub(r'(\d{1,2})[/:](\d{2})[/:](\d{2})?(am|pm)+', '', sent) for sent in data]
        # Remove whitespace including new ine characters
        data = [re.sub(r'\s\s+', ' ', sent) for sent in data]
        # Remove single space remaining at the from of the tweet
        data = [sent.lstrip(' ') for sent in data]
        # Remove characters beyond Basic Multilingual Plane (BMP) of Unicode
        data = [''.join(c for c in sent if c <= '\uFFFF') for sent in data]
        for sentence in data:
            yield simple_preprocess(str(sentence), deacc=True)

    def remove_stopwords(self):
        stop_words = stopwords.words('english') + list(punctuation)
        stop_words = stop_words + ['from', 'say', 'subject', 're', 'edu', 'use', 'rt']
        return [[word for word in text if word not in stop_words] for text in
                self.sent_to_words()]

    def make_trigrams(self):
        bigram = Phrases(self.sent_to_words(), min_count=5, threshold=100)
        bigram_mod = Phraser(bigram)
        bigram_data_words = [bigram_mod[doc] for doc in self.remove_stopwords()]

        trigram = Phrases(bigram[self.sent_to_words()], threshold=100)
        trigram_mod = Phraser(trigram)
        return [trigram_mod[bigram_mod[doc]] for doc in bigram_data_words]

    def lemmatization(self):
        texts_out = []
        allowed_postags = ['J', 'v', 'R', 'N']
        tag_map = defaultdict(lambda: wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV

        lmtzr = WordNetLemmatizer()

        for sent in self.make_trigrams():
            texts_out.append([lmtzr.lemmatize(token, tag_map[tag[0]]) for token, tag in pos_tag(sent)
                              if tag[0] in allowed_postags])
        return texts_out

    def get_clean_text(self):
        lemmatized_words = self.lemmatization()
        clean_texts = [' '.join(word for word in sent) for sent in lemmatized_words]
        clean_df = pd.concat([self.df, pd.Series(clean_texts, name='Clean_Text')], axis=1)

        return clean_df


class TweetScrapper:
    """
    The TweetScrapper class facilitates scrapping and cleaning of tweets from twitter api.
    """
    def __init__(self, queries=None):
        """
        Create a new instance of TweetScrapper with queries and location argument.
        :param queries: list of strings to query
        """
        self.queries = queries

    @ staticmethod
    def authenticate_twitter_app():
        """
        This function authenticates access to twitter api using consumer key and coonsumer secret.
        :return: authenticated api
        """
        api = TwitterAPI(tc.consumer_key, tc.consumer_secret_key, auth_type='oAuth2')
        return api

    def get_tweets(self):
        """
        This function makes a request using earlier authenticated api, and instance argument i.e. queries to extract
        tweets.
        :return: dataframe with tweets and other metadata relevant to each tweet.
        """
        tweets = []
        api = self.authenticate_twitter_app()
        search_term = ' '.join(['(' + ' OR '.join([query for query in self.queries]) + ')',
                                'lang:en', 'place_country:US'])
        product = '30day'
        label = 'dev'
        try:
            results = api.request('tweets/search/%s/:%s' % (product, label), {'query': search_term})
            for status in results:
                tweet = {}
                if 'retweeted_status' in status.keys():
                    if 'extended_tweet' in status['retweeted_status']:
                        text = status['retweeted_status']['extended_tweet']['full_text']
                    else:
                        text = status['retweeted_status']['text']
                elif 'extended_tweet' in status.keys():
                    text = status['extended_tweet']['full_text']
                else:
                    text = status['text']
                if len(text.split()) >= 3:
                    tweet['Published'] = dateutil.parser.parse(status['created_at']).strftime('%Y/%m/%d')
                    tweet['Text'] = text
                    tweets.append(tweet)
            df = pd.DataFrame(tweets)
            df.dropna(inplace=True)
            df.drop_duplicates(subset=['Text'], inplace=True, ignore_index=True)
            return df

        except TwitterRequestError as e:
            if e.status_code < 500:
                # something needs to be fixed before re-connecting
                raise
            else:
                # temporary interruption, re-try request
                pass
        except TwitterConnectionError:
            # temporary interruption, re-try request
            pass

    def get_clean_tweet(self):
        """
        This function cleans, removes stopwords, creates bi-grams, tri-grams, filter words for allowed POS tags, and
        generates lemma of tweets extracted by get_tweets method
        :return: dataframe with processed tweets
        """
        tweet_df = self.get_tweets()
        text_cleaner = TextCleaner(tweet_df)
        clean_tweets_df = text_cleaner.get_clean_text()

        return clean_tweets_df[['Clean_Text', 'Published', 'Text']]


class ContentExtractor:
    """
    The ContentExtractor class facilitates scrapping full articles content from a list of urls.
    """

    def __init__(self, urls):
        """Create a new instance of ContentExtractor with the series of urls.
        :param urls: list of urls to scrape
        """
        self.urls = urls

    def content_to_dataframe(self):
        """This function facilitates scrapping news articles from urls using urllib and return pandas dataframe.
        """
        # async def check_url_get_content(session, url):
        def check_url_get_content(url):
            """This function takes url as argument, extracts text content and other information using Article class
            from newspaper library and returns result as a dictionary.
            """

            result = {}
            try:
                # async with session.get(url, timeout=600) as resp:
                with urllib.request.urlopen(url, timeout=600) as resp:
                    # content = await resp.read()
                    content = resp.read()
                    # if content:
                    article = Article(url)
                    article.set_html(content)
                    article.parse()
                    article.nlp()
                    text = article.text
                    keywords = article.keywords
                    status_code = resp.status

                    # else:
                    #     text = 'none'
                    #     keywords = 'none'
                    #     status_code = 'none'
            except Exception as e:
                text = 'none'
                keywords = 'none'
                status_code = 'none'

            result['Text'] = text
            result['Keywords'] = keywords
            result['status_code'] = status_code

            return result

        # async def main(self):
        def main():
            """This function calls check_url_get_content function in a loop for each url in list of urls and
            returns list of result dictionaries.
            """
            tasks = []
            urls = self.urls
            # async with aiohttp.ClientSession() as session:
            for url in urls:
                # task = asyncio.ensure_future(check_url_get_content(session, url))
                task = check_url_get_content(url)
                tasks.append(task)
                # responses = await asyncio.gather(*tasks)
            responses = tasks

            return responses

        # loop = asyncio.new_event_loop()
        # results = loop.run_until_complete(main(self))
        results = main()
        df = pd.DataFrame(results)
        # loop.close()

        return df


class ContentFilter:
    """
    The ContentFilter class facilitates filtering of news articles for text length, empty text as well as
    unsuccessful query
    """

    def __init__(self, df):
        """Create a new instance of ContentFilter with dataframe as argument.
        """
        self.df = df

    def filter_content(self):
        """
        This function filters rows for empty text, text with character length of less than 500,
        text pointing to another link as well as status code other than 200 and returns filtered dataframe.
        :return: filtered dataframe
        """
        df_filtered = self.df[(self.df.Text != '') &
                              (~self.df.Text.str.contains('The previous page is sending you to')) &
                              (self.df.Text.str.len() > int(500)) &
                              (self.df.status_code == int(200))]
        df_filtered = df_filtered.drop('status_code', axis=1)
        return df_filtered


class NewsScrapper:
    """
    The NewsScrapper class facilitates scrapping and cleaning of news articles from NewsApi
    """
    def __init__(self, queries=None):
        """
        Create a news instance of NewsScrapper with queries argument.
        :param queries: list of strings to query
        """
        self.queries = queries

    def get_news(self):
        """
        This function makes a request using authenticated api and instance argument to extract news articles.
        :return: dataframe with articles and other metadata relevant to each article
        """
        start = (datetime.today() - timedelta(days=7)).strftime('%Y-%m-%d')
        end = datetime.today().strftime('%Y-%m-%d')
        # Initialise NewsApiClient with an api key
        newsapi = NewsApiClient(api_key=nc.api_key)

        query = ' '.join(['(' + ' OR '.join([query for query in self.queries]) + ')'])

        # Query for articles using keyword
        all_articles = newsapi.get_everything(q=query,
                                              from_param=start,
                                              to=end,
                                              language='en',
                                              sort_by='relevancy',
                                              page_size=100)
        # Extract articles from returned json and store in articles variable
        articles = all_articles['articles']
        # Convert articles into dataframe
        articles_df = pd.DataFrame(articles)
        # Use only name part in the source columns
        articles_df['source'] = articles_df.source.map(lambda x: x['name'])
        # Select relevant columns for analysis
        articles_df = articles_df[['source', 'title', 'url', 'publishedAt', 'content']]
        articles_df.columns = ['Source', 'Title', 'Url', 'Published', 'Content']

        return articles_df

    def extract_filter_content(self):
        articles_df = self.get_news()
        content_extractor = ContentExtractor(articles_df['Url'])
        full_content_df = content_extractor.content_to_dataframe()
        combined_df = pd.concat([articles_df, full_content_df], axis=1)

        content_filter = ContentFilter(combined_df)
        filtered_df = content_filter.filter_content()

        return filtered_df


class SentimentAnalyzer:
    """
    The SentimentAnalyzer class facilitates converting text into numerical vectors, predicting sentiment
    using pre-trained model and creating a dataframe that contains text, predicted sentiment, prediction confidence.
    """

    def __init__(self, text_df, column):
        """Create a new instance of SentimentAnalyzer with dataframe and columns to be used for prediction as
        arguments.
        :param text_df: dataframe with relevant texts
        :param column: string column name to use for prediction
        """
        self.text_df = text_df
        self.column = column

    def vectorize_ngrams(self):
        """
        This function performs vectorization and feature selection on column of text using pre-trained vectorizer and
        feature selector.
        :return: series of vectorized and feature selected texts
        """
        tagged_texts = self.text_df[self.column]
        vectorizer = joblib.load(FILE_DIR + 'fitted_vectorizer.pkl')
        vectorized_texts = vectorizer.transform(tagged_texts)
        selector = joblib.load(FILE_DIR + 'fitted_feature_selector.pkl')
        feature_selected_texts = selector.transform(vectorized_texts).astype('float32')
        del tagged_texts, vectorizer, vectorized_texts, selector
        return feature_selected_texts

    def vectorize_sequence(self):
        """
        This fucntion tokenizes column of text, converts tokens into sequence and pads them using pre-trained
        tokenizer and max sequence length.
        :return: series of tokenized and padded sequence of texts
        """
        tagged_texts = self.text_df[self.column]
        tokenizer = joblib.load(FILE_DIR + 'fitted_tokenizer.pkl')
        tokenized_texts = tokenizer.texts_to_sequences(tagged_texts)
        max_length = joblib.load(FILE_DIR + 'max_length.pkl')
        padded_texts = pad_sequences(tokenized_texts, maxlen=max_length)
        del tagged_texts, tokenizer, tokenized_texts, max_length
        return padded_texts

    def generate_embeddings(self):
        """
        This function generates embeddings for articles using pre-trained doc2vec model.
        :return: series of embeddings of article texts
        """
        tagged_texts = self.text_df[self.column]
        tokenized_texts = []
        for doc in tagged_texts:
            tokens = gensim.utils.simple_preprocess(doc)
            tokenized_texts.append(tokens)
        del tagged_texts

        model = gensim.models.doc2vec.Doc2Vec.load(FILE_DIR + 'trained_doc2vec_model')

        vectorized_texts = []
        for doc in tokenized_texts:
            inferred_vec = model.infer_vector(doc)
            vectorized_texts.append(inferred_vec)
        del inferred_vec, model, tokenized_texts
        vectorized_texts = np.array(vectorized_texts)
        vectorized_texts = vectorized_texts.reshape(vectorized_texts.shape[0], vectorized_texts.shape[1], 1)
        return vectorized_texts

    def analyze_sentiment(self):
        """
        This function uses transformed articles returned by vectorize_ngrams or vectorize_sequence or
        generate_embeddings method and passes them to pre-trained model to predict sentiment and confidence levels
        for each text.
        :return: dataframe with predicted sentiment labels and confidence metircs for each text
        """
        best_params = joblib.load(FILE_DIR + 'best_params.pkl')
        best_model = load_model(FILE_DIR + 'best_model.h5')
        class_weights = joblib.load(FILE_DIR + 'class_weights.pkl')

        if best_params['model_type'] == 'MLP':
            pred_proba = best_model.predict(self.vectorize_ngrams().toarray())
        elif best_params['model_type'] == 'CNN_BiLSTM':
            pred_proba = best_model.predict(self.generate_embeddings())
        else:
            pred_proba = best_model.predict(self.vectorize_sequence())

        if pred_proba.shape[1] == 3:
            conf_df = pd.DataFrame(pred_proba, columns=['negative_conf', 'positive_conf', 'neutral_conf'])
            predictions = pred_proba.argmax(axis=-1)
            predictions = np.where(predictions == 0, -1, predictions)
            predictions = np.where(predictions == 2, 0, predictions)
        else:
            conf_df = pd.DataFrame(pred_proba, columns=['positive_conf'])
            conf_df['negative_conf'] = conf_df['postive_conf'].apply(lambda x: 1-x)
            predictions = np.array([0 if proba[0] < 0.5 else 1 for proba in pred_proba])
            predictions = np.where(predictions == 0, -1, predictions)

        pred_df = pd.DataFrame(predictions, columns=['predicted_sentiment'])
        predictions_df = pd.concat([pred_df, conf_df], axis=1)
        if pred_proba.shape[1] == 3:
            neutral_weight = class_weights[2] / sum(class_weights.values())
            positive_weight = class_weights[1] / sum(class_weights.values())
            negative_weight = class_weights[0] / sum(class_weights.values())
            predictions_df['weighted_sentiment'] = predictions_df['neutral_conf'] * neutral_weight + \
                predictions_df['positive_conf'] * positive_weight + \
                predictions_df['negative_conf'] * negative_weight
        else:
            positive_weight = class_weights[1] / sum(class_weights.values())
            negative_weight = class_weights[0] / sum(class_weights.values())
            predictions_df['weighted_sentiment'] = predictions_df['positive_conf'] * positive_weight + \
                predictions_df['negative_conf'] * negative_weight

        del best_params, best_model, pred_proba, predictions, conf_df, pred_df

        return predictions_df

    def sentiment_to_dataframe(self):
        """
        This function combined instance dataframe argument with predictions dataframe returned by analyze sentiment
        method and provides appropriate column labels.
        :return: dataframe prediction and original information
        """
        df = self.text_df
        predictions_df = self.analyze_sentiment()
        df['Sentiment'] = predictions_df['predicted_sentiment']
        df['Positive_Confidence'] = predictions_df['positive_conf']
        if 'neutral_conf' in predictions_df.columns:
            df['Neutral_Confidence'] = predictions_df['neutral_conf']
        df['Negative_Confidence'] = predictions_df['negative_conf']
        df['Weighted_Confidence'] = predictions_df['weighted_sentiment']
        df.drop_duplicates(subset='Text', keep='last', inplace=True)
        df.reset_index(inplace=True, drop=True)

        return df


def get_clean_text(string_list):
    word_list = string_list.strip("][").split(', ')
    word_list = [word.strip("''") for word in word_list]
    clean_text = ' '.join(word_list)
    return clean_text
