import pandas as pd
import numpy as np
import joblib
import re
# import random
import json
from datetime import datetime, timedelta
import dateutil.parser
import logging
from collections import defaultdict
from string import punctuation
import urllib
from newsapi.newsapi_client import NewsApiClient
from newspaper import Article
from TwitterAPI import TwitterAPI, TwitterConnectionError, TwitterRequestError

import NewsAPI_Credentials as nc
import Twitter_Credentials as tc


# import json
# import asyncio
import nest_asyncio
# import aiohttp
# import async_timeout

from imblearn.over_sampling import BorderlineSMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel
from gensim.models.phrases import Phrases, Phraser

import nltk
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('average_perceptron_tagger')
from nltk.corpus import stopwords
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag

import warnings
import os

import multiprocessing
cores = multiprocessing.cpu_count()

nest_asyncio.apply()

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

FILE_DIR = '/tmp/'
if not os.path.exists(FILE_DIR):
    os.mkdir(FILE_DIR)


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

    def authenticate_twitter_app(self):
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
                        Text = status['retweeted_status']['extended_tweet']['full_text']
                    else:
                        Text = status['retweeted_status']['text']
                elif 'extended_tweet' in status.keys():
                    Text = status['extended_tweet']['full_text']
                else:
                    Text = status['text']
                if not len(Text.split()) >= 3:
                    tweet['Published'] = dateutil.parser.parse(status['created_at']).strftime('%Y/%m/%d')
                    tweet['Text'] = Text
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
        This function cleans, removes stopwords, creates bigrams, trigrams, filter words for allowed POS tags, and
        lemmatizes tweets extracted by get_tweets method
        :return: dataframe with processed tweets
        """
        tweet_df = self.get_tweets()

        def sent_to_words():
            """
            This generator performs several cleaning actions to remove unwanted information such as number, hyperlinks,
            special characters.
            :return: clean tweets
            """
            data = tweet_df.Text.values.tolist()
            # Remove HTML special entities(e.g. &amp;)
            data = [re.sub(r'\&\w*;', '', sent) for sent in data]
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
                                   "oughtn't've": "ought not have", "shan't": "shall not", "shan't've": "shall not have",
                                   "shouldn't": "should not", "shouldn't've": "should not have", "wasn't": "was not",
                                   "weren't": "were not", "won't": "will not", "won't've": "will not have",
                                   "wouldn't": "would not", "wouldn't've": "would not have"}
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
            # Remove special characters
            data = [re.sub(r'[^a-zA-A0-9]+', '', sent) for sent in data]
            for sentence in data:
                yield simple_preprocess(str(sentence), deacc=True)

        def remove_stopwords():
            """
            This function removes stopwords from clean tweets yielded by clean_tweets generator.
            :return: list of tweets with clean tweets excluding stopwords
            """
            stop_words = stopwords.words('english') + list(punctuation)
            stop_words = stop_words + ['from', 'say', 'subject', 're', 'edu', 'use', 'rt']
            return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in
                    sent_to_words()]

        def make_trigrams():
            bigram = Phrases(sent_to_words(), min_count=5, threshold=100)
            bigram_mod = Phraser(bigram)
            bigram_data_words = [bigram_mod[doc] for doc in remove_stopwords()]

            trigram = Phrases(bigram[sent_to_words()], threshold=100)
            trigram_mod = Phraser(trigram)
            return [trigram_mod[bigram_mod[doc]] for doc in bigram_data_words]

        def lemmatization():
            texts_out = []
            allowed_postags = ['J', 'v', 'R', 'N']
            tag_map = defaultdict(lambda: wn.NOUN)
            tag_map['J'] = wn.ADJ
            tag_map['V'] = wn.VERB
            tag_map['R'] = wn.ADV

            lmtzr = WordNetLemmatizer()

            for sent in make_trigrams():
                texts_out.append([lmtzr.lemmatize(token, tag_map[tag[0]]) for token, tag in pos_tag(sent)
                                  if tag[0] in allowed_postags])
            return texts_out

        lemmatized_words = lemmatization()
        clean_tweets_df = pd.concat([tweet_df, pd.Series(lemmatized_words)], axis=1)
        clean_tweets_df.columns = ['Published', 'Text', 'Clean_Text']

        return clean_tweets_df[['Clean_Text', 'Published', 'Text']]


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
        results = []
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
        """This function faciliates scrapping news articles from urls using urllib and return pandas dataframe.
        """
        results = []

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
            except:
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


class TextCleaner:
    def __init__(self, df):
        self.df = df

    def sent_to_words(self):
        data = self.df.Text.values.tolist()
        data = [re.sub(r'\w*\$;', '', sent) for sent in data]
        data = [re.sub(r'(http|htpps)?:[A-Za-z0-9.\-r\n]+', '', sent) for sent in data]
        data = [re.sub(r'www.[A-Za-z0-9.]+', '', sent) for sent in data]
        data = [re.sub(r'[0-9]+', '', sent) for sent in data]
        data = [re.sub(r'(\d{1,2})[/.-](\d{1,2})[/.-](\d{2,4})+', '', sent) for sent in data]
        data = [re.sub(r'(\d{1,2})[/:](\d{2})[/:](\d{2})?(am|pm)+', '', sent) for sent in data]
        data = [re.sub(r'\s\s+', ' ', sent) for sent in data]
        data = [re.sub(r'\S*@\S*\s?', '', sent) for sent in data]
        data = [re.sub(r'\s+', ' ', sent) for sent in data]
        data = [re.sub(r"\'", '', sent) for sent in data]
        for sentence in data:
            yield simple_preprocess(str(sentence), deacc=True)

    def remove_stopwords(self):
        stop_words = stopwords.words('english') + list(punctuation)
        stop_words = stop_words + ['from', 'say', 'subject', 're', 'edu', 'use', 'rt']
        return [[word for word in simple_preprocess(str(text)) if word not in stop_words] for text in
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


class TopicModeler:
    """TopicModeler class facilitates topic modeling for each article.
    """
    def __init__(self, news_df):
        """
        Create a new instance of TopicModeler class with news dataframe argument
        :param news_df:
        """
        self.news_df = news_df

    def find_optimum_model(self):
        """
        This function implements LDA model to extract optimum topic model for the corpus created using the list
        of words returned by lemmatization method from TextCleaner class
        """
        global id2word, all_corpus, optimal_model
        text_cleaner = TextCleaner(self.news_df)
        lemmatized_words = text_cleaner.lemmatization()
        id2word = corpora.Dictionary(lemmatized_words)
        all_corpus = [id2word.doc2bow(text) for text in lemmatized_words]

        def compute_coherence_values(dictionary, all_corpus, texts, limit, start=2, step=4):
            coherence_values = []
            model_list = []
            num_topics_list = []

            for num_topics in range(start, limit, step):
                model = gensim.models.ldamulticore.LdaMulticore(corpus=all_corpus, num_topics=num_topics,
                                                                id2word=dictionary, eval_every=1, passes=10,
                                                                iterations=50, workers=cores, random_state=42)
                model_list.append(model)
                coherence_model = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
                coherence_values.append(coherence_model.get_coherence())
                num_topics_list.append(num_topics)

            return model_list, coherence_values, num_topics_list

        model_list, coherence_values, num_topics_list = compute_coherence_values(dictionary=id2word,
                                                                                 all_corpus=all_corpus,
                                                                                 texts=lemmatized_words,
                                                                                 start=3, limit=11, step=1)
        model_values_df = pd.DataFrame({'model_list': model_list, 'coherence_values': coherence_values,
                                        'num_topics': num_topics_list})
        optimal_num_topics = model_values_df.loc[model_values_df['coherence_values'].idxmax()]['num_topics']
        optimal_model = model_values_df.loc[model_values_df['coherence_values'].idxmax()]['model_list']

        return lemmatized_words, id2word, optimal_model

    def generate_dominant_topic(self):
        lemmatized_words, id2word, optimal_model = self.find_optimum_model()
        new_corpus = all_corpus

        def format_topics_sentences(ldamodel, new_corpus):
            sent_topics_df = pd.DataFrame()
            for i, row in enumerate(ldamodel[new_corpus]):
                for j, (topic_num, prop_topic) in enumerate(row):
                    if j == 0:
                        wp = ldamodel.show_topic(topic_num)
                        topic_keywords = ", ".join([word for word, prop in wp])
                        sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4),
                                                                          topic_keywords]), ignore_index=True)
                    else:
                        break
                sent_topics_df = pd.concat([sent_topics_df, pd.Series(lemmatized_words)], axis=1)
                sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Clean_Text']
                return sent_topics_df

        df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, new_corpus=new_corpus)
        df_topic_sents_keywords = pd.concat([df_topic_sents_keywords, self.news_df[['Published', 'Source', 'Text',
                                                                                    'Keywords', 'Url']]], axis=1)
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contribution', 'Topic_Keywords',
                                     'Clean_Text', 'Published', 'Source', 'Text', 'Keywords', 'Url']
        df_dominant_topic.dropna(inplace=True)

        return df_dominant_topic, optimal_model, new_corpus


def over_under_sampling(x, y):
    print('Generating synthetic samples...')
    over = BorderlineSMOTE()
    # under = RandomUnderSampler(sampling_strategy=0.5)
    # steps = [('o', over), ('u', under)]
    # pipeline = Pipeline(steps=steps)
    # x, y = pipeline.fit_resample(x, y)
    x, y = over.fit_resample(x, y.idxmax(axis=1))
    y = pd.get_dummies(y)
    return x, y


def ngram_vectorize(training_texts, training_labels, validating_texts, testing_texts):
    """Vectorizes texts and n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    :param training_texts: list, training text strings.
    :param training_labels: np.ndarray, training labels.
    :param validating_texts: list, validation text strings.
    :param testing_texts: list, testing text strings.
    :return: x_train, x_val, x_test: vectorized training, validation, and testing texts
    """
    # Vectorization parameters
    # Range (inclusive) of n-gram sizes for tokenizing text.
    ngram_range = (1, 2)

    # Limit on the number of features. We use the top 20K features.
    top_k = 20000

    # Whether text should be split into words or character n-grams.
    # One of 'word', 'char'.
    token_mode = 'word'

    # Minimum document/corpus frequency below which a token will be discarded.
    min_document_frequency = 2

    # Create keyword arguments to pass to the tf-idf vectorizer.
    kwargs = {'ngram_range': ngram_range,
              'dtype': 'float32',
              'strip_accents': 'unicode',
              'decode_error': 'replace',
              'analyzer': token_mode,
              'min_df': min_document_frequency}
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(training_texts)
    joblib.dump(vectorizer, 'artifacts/fitted_vectorizer.pkl')
    vectorizer = joblib.load('artifacts/fitted_vectorizer.pkl')

    # Vectorize validation and testing texts.
    x_val = vectorizer.transform(validating_texts)
    x_test = vectorizer.transform(testing_texts)

    del training_texts, validating_texts, testing_texts

    # Over and/or under sampling to check class imbalance
    x_train, training_labels = over_under_sampling(x_train, training_labels)

    # Slect top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(top_k, x_train.shape[1]))
    selector.fit(x_train, training_labels.idxmax(axis=1))
    # selector.fit(x_train, training_labels)
    joblib.dump(selector, 'artifacts/fitted_feature_selector.pkl')
    selector = joblib.load('artifacts/fitted_feature_selector.pkl')
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    x_test = selector.transform(x_test).astype('float32')

    del vectorizer, selector

    return x_train, training_labels, x_val, x_test


def sequence_vectorize(training_texts, validating_texts, testing_texts, max_sequence_length):
    """

    :param training_texts: list, training text strings.
    :param validating_texts: list, validation text strings.
    :param testing_texts: list, testing text strings.
    :param max_sequence_length: desired length of the sequence.
    :return: x_train, x_val, x_test: sequence vectors for training, validating and testing texts.
    :return: tokenizer.word_index: word_index of the fitted tokenizer.
    """
    # Limit on the number of features. We use the top 20K features.
    top_k = 20000

    # Create vocabulary with training texts.
    tokenizer = Tokenizer(num_words=top_k)
    tokenizer.fit_on_texts(training_texts)
    joblib.dump(tokenizer, 'artifacts/fitted_tokenizer.pkl')
    tokenizer = joblib.load('artifacts/fitted_tokenizer.pkl')

    # Vectorize training, validating, and testing texts.
    x_train_seq = tokenizer.texts_to_sequences(training_texts)
    x_val_seq = tokenizer.texts_to_sequences(validating_texts)
    x_test_seq = tokenizer.texts_to_sequences(testing_texts)

    del training_texts, validating_texts, testing_texts

    # Get max sequence length
    max_length = len(max(x_train_seq, key=len))
    if max_length > max_sequence_length:
        max_length = max_sequence_length
    # Save max length to use during prediction.
    joblib.dump(max_length, 'artifacts/max_length.pkl')

    # Fix sequence length to max value. Sequences shorted than the length are
    # padded in the beginning and sequences longer are truncated at the beginning.
    x_train = pad_sequences(x_train_seq, maxlen=max_length).astype('float32')
    x_val = pad_sequences(x_val_seq, maxlen=max_length).astype('float32')
    x_test = pad_sequences(x_test_seq, maxlen=max_length).astype('float32')

    del x_train_seq, x_val_seq, x_test_seq

    return x_train, x_val, x_test, tokenizer.word_index


def generate_embedding_matrix(training_texts, validating_texts, testing_texts):
    """

    :param training_texts: list, training text strings.
    :param validating_texts: list, validation text strings.
    :param testing_texts: list, testing text strings.
    :return: train_vector, val_vector, test_vector: training, validating and testing embedded matrices using Doc2Vec
    """

    def tokenize_tag(texts, tokens_only=False):
        for i, doc in enumerate(texts):
            tokens = gensim.utils.simple_preprocess(doc)
            if tokens_only:
                yield tokens
            else:
                yield gensim.models.doc2vec.TaggedDocument(tokens, [i])

    train_corpus = list(tokenize_tag(texts=training_texts))
    val_corpus = list(tokenize_tag(validating_texts, tokens_only=True))
    test_corpus = list(tokenize_tag(testing_texts, tokens_only=True))

    model = gensim.models.doc2vec.Doc2Vec(vector_size=200, min_count=2, dm=1, epochs=40, workers=cores)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    train_vector = []
    for text in train_corpus:
        inferred_vec = model.infer_vector(text.words)
        train_vector.append(inferred_vec)

    val_vector = []
    for text in val_corpus:
        inferred_vec = model.infer_vector(text)
        val_vector.append(inferred_vec)

    test_vector = []
    for text in test_corpus:
        inferred_vec = model.infer_vector(text)
        test_vector.append(inferred_vec)

    train_vector = np.array(train_vector)
    val_vector = np.array(val_vector)
    test_vector = np.array(test_vector)
    train_vector = train_vector.reshape((train_vector.shape[0], train_vector.shape[1], 1))
    val_vector = val_vector.reshape((val_vector.shape[0], val_vector.shape[1], 1))
    test_vector = test_vector.reshape((test_vector.shape[0], test_vector.shape[1], 1))

    del train_corpus, val_corpus, test_corpus

    return train_vector, val_vector, test_vector


def convert_dataframe_to_text(train_texts, val_texts, test_texts, train_labels, val_labels, test_labels):
    train_df = pd.concat([train_labels, train_texts], axis=1)
    train_df['Sentiment'] = train_df.Sentiment.apply(lambda x: '__label__' + str(x))
    val_df = pd.concat([val_labels, val_texts], axis=1)
    val_df['Sentiment'] = val_df.Sentiment.apply(lambda x: '__label__' + str(x))
    test_df = pd.concat([test_labels, test_texts], axis=1)
    test_df['Sentiment'] = test_df.Sentiment.apply(lambda x: '__label__' + str(x))
    
    train_df.to_csv(FILE_DIR + 'train.txt', header=False, index=False, sep='\t')
    val_df.to_csv(FILE_DIR + 'val.txt', header=False, index=False, sep='\t')
    test_df.to_csv(FILE_DIR + 'test.txt', header=False, index=False, sep='\t')


class SentimentAnalyzer:
    """
    The SentimentAnalyzer class facilitates converting article text into numerical vectors, predicting sentiment
    using pre-trained model and creating a dataframe that contains text, predicted sentiment, prediction confidence.
    """

    def __init__(self, artice_df, column):
        """Create a new instance of SentimentAnalyzer with dataframe and columns to be used for prediction as
        arguments.
        :param artice_df: dataframe with relevant articles
        :param column: string column name to use for prediction
        """
        self.article_df = artice_df
        self.column = column

    def vectorize_ngrams(self, articles):
        """
        This function performs vectorization and feature selection on column of text using pre-trained vectorizer and
        feature selector.
        :param articles: series of clean article texts
        :return: series of vectorized and feature selected article texts
        """
        tagged_articles = articles
        vectorizer = joblib.load(FILE_DIR + 'fitted_vectorizer.pkl')
        vectorized_article = vectorizer.transform(tagged_articles)
        selector = joblib.load(FILE_DIR + 'fiited_feature_selector.pkl')
        feature_selected_article = selector.transform(vectorized_article).astype('float32')
        del tagged_articles, vectorizer, vectorized_article, selector
        return feature_selected_article

    def vectorize_sequence(self, articles):
        """
        This fucntion tokenizes column of text, converts tokens into sequence and pads them using pre-trained
        tokenizer and max sequence length.
        :param articles: series of clean article texts
        :return: series of tokenized and padded sequence of article texts
        """
        tagged_articles = articles
        tokenizer = joblib.load(FILE_DIR + 'fitted_tokenizer.pkl')
        tokenized_article = tokenizer.texts_to_sequences(tagged_articles)
        max_length = joblib.load(FILE_DIR + 'max_length.pkl')
        padded_articles = pad_sequences(tokenized_article, maxlen=max_length)
        del tagged_articles, tokenizer, tokenized_article, max_length
        return padded_articles

    def generate_embeddings(self, articles):
        """
        This function generates embeddings for articles using pre-trained doc2vec model.
        :param: articles:series of clean article texts
        :return: series of embeddings of article texts
        """
        tagged_articles = articles
        tokenized_articles = []
        for doc in tagged_articles:
            tokens = gensim.utils.simple_preprocess(doc)
            tokenized_articles.append(tokens)
        del tagged_articles

        model = gensim.models.doc2vec.Doc2Vec.load(FILE_DIR + 'trained_doc2vec_model')

        vectorized_articles = []
        for doc in tokenized_articles:
            inferred_vec = model.infer_vector(doc)
            vectorized_articles.append(inferred_vec)
        del inferred_vec, model, tokenized_articles
        vectorized_articles = np.array(vectorized_articles)
        vectorized_articles = vectorized_articles.reshape(vectorized_articles.shape[0], vectorized_articles.shape[1], 1)
        return vectorized_articles

    def analyze_sentiment(self, articles):
        """
        This function uses transformed articles returned by vectorize_ngrams or vectorize_sequence or generate_embeddings
        method and passes them to pre-trained model to predict sentiment and confidence levels for each article.
        :param articles: series of clean article texts
        :return: dataframe with predicted sentiment labels and confidence metircs for each article
        """
        best_params = joblib.load(FILE_DIR + 'best_params.pkl')
        best_model = load_model(FILE_DIR + 'best_model.h5')
        class_weights = joblib.load(FILE_DIR + 'class_weights.pkl')

        if best_params['model_type'] == 'MLP':
            pred_proba = best_model.predict(self.vectorize_ngrams(articles).toarray())
        elif best_params['model_type'] == 'CNN_BiLSTM':
            pred_proba = best_model.predict(self.generate_embeddings(articles))
        else:
            pred_proba = best_model.predict(self.vectorize_sequence(articles))

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

    def news_to_datafrmae(self):
        """
        This function combined instance dataframe argument with predictions dataframe returned by analyze sentiment
        method and provides appropriate column labels.
        :return: dataframe prediction and original information
        """
        df = self.article_df
        predictions_df = self.analyze_sentiment(df[self.column])
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
    return word_list
