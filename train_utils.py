import pandas as pd
import numpy as np
import joblib
import re
from collections import defaultdict
from string import punctuation

from imblearn.over_sampling import BorderlineSMOTE
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

from tensorflow.keras.preprocessing.text import Tokenizer
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
    joblib.dump(vectorizer, FILE_DIR + 'fitted_vectorizer.pkl')
    vectorizer = joblib.load(FILE_DIR + 'fitted_vectorizer.pkl')

    # Vectorize validation and testing texts.
    x_val = vectorizer.transform(validating_texts)
    x_test = vectorizer.transform(testing_texts)

    del training_texts, validating_texts, testing_texts

    # Over and/or under sampling to check class imbalance
    # x_train, training_labels = over_under_sampling(x_train, training_labels)

    # Slect top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(top_k, x_train.shape[1]))
    selector.fit(x_train, training_labels.idxmax(axis=1))
    # selector.fit(x_train, training_labels)
    joblib.dump(selector, FILE_DIR + 'fitted_feature_selector.pkl')
    selector = joblib.load(FILE_DIR + 'fitted_feature_selector.pkl')
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
    joblib.dump(tokenizer, FILE_DIR + 'fitted_tokenizer.pkl')
    tokenizer = joblib.load(FILE_DIR + 'fitted_tokenizer.pkl')

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
    joblib.dump(max_length, FILE_DIR + 'max_length.pkl')

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
