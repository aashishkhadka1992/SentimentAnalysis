import pandas as pd
import numpy as np
import re
from collections import defaultdict
from string import punctuation

import gensim
import gensim.corpora as corpora
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


class TopicModeler:
    """TopicModeler class facilitates topic modeling for each article.
    """
    def __init__(self, news_df):
        """
        Create a new instance of TopicModeler class with news dataframe argument
        :param news_df:
        """
        self.news_df = news_df
        text_cleaner = TextCleaner(self.news_df)
        self.lemmatized_words = text_cleaner.lemmatization()
        self.id2word = corpora.Dictionary(self.lemmatized_words)
        self.all_corpus = [self.id2word.doc2bow(text) for text in self.lemmatized_words]

    def find_optimum_model(self):
        """
        This function implements LDA model to extract optimum topic model for the corpus created using the list
        of words returned by lemmatization method from TextCleaner class
        """

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

        models, coh_values, topics_count = compute_coherence_values(dictionary=self.id2word,
                                                                    all_corpus=self.all_corpus,
                                                                    texts=self.lemmatized_words,
                                                                    start=3, limit=11, step=1)
        model_values_df = pd.DataFrame({'model_list': models, 'coherence_values': coh_values,
                                        'num_topics': topics_count})
        # optimal_num_topics = model_values_df.loc[model_values_df['coherence_values'].idxmax()]['num_topics']
        optimal_model = model_values_df.loc[model_values_df['coherence_values'].idxmax()]['model_list']

        return optimal_model

    def generate_dominant_topic(self):
        optimal_model = self.find_optimum_model()
        new_corpus = self.all_corpus

        def format_topics_sentences(ldamodel, corpus):
            sent_topics_df = pd.DataFrame()
            for i, row in enumerate(ldamodel[corpus]):
                for j, (topic_num, prop_topic) in enumerate(row):
                    if j == 0:
                        wp = ldamodel.show_topic(topic_num)
                        topic_keywords = ", ".join([word for word, prop in wp])
                        sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4),
                                                                          topic_keywords]), ignore_index=True)
                    else:
                        break
            sent_topics_df = pd.concat([sent_topics_df, pd.Series(self.lemmatized_words)], axis=1)
            sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords', 'Clean_Text']
            return sent_topics_df

        df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=new_corpus)
        df_topic_sents_keywords = pd.concat([df_topic_sents_keywords, self.news_df[['Published', 'Source', 'Text',
                                                                                    'Keywords', 'Url']]], axis=1)
        df_dominant_topic = df_topic_sents_keywords.reset_index()
        df_dominant_topic.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contribution', 'Topic_Keywords',
                                     'Clean_Text', 'Published', 'Source', 'Text', 'Keywords', 'Url']
        df_dominant_topic.dropna(inplace=True)

        return df_dominant_topic, optimal_model, new_corpus
