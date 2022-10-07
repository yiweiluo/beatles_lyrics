import re
from termcolor import colored
from IPython.display import HTML as html_print
import pandas as pd
import numpy as np
import pickle
from collections import Counter

def strip_punc(s):
    return re.sub(r"[^\w\d'\s-]+", '', s).strip()

def strip_punc_title(s):
    return re.sub(r"[^\w\d\s-]+", '', s).strip()

# print(strip_punc("No-[o]ne's #fright$ened o,f playing it."))
# print(strip_punc_title("No-[o]ne's #fright$ened o,f playing it."))

def dedupe_lyrics(lines, verbose=False):
    lyrics = strip_punc_title(lines).lower()
    
    lines = [strip_punc(l) for l in lyrics.split('\n')
             if len(strip_punc(l)) > 0]
    unique_lines = []
    for line in lines:
        if line not in unique_lines:
            tokens = line.split()
            unique_tokens = []
            token_colors = []
            for ix_tok, tok in enumerate(tokens):
                if (ix_tok == 0) or (tok != tokens[ix_tok-1]):
                    unique_tokens.append(tok)
                    token_colors.append("grey")
                else:
                    token_colors.append("red")
                    continue
            unique_lines.append(' '.join(unique_tokens))
            if verbose:
                print(' '.join([colored(tokens[i], token_colors[i])
                                          for i in range(len(tokens))]))
        else:
            if verbose:
                print(colored(line, 'red'))
    
    return '\n'.join(unique_lines)

import sklearn
# from sklearn.feature_extraction.text import CountVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

class TTRExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns TTR values"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2ttr[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class FirstSgPronounExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns counts of first sg. pronouns"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2first_sg[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class FirstPlPronounExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns counts of first sg. pronouns"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2first_pl[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class SecondPronounExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns counts of second person pronouns"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2second[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class ShePronounExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns counts of she/her pronouns"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2she[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class HePronounExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns counts of he/him/his pronouns"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2he[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class ItPronounExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns counts of it pronouns"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2it[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class TheyPronounExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns counts of they/them pronouns"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2they[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class ValenceExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns total valence"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2valence[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class ArousalExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns total arousal"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2arousal[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class DominanceExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns total dominance rating"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2dominance[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class NegationExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns counts of negation"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2neg[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class TotalWordsExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns the total number of words in a song"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2total_words[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class TotalLemmasExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns the total number of words in a song"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2total_lemmas[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class TotalLinesExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns the total number of lines in a song"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2total_lines[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self
    
class MeanWordsPerLineExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns the mean number of words per line in a song"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2mean_words[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

class MeanCharsPerWordExtractor(BaseEstimator, TransformerMixin):
    """Takes in feature dataframe, returns the mean number of chars per word in a song"""

    def __init__(self):
        pass

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return np.array([[text2mean_chars[text]] for text in texts])

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self

def extract_ngrams(corpus):
    vectorizer = CountVectorizer(analyzer='word', ngram_range=(1,4))
    X = vectorizer.fit_transform(corpus)
    vocab = vectorizer.get_feature_names()
    
    return {'vocab': vocab, 'array': X.toarray()}
    
def extract_TTR(song_title, verbose=False):
    doc = song_title2doc[song_title]
    all_tokens = [tok.lemma_ for tok in doc
                  if tok.pos_ not in {'SPACE'}]
    all_types = set(all_tokens)
    
    if verbose:
        print(f'Found {len(all_types)} unique lemmas:\n {all_types}.')
        print(f'\nFound {len(all_tokens)} total lemmas:\n {all_tokens}.')
    return len(all_types)/len(all_tokens)

PRONOUNS_DICT = {
    'first_sg' : {'I','me','my','mine'},
    'first_pl' : {'we','us','our','ours'},
    'second' : {'you','your','yours'},
    'third_sg_f' : {'she','her','hers'},
    'third_sg_m' : {'he','him','his'},
    'third_sg_n' : {'it','its'},
    'third_pl' : {'they','them','their','theirs'}}

def extract_pronoun_counts(song_title, pronouns_dict=PRONOUNS_DICT, verbose=False):
    doc = song_title2doc[song_title]
    all_pronouns = [tok.text for tok in doc if tok.pos_ == 'PRON']
    all_pronouns_set = set(all_pronouns)
    all_pronoun_counts = Counter(all_pronouns)
    pronouns_per_cat = {pronoun_cat: all_pronouns_set.intersection(pronouns_dict[pronoun_cat])
                        for pronoun_cat in pronouns_dict}
    pronoun_counts_per_cat = {pronoun_cat: sum([all_pronoun_counts[p] for p in pronouns_per_cat[pronoun_cat]])
                              for pronoun_cat in pronouns_dict}
    if verbose:
        print(all_pronouns)
    
    return pronoun_counts_per_cat

def extract_negation_counts(song_title, neg_lemmas={'no','not',"n't",'nothing','never','noone','nowhere'},
                            verbose=False):
    doc = song_title2doc[song_title]
    all_negs = [tok.lemma_ for tok in doc if tok.lemma_ in neg_lemmas]
    
    if verbose:
        print(all_negs)
    return len(all_negs)

PATH_TO_VAD = '/nlp/u/yiweil/datasets/NRC-VAD-Lexicon.txt'
vad_df = pd.read_csv(PATH_TO_VAD,sep='\t')
lookups = {feat: dict(zip(vad_df['Word'].values, vad_df[feat.capitalize()].values))
           for feat in ['valence','arousal','dominance']}

def extract_VAD(song_title, vad_lookups=lookups, verbose=False):
    """Sums up the total valence, arousal, and dominance of the song by aggregating over lemmas."""
    doc = song_title2doc[song_title]
    all_lemmas = [tok.lemma_ for tok in doc]
    lemma_counts = Counter(all_lemmas)
    vad_vocab = set(vad_lookups['valence'].keys())
    shared_vocab = set([tok.lemma_ for tok in doc]).intersection(vad_vocab)
    vad_totals = {feat: sum([vad_lookups[feat][lemma]*lemma_counts[lemma] for lemma in shared_vocab])
                  for feat in vad_lookups}
    
    if verbose:
        shared_vocab = list(shared_vocab)
        df = pd.DataFrame({
            'word': shared_vocab,
            'count': [lemma_counts[w] for w in shared_vocab],
            'valence': [vad_lookups['valence'][w] for w in shared_vocab],
            'arousal': [vad_lookups['arousal'][w] for w in shared_vocab],
            'dominance': [vad_lookups['dominance'][w] for w in shared_vocab],
        })
        display(df)
        print(df['count'].dot(df['valence']))
        print(df['count'].dot(df['arousal']))
        print(df['count'].dot(df['dominance']))
    
    return vad_totals

def extract_length(song_title, verbose=False):
    lyrics = song_title2lyrics[song_title]
    lines = lyrics.split('\n')
    nonempty_lines = [line for line in lines if len(line.strip()) > 0]
    words = lyrics.split()
    nonempty_words = [word for word in words if len(word.strip()) > 0]
    num_words_per_line = [len([word for word in line.split() if len(word.strip()) > 0]) 
                          for line in nonempty_lines]
    num_chars_per_word = [len(word) for word in nonempty_words]
    
    if verbose:
        print(nonempty_lines)
        print(nonempty_words)
        print(num_words_per_line)
        print(num_chars_per_word)
        
    return {'total_num_lines': len(nonempty_lines),
            'total_num_words': len(nonempty_words),
            'total_num_lemmas': len(set([tok.lemma_ for tok in song_title2doc[song_title]])),
            'mean_words_per_line': np.mean(num_words_per_line), 
            'max_words_per_line': np.max(num_words_per_line),
            'mean_chars_per_word': np.mean(num_chars_per_word),
            'max_chars_per_word': np.max(num_chars_per_word)}

song_title2lyrics = pickle.load(open('song2lyrics.pkl','rb'))
lyrics2song_title = dict(zip(song_title2lyrics.values(), song_title2lyrics.keys()))
song_title2dedup_lyrics = pickle.load(open('song2dedup_lyrics.pkl','rb'))

import spacy
from spacy.tokens import DocBin

nlp = spacy.load("en_core_web_sm")
bytes_data = pickle.load(open('pickled_spacy_docs/bytes_data.pkl','rb'))
doc_bin = DocBin().from_bytes(bytes_data)
docs = list(doc_bin.get_docs(nlp.vocab))
song_title2doc = dict(zip(song_title2dedup_lyrics.keys(), docs))

song_titles = list(song_title2dedup_lyrics.keys())
song_feats_df = pd.DataFrame({
    'reg_title': song_titles,
    'ttr': [extract_TTR(song) for song in song_titles],
    'pronoun_counts': [extract_pronoun_counts(song) for song in song_titles],
    'neg_counts': [extract_negation_counts(song) for song in song_titles],
    'VAD': [extract_VAD(song) for song in song_titles],
    'lengths': [extract_length(song) for song in song_titles],})
for pronoun_cat in PRONOUNS_DICT:
    song_feats_df[pronoun_cat] = song_feats_df['pronoun_counts'].apply(lambda x: x[pronoun_cat])
for vad_feat in lookups:
    song_feats_df[vad_feat] = song_feats_df['VAD'].apply(lambda x: x[vad_feat])
LEN_FEATS = ['total_num_lines','total_num_words','total_num_lemmas',
                 'mean_words_per_line','max_words_per_line',
                 'mean_chars_per_word','max_chars_per_word']
for len_feat in LEN_FEATS:
    song_feats_df[len_feat] = song_feats_df['lengths'].apply(lambda x: x[len_feat])

song_title2feature = dict()
for col in song_feats_df:
    if col in {'ttr','neg_counts','first_sg','first_pl','second','third_sg_f','third_sg_m','third_sg_n','third_pl',
               'valence','arousal','dominance',
               'total_num_lines','total_num_words','total_num_lemmas','mean_words_per_line','mean_chars_per_word'}:
        song_title2feature[col] = dict(zip(song_feats_df['reg_title'].values, song_feats_df[col].values))

text2ttr = {text: song_title2feature['ttr'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2neg = {text: song_title2feature['neg_counts'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2first_sg = {text: song_title2feature['first_sg'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2first_pl = {text: song_title2feature['first_pl'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2second = {text: song_title2feature['second'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2she = {text: song_title2feature['third_sg_f'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2he = {text: song_title2feature['third_sg_m'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2it = {text: song_title2feature['third_sg_n'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2they = {text: song_title2feature['third_pl'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2valence = {text: song_title2feature['valence'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2arousal = {text: song_title2feature['arousal'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2dominance = {text: song_title2feature['dominance'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2total_lines = {text: song_title2feature['total_num_lines'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2total_words = {text: song_title2feature['total_num_words'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2total_lemmas = {text: song_title2feature['total_num_lemmas'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2mean_words = {text: song_title2feature['mean_words_per_line'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}
text2mean_chars = {text: song_title2feature['mean_chars_per_word'][lyrics2song_title[text]]
            for text in song_title2lyrics.values()}