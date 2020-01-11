#!/usr/bin/env python3

import concurrent.futures
from collections import Counter
from collections import namedtuple
import json

import glob
import string

# Topic Modeling libraries
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.corpora.dictionary import Dictionary
from htrc_features import FeatureReader

# Language Processing libraries
import nltk
from nltk.corpus import wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

ADD_PUNC = "”“’’–˙ˆ‘"
STOPWORDS = {
    "d",
    "c",
    "e",
    "s",
    "œ",
    "dhs",
    "hk",
    "nagy",
    "eology",
    "ey",
    "g",
    "ing",
    "tion",
    "er",
    "rst",
    "vol",
    "ed",
}
AUTHOR_NAMES = {
    "cruz",
    "frederiks",
    "nagy",
    "snyder",
    "nguyen",
    "prior",
    "cavanaugh",
    "heyer",
    "schmil",
    "smith",
    "groody",
    "campese",
    "izuzquiza",
    "heimburger",
    "myers",
    "colwell",
    "olofinjana",
    "krabill",
    "norton",
    "theocharous",
    "nacpil",
    "nnamani",
    "soares",
    "thompson",
    "zendher",
    "ahn",
    "haug",
    "sarmiento",
    "davidson",
    "rowlands",
    "strine",
    "zink",
    "jimenez",
}
STOPWORDS = STOPWORDS.union(AUTHOR_NAMES)
STOPWORDS = STOPWORDS.union(ENGLISH_STOP_WORDS)
PUNCDIG_TRANSLATOR = str.maketrans(
    "", "", string.punctuation + string.digits + ADD_PUNC
)

Topic = namedtuple("Topic", ["top_num", "perc"])
BestMatch = namedtuple("BestMatch", ["page", "top_num", "perc"])
Book = namedtuple("Book", ["ht_id", "top_topic", "best_match", "most_common_topic"])


def text_clean(text, lemmatizer):
    clean_list = []
    words = nltk.word_tokenize(text)
    for w in words:
        if w not in STOPWORDS and len(w) > 2:  # removing two character words
            w = w.translate(PUNCDIG_TRANSLATOR)
            if w != "":
                clean_list.append(lemmatizer.lemmatize(w))
    return clean_list


def mean(lst):
    return sum(lst) / len(lst)


def volume_parser(vol, lemmatizer):
    vol_list = []
    for page in vol.pages():
        df = page.tokenlist("body", case=False, pos=False)
        dicty = df.to_dict()
        count = dicty["count"]
        clean_list = []
        for key in count.keys():
            w = key[2]
            if w not in STOPWORDS and len(w) > 2:  # removing two character words
                w = w.translate(PUNCDIG_TRANSLATOR)
                if w != "":
                    clean_list += [lemmatizer.lemmatize(w)] * count[key]
                    # clean_list.append(lemmatizer.lemmatize(w) * count[key])
        vol_list.append(clean_list)
    return vol_list


def get_topic_average(sorted_list):
    dicty = {}
    for (_, x, y) in sorted_list:
        dicty.setdefault(x, []).append(y)
    topic_averages = [(x, mean(y)) for x, y in dicty.items()]
    return topic_averages


def analyze_corpus_with_model(other_corpus, lda_model):
    pot_match = []
    for doc_num, doc in enumerate(other_corpus):
        vector = lda_model[doc]
        # row = sorted(vector[0], key=lambda x: x[1], reverse=True)
        # topic_num, prop_topic = row[0]
        topic_num, prop_topic = max(vector[0], key=lambda x: x[1])
        if topic_num in (0, 1, 3, 5, 6, 11) and prop_topic > 0.04:
            pot_match.append((doc_num, topic_num, prop_topic))
    return pot_match
    """
    sorted_list = sorted(pot_match, key=lambda x: x[-1], reverse=True)
    return sorted_list
    """


def corpus_parser(corpus_list, corpus_dict, ldamodel):
    other_corpus = [corpus_dict.doc2bow(text) for text in corpus_list]
    sorted_list = analyze_corpus_with_model(other_corpus, ldamodel)
    best_match = max(sorted_list, key=lambda x: x[-1])
    most_common_topic = Counter([x[1] for x in sorted_list]).most_common(1).pop()
    top_topic = max(get_topic_average(sorted_list), key=lambda x: x[-1])
    return best_match, most_common_topic, top_topic


def analyzed_row_creator(volume, corpus_dict, ldamodel, lemmatizer):
    corpus_list = volume_parser(volume, lemmatizer)
    best_match, most_common_topic, top_topic = corpus_parser(
        corpus_list, corpus_dict, ldamodel
    )
    return Book(
        volume.id, Topic(*top_topic), BestMatch(*best_match), Topic(*most_common_topic)
    )


def main():
    lda_model = LdaModel.load("./models/PrelimTopicModel2")
    corpus_dict = Dictionary.load_from_text("./models/corpus_dictionary_2")
    lemmatizer = WordNetLemmatizer()
    files = glob.glob("../data/test/*.bz2")
    fr = FeatureReader(files)
    output_data = {}
    print(wn.__class__)
    wn.ensure_loaded()
    print(wn.__class__)
    """
    for vol in fr.volumes():
        output_data[vol.id] = analyzed_row_creator(
            vol, corpus_dict, lda_model, lemmatizer
        )
    """
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        future_to_id = {
            executor.submit(
                analyzed_row_creator, vol, corpus_dict, lda_model, lemmatizer
            ): vol.id
            for vol in fr.volumes()
        }
        for future in concurrent.futures.as_completed(future_to_id):
            ht_id = future_to_id[future]
            try:
                data = future.result()
                output_data[ht_id] = data
            except Exception as exc:
                print("%r generated an exception: %s" % (ht_id, exc))
            else:
                output_data[ht_id] = data
    with open("analyzed_corpus1.json", "w") as fp:
        fp.write(str(output_data))


if __name__ == "__main__":
    main()
