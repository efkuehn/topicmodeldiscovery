
# Parsing PDFs 

Pull in all of the PDF files and create objects for the text inside each one. 



[One of the sources I am using for the topic modeling](https://www.machinelearningplus.com/nlp/topic-modeling-gensim-python/)

[This is a good post on Lemmatizing in python](https://www.machinelearningplus.com/nlp/lemmatization-examples-python/)

For the pdf manipulation we use `PyPDF2`. This allows the text to be easily extracted from the scanned pdfs. 


```python
import PyPDF2 
from glob import glob

pdfs = glob('../pdfs/*.pdf') 
```

# Preparing Texts 

Because these texts are pdf scans a lot of clean up will need to go into the OCR. It would be better to add spell check and other things, but we will just be pulling out the stop words, the obvious misspellings, and misdivided words. We used the `WordNetLemmatize` for lemmatizing the words, and SciKit Learn's stopwords collection for english. 

The first time this is ran, you will need to download both `wordnet` and `punkt` for the NLTK libraries to work. 

The NLTK library is slow, another option would have been to use [SpaCy](https://spacy.io/) to do the tokenization and lemmatization. The slowness of NLTK really showed up when we ran this over the entire corpus. 

## Lemmatizing and cleaning 


```python
import string
import nltk 
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
nltk.download('punkt')
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS 

ADD_PUNC = '”“’’–˙ˆ‘'
STOPWORDS = {'d', 'c', 'e', 's', 'œ', 'dhs', 'hk', 'nagy', 'eology', 'ey', 'g', 'ing', 'tion', 'er', 'rst', 'vol', 'ed'} 
AUTHOR_NAMES = {'cruz', 'frederiks', 'nagy', 'snyder', 'nguyen', 'prior', 'cavanaugh', 'heyer', 'schmil', 'smith', 'groody', 'campese', 'izuzquiza', 'heimburger', 'myers', 'colwell', 'olofinjana', 'krabill', 'norton', 'theocharous', 'nacpil', 'nnamani', 'soares', 'thompson', 'zendher', 'ahn', 'haug', 'sarmiento', 'davidson', 'rowlands', 'strine', 'zink', 'jimenez'}
STOPWORDS = STOPWORDS.union(AUTHOR_NAMES)
STOPWORDS = STOPWORDS.union(ENGLISH_STOP_WORDS)
PUNCDIG_TRANSLATOR = str.maketrans('', '', string.punctuation+string.digits+ADD_PUNC)

lemmatizer = WordNetLemmatizer()
```

    [nltk_data] Downloading package wordnet to
    [nltk_data]     /Users/sgoodwin/nltk_data...
    [nltk_data]   Package wordnet is already up-to-date!
    [nltk_data] Downloading package punkt to /Users/sgoodwin/nltk_data...
    [nltk_data]   Package punkt is already up-to-date!



```python
def text_clean(text):
    '''This function uses NLTK to tokenize the text (into words) as well as 
    remove stop words, and lemmatize the words. So that `dogs` becomes `dog`.'''
    clean_list = []
    words = nltk.word_tokenize(text)
    for w in words:
        if w not in STOPWORDS and len(w) > 2: # removing two character words
            # The Translator should have been placed before the if test
            w = w.translate(PUNCDIG_TRANSLATOR)
            if w != '':
                clean_list.append(lemmatizer.lemmatize(w))
    return clean_list
```


```python
text_clean("The striped bats are hanging on their feet3 for best.".lower())
```




    ['striped', 'bat', 'hanging', 'foot', 'best']



## Extracting Texts

PyPDf2 takes an open file object. The following cells show a couple of features of the library.


```python
pdf = open(pdfs[0], 'rb')
pdf_obj = PyPDF2.PdfFileReader(pdf)

```

    PdfReadWarning: Xref table not zero-indexed. ID numbers for objects will be corrected. [pdf.py:1736]



```python
print('No. of pages: {}'.format(pdf_obj.numPages))
```

    No. of pages: 19


This function will be used to create two different lists of lists. One is a running list of the pages of the text, the second is a list of tuples with the file_name and page number for the given extract. This will allow look up of a particular page the corpus list is refering to. 


```python
def pdf_extractor(pdf, corpus_list, text_list):
    '''Extract the text of pdfs and return a dictionary with
    the file name as a key, and the value being a list of the pages
    and the containing texts
    '''
    pdf_file_obj = open(pdf, 'rb')
    pdf_obj = PyPDF2.PdfFileReader(pdf_file_obj)
    for pn in range(0,pdf_obj.numPages):
        page = pdf_obj.getPage(pn)
        text = page.extractText().lower()
        cleaned_list = text_clean(text)
        corpus_list.append(cleaned_list)
        text_list.append((pdf, pn))
        # if you want to create a dictionary
        # text_dict.setdefault(pdf, []).append(page.extractText())
    pdf_file_obj.close()
    return corpus_list, text_list
```


```python
corpus_list = []
text_list = []

for pdf in pdfs:
    corpus_list, text_list = pdf_extractor(pdf, corpus_list, text_list)
```

# Creating LDA Model 

[LDA Topic Models](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation) are short for Latent Dirichlet allocation. It does have an advantages, but it isn't a deterministic algorithim like [NMF](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization). This means, that everytime the follwoing code is run, it will produce a different model. 

In the future I think it would be helpful to look at using NMF models for this process. 

The form of the model that we are using is installed with gensim. 


```python
from gensim import corpora 
from gensim.models.ldamodel import LdaModel 
```

There are a lot of different perameters that can be tweaked inside a topic mdoel. One of the big points about topic modeling is the number of topics. Random_state gives the computer a starting point. Chunksize is how big a piece is taken in. Filter-extremes means that rare words and super common words are not used for topic consideration. 


```python
def prepare_topic_model(corpus_list):
    corpus_dict = corpora.Dictionary(corpus_list)
    corpus_dict.filter_extremes(no_below=100, no_above=0.5)
    corpus = [corpus_dict.doc2bow(text) for text in corpus_list]
    lda_model = LdaModel(corpus=corpus, 
                        id2word=corpus_dict, num_topics=25,
                        random_state=100, update_every=1,
                        chunksize=100, passes=50,
                        alpha='symmetric', per_word_topics=True)
    return lda_model, corpus, corpus_dict
```


```python
lda_model, corpus, corpus_dict = prepare_topic_model(corpus_list)
```


```python
import json
# After the model has been created save the model 
# lda_model.save('./models/PrelimTopicModel2')
# corpus_dict.save_as_text('./models/corpus_dictionary_2')
# with open('./models/corpus.json', 'w') as fp:
#    json.dump(corpus, fp)
```
