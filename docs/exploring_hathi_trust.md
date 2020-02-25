
# Finding The Topics in HathiTrust Data 

To apply the topic model, we need to import it and the `corpus_dict` again. 


```python
from gensim.models.ldamodel import LdaModel 
from gensim.corpora.dictionary import Dictionary
```


```python
lda_model = LdaModel.load('./models/PrelimTopicModel2') 
corpus_dict = Dictionary.load_from_text('./models/corpus_dictionary_2')
```

We also import the text cleaning resources from `pool_processing.py`. These are the same text cleaning resources we used for creating the topic model in `create_topic_models.ipynb`. `pool_process.py` is the script used to iterate over the entire Political Theology corpus. 


```python
from pool_processing import STOPWORDS
from pool_processing import PUNCDIG_TRANSLATOR
from pool_processing import WordNetLemmatizer
```


```python
lemmatizer = WordNetLemmatizer()
```

Thankfully, HathiTrust as created a really helpful tool for feature extraction the [htrc-feature-reader](https://github.com/htrc/htrc-feature-reader) provides tools for accessing the data. 


```python
from htrc_features import FeatureReader
```

To get test files, to explore how the feature reader works, and how the LDA model works with some of the files in our Political Theology corpus, we can use the feature reader to download files directly from HathiTrust. This method will be too slow when we iterate over the entire corpus, but this will work as we explore the corpus. 


```python
fr = FeatureReader(ids=['chi.19073766'])
# other option: uc1.32106018820081
```

The following code produces a list of the volume objects collected from HathiTrust. These volumes can the be explored to see what is in the volume objects. 


```python
volumes = [vol for vol in fr.volumes()]
```


```python
print(volumes[0].id, ' | ', volumes[0].title)
```

    chi.19073766  |  The French Revolution : a political history, 1789-1804 / by A. Aulard ; translated from the French of the 3rd ed., with a preface, notes, and historical summary by Bernard Miall.


The following cell cleans the volume list according to the functions imported from `pool_process`, the data from HathiTrust is in a Pandas Dataframe. Because these words appear as feature extractions, we need to multiply their count based on how many times that word appears in the process. Also, 


```python
vol_list = []
for vol in volumes:
    for page in vol.pages():
        df = page.tokenlist('body', case=False, pos=False)
        dicty = df.to_dict()
        count = dicty['count']
        clean_list = []
        for key in count.keys():
            w = key[2]
            if w not in STOPWORDS and len(w) > 2: # removing two character words
                # The Translator should have been placed before the if test
                w = w.translate(PUNCDIG_TRANSLATOR)
                if w != '':
                    clean_list += [lemmatizer.lemmatize(w)] * count[key]
        vol_list.append(clean_list)

        
```

Here is what one of the pages looks like after it has been cleaned. Notice these are just collection of the words. Many of them are errors because of OCR problems. This is one advantage to LDA topic modeling, it can be forgiving of errors as long as there is enough text. 


```python
vol_list[100]
```




    ['erberhvth',
     'th',
     'brissot',
     'admitted',
     'afﬁliated',
     'arcis',
     'ary',
     'assemble',
     'assembly',
     'attempted',
     'bayonne',
     'body',
     'bordeaux',
     'built',
     'canton',
     'cavalry',
     'central',
     'ceux',
     'chalons',
     'church',
     'citizen',
     'club',
     'club',
     'complaint',
     'composed',
     'conquered',
     'contrary',
     'convention',
     'convert',
     'cordeliers',
     'danton',
     'day',
     'deliberation',
     'demanded',
     'demonstration',
     'department',
     'department',
     'department',
     'deputation',
     'deputation',
     'deputation',
     'deputation',
     'detach',
     'devol',
     'did',
     'direction',
     'défendront',
     'eightyfour',
     'eightyfour',
     'electoral',
     'entire',
     'expelled',
     'fact',
     'far',
     'federal',
     'federal',
     'federal',
     'federal',
     'federal',
     'federal',
     'federative',
     'following',
     'fortyeight',
     'french',
     'friend',
     'gallery',
     'garrison',
     'gireydupré',
     'girondist',
     'girondist',
     'guard',
     'gué',
     'hall',
     'i',
     'invitation',
     'jacobin',
     'jacobin',
     'janu—',
     'lanthenas',
     'later',
     'le',
     'liberty',
     'lisieux',
     'lorient',
     'majority',
     'man',
     'marat',
     'marseillais',
     'mean',
     'member',
     'met',
     'miscarried',
     'morning',
     'mountain',
     'movement',
     'nantes',
     'obtained',
     'october',
     'ouvet',
     'paris',
     'parisian',
     'pelled',
     'people',
     'permission',
     'perpignan',
     'policy',
     'qui',
     'ran',
     'receive',
     'refrain',
     'representative',
     'representing',
     'riom',
     'robespierre',
     'roland',
     'saintbon',
     'secession',
     'secession',
     'section',
     'section',
     'share',
     'sixteen',
     'society',
     'society',
     'song',
     'soon',
     'specially',
     'spend',
     'street',
     'striking',
     'swore',
     'threatening',
     'threat',
     'tous',
     'téte',
     'undertook',
     'use',
     'valognes',
     'were']



After the text is cleaned in needs to be turned into the list of vectors that can be used to run the topic model against. This is where the `corpus_dict` comes in.


```python
other_corpus = [corpus_dict.doc2bow(text) for text in vol_list]
```

The particular vectors for a given page can be explored in the model.


```python
vector = lda_model[other_corpus[100]]
```

The zeroth item of the vector shows the closest topic matches


```python
vector[0]
```




    [(5, 0.34206182), (6, 0.15319963), (12, 0.37902364)]



When we run the full text corpus through the topic model, we can see which pages match best to the topics defined in our test corpus. 


```python
pot_match = []
for doc_num, doc in enumerate(other_corpus):
    vector = lda_model[doc]
    row = sorted(vector[0], key=lambda x: x[1], reverse=True)
    topic_num, prop_topic = row[0]
    # notice, we are filtering on only the topics we think are   
    # interesting and most coherent.
    if topic_num in (0, 1, 3, 5, 6, 11):
        pot_match.append((doc_num, topic_num, prop_topic))
    
    
```

The list of potential matches can be sorted based on the percentage given by the model with the follwing line.


```python
sorted(pot_match, key=lambda x: x[-1], reverse=True)
```




    [(375, 5, 0.8198943),
     (154, 5, 0.8079995),
     (24, 1, 0.8039998),
     (115, 5, 0.7599994),
     (227, 5, 0.7599993),
     (393, 1, 0.7599976),
     (121, 5, 0.75499904),
     (374, 5, 0.7309115),
     (208, 5, 0.67999905),
     (67, 5, 0.66688335),
     (213, 5, 0.65270615),
     (79, 5, 0.6517023),
     (232, 6, 0.6106287),
     (228, 5, 0.5843488),
     (185, 11, 0.577145),
     (103, 5, 0.5682204),
     (186, 11, 0.5482592),
     (42, 5, 0.5455658),
     (129, 5, 0.5329784),
     (174, 5, 0.5318797),
     (8, 5, 0.51999915),
     (359, 5, 0.51999915),
     (236, 5, 0.5180849),
     (328, 1, 0.5100005),
     (139, 1, 0.50999945),
     (166, 11, 0.50500137),
     (62, 5, 0.50333136),
     (263, 11, 0.50250286),
     (100, 5, 0.49848774),
     (183, 11, 0.49669254),
     (145, 5, 0.4828389),
     (268, 6, 0.460285),
     (246, 5, 0.45520547),
     (184, 11, 0.43896687),
     (9, 11, 0.43428633),
     (46, 11, 0.42833412),
     (229, 5, 0.4201152),
     (314, 5, 0.4197382),
     (10, 6, 0.4171695),
     (136, 5, 0.41648477),
     (182, 11, 0.4160481),
     (164, 11, 0.41353098),
     (81, 11, 0.40800238),
     (270, 11, 0.40400216),
     (255, 6, 0.4007121),
     (171, 11, 0.399741),
     (322, 5, 0.3915353),
     (385, 5, 0.39085233),
     (134, 5, 0.38020995),
     (283, 5, 0.36365366),
     (271, 6, 0.36308786),
     (53, 5, 0.35905674),
     (49, 11, 0.34905064),
     (65, 5, 0.3481434),
     (34, 5, 0.3466663),
     (188, 11, 0.34500867),
     (261, 6, 0.34464112),
     (262, 11, 0.34462065),
     (326, 5, 0.34362164),
     (330, 5, 0.34173334),
     (192, 11, 0.34000042),
     (325, 3, 0.33778),
     (52, 5, 0.33777723),
     (159, 6, 0.33389673),
     (161, 3, 0.33381954),
     (64, 5, 0.3328618),
     (155, 11, 0.3301619),
     (175, 6, 0.32662037),
     (128, 11, 0.32293704),
     (187, 11, 0.31517765),
     (388, 5, 0.31238306),
     (390, 5, 0.3090886),
     (247, 5, 0.30744436),
     (269, 5, 0.30293378),
     (265, 6, 0.29570827),
     (48, 11, 0.2870301),
     (258, 11, 0.27589446),
     (160, 5, 0.27569914),
     (140, 5, 0.27295923),
     (97, 3, 0.26865536),
     (294, 5, 0.26687807),
     (303, 1, 0.25153613),
     (16, 6, 0.24684039),
     (20, 11, 0.20762493),
     (63, 0, 0.19941553),
     (0, 0, 0.04),
     (1, 0, 0.04),
     (2, 0, 0.04),
     (3, 0, 0.04),
     (4, 0, 0.04),
     (5, 0, 0.04),
     (7, 0, 0.04),
     (11, 0, 0.04),
     (12, 0, 0.04),
     (13, 0, 0.04),
     (29, 0, 0.04),
     (44, 0, 0.04),
     (153, 0, 0.04),
     (357, 0, 0.04),
     (363, 0, 0.04),
     (396, 0, 0.04),
     (397, 0, 0.04),
     (398, 0, 0.04),
     (399, 0, 0.04),
     (400, 0, 0.04),
     (401, 0, 0.04)]



Because we still have vol_list defined as one particular volume, you can compare what words were fed to the model to produce the above results. 


```python
vol_list[375]
```




    ['th',
     'th',
     'th',
     'rd',
     'th',
     'th',
     'th',
     'tribunal',
     'accepted',
     'adjourned',
     'alternative',
     'appeared',
     'article',
     'article',
     'assemblage',
     'assemble',
     'assemble',
     'club',
     'commune',
     'commune',
     'condemned',
     'conformably',
     'constitution',
     'contrary',
     'correctional',
     'court',
     'day',
     'debated',
     'delivered',
     'dissolved',
     'duplantier',
     'election',
     'following',
     'following',
     'franc',
     'french',
     'fructidor',
     'general',
     'general',
     'germinal',
     'germinal',
     'germinal',
     'guilty',
     'heard',
     'imprisonment',
     'increased',
     'individual',
     'inhabitant',
     'inhabitant',
     'law',
     'law',
     'law',
     'law',
     'law',
     'led',
     'mailhe',
     'mailhes',
     'majority',
     'measure',
     'measure',
     'meet',
     'meeting',
     'member',
     'member',
     'messidor',
     'month',
     'month',
     'motion',
     'motion',
     'motion',
     'new',
     'number',
     'occupying',
     'occupying',
     'order',
     'order',
     'pay',
     'people',
     'police',
     'policecourts',
     'political',
     'political',
     'political',
     'premise',
     'present',
     'prevent',
     'principal',
     'principle',
     'principle',
     'private',
     'private',
     'professed',
     'professed',
     'prohibited',
     'prohibition',
     'proposal',
     'proposed',
     'proposed',
     'proprietor',
     'proscribe',
     'prosecuted',
     'provisionally',
     'punished',
     'punishment',
     'question',
     'question',
     'receive',
     'reconstitution',
     'relating',
     'repealed',
     'replaced',
     'report',
     'report',
     'respectively',
     'restrictive',
     'resumed',
     'revolutionary',
     'ridiculous',
     'riotous',
     'rise',
     'severity',
     'shall',
     'shall',
     'shall',
     'shall',
     'shall',
     'shall',
     'simpler',
     'society',
     'society',
     'society',
     'society',
     'society',
     'society',
     'subject',
     'sunset',
     'tenant',
     'terminate',
     'thermidor',
     'thermidor',
     'thermidor',
     'time',
     'twice',
     'undergo',
     'vaublanc',
     'worded',
     'year',
     'year',
     'year',
     'the',
     'ﬁne']



The matches below .04 are not worth concedering. The match is too low to be relevant, so we can filter these matches out with the following list comprehension.


```python
short_sort = [x for x in pot_match if x[-1] > .04]
```

With the shortened list, we can easily create a pandas dataframe and do various operiations to find out more about the book as a whole. In the first example, we found the average rating for the topics in the book. This calculates the average rating everytime a given topic is dominant on a page. 


```python
import pandas as pd
```


```python
sorted_df = pd.DataFrame(short_sort, columns=['page', 'topic_num', 'perc'])
sorted_df.groupby(['topic_num'])['perc'].mean()
```




    topic_num
    0     0.199416
    1     0.567107
    3     0.313418
    5     0.483392
    6     0.379959
    11    0.395218
    Name: perc, dtype: float64



This sections adds `mean` as a column in the dataframe


```python
short_sorted_df = pd.DataFrame(short_sort, columns=['page', 'topic_num', 'perc'])
short_sorted_df['mean'] = short_sorted_df.groupby('topic_num')['perc'].transform('mean')
```

This code adds the count of how many times a given topic number is dominant on a page. This would help to calculate how much of the book is about a given topic.


```python
short_sorted_df['count'] = short_sorted_df.groupby(['topic_num'])['topic_num'].transform('count')
```

With `count` as a column, we can see that topic number 5 is the most prominant topic in the book, having the best match, the highest average, and on the most pages. 


```python
short_sorted_df.groupby(['topic_num']).max()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>page</th>
      <th>perc</th>
      <th>mean</th>
      <th>count</th>
    </tr>
    <tr>
      <th>topic_num</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>63</td>
      <td>0.199416</td>
      <td>0.199416</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>393</td>
      <td>0.804000</td>
      <td>0.567107</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>325</td>
      <td>0.337780</td>
      <td>0.313418</td>
      <td>3</td>
    </tr>
    <tr>
      <th>5</th>
      <td>390</td>
      <td>0.819894</td>
      <td>0.483392</td>
      <td>43</td>
    </tr>
    <tr>
      <th>6</th>
      <td>271</td>
      <td>0.610629</td>
      <td>0.379959</td>
      <td>10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>270</td>
      <td>0.577145</td>
      <td>0.395218</td>
      <td>23</td>
    </tr>
  </tbody>
</table>
</div>



This is what the dataframe looks like now. 


```python
short_sorted_df.head(15)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>page</th>
      <th>topic_num</th>
      <th>perc</th>
      <th>mean</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8</td>
      <td>5</td>
      <td>0.519999</td>
      <td>0.483392</td>
      <td>43</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9</td>
      <td>11</td>
      <td>0.434286</td>
      <td>0.395218</td>
      <td>23</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10</td>
      <td>6</td>
      <td>0.417170</td>
      <td>0.379959</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>16</td>
      <td>6</td>
      <td>0.246840</td>
      <td>0.379959</td>
      <td>10</td>
    </tr>
    <tr>
      <th>4</th>
      <td>20</td>
      <td>11</td>
      <td>0.207625</td>
      <td>0.395218</td>
      <td>23</td>
    </tr>
    <tr>
      <th>5</th>
      <td>24</td>
      <td>1</td>
      <td>0.804000</td>
      <td>0.567107</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>34</td>
      <td>5</td>
      <td>0.346666</td>
      <td>0.483392</td>
      <td>43</td>
    </tr>
    <tr>
      <th>7</th>
      <td>42</td>
      <td>5</td>
      <td>0.545566</td>
      <td>0.483392</td>
      <td>43</td>
    </tr>
    <tr>
      <th>8</th>
      <td>46</td>
      <td>11</td>
      <td>0.428334</td>
      <td>0.395218</td>
      <td>23</td>
    </tr>
    <tr>
      <th>9</th>
      <td>48</td>
      <td>11</td>
      <td>0.287030</td>
      <td>0.395218</td>
      <td>23</td>
    </tr>
    <tr>
      <th>10</th>
      <td>49</td>
      <td>11</td>
      <td>0.349051</td>
      <td>0.395218</td>
      <td>23</td>
    </tr>
    <tr>
      <th>11</th>
      <td>52</td>
      <td>5</td>
      <td>0.337777</td>
      <td>0.483392</td>
      <td>43</td>
    </tr>
    <tr>
      <th>12</th>
      <td>53</td>
      <td>5</td>
      <td>0.359057</td>
      <td>0.483392</td>
      <td>43</td>
    </tr>
    <tr>
      <th>13</th>
      <td>62</td>
      <td>5</td>
      <td>0.503331</td>
      <td>0.483392</td>
      <td>43</td>
    </tr>
    <tr>
      <th>14</th>
      <td>63</td>
      <td>0</td>
      <td>0.199416</td>
      <td>0.199416</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



## Running over the Corpus 

To run over the entire Political Theology corpus, we can simplify the above explorations into a few functions. These functions form the bases for the `pool_process.py` script that was used in the creation of our data set. 

The doc strings in the following functions explain what the code does. 


```python
from collections import namedtuple
from collections import Counter
```


```python
Topic = namedtuple('Topic', ['top_num', 'perc'])
BestMatch = namedtuple('BestMatch', ['page', 'top_num', 'perc'])
Book = namedtuple('Book', ['ht_id', 'top_topic', 'best_match', 'most_common_topic'])
```


```python
def mean(lst):
    '''This takes an average of a python list. Numpy Arrays have a built in method for this.'''
    return sum(lst) / len(lst)


def volume_parser(vol):
    '''Clean the body of a HathiTrust volume. This runs the Wordnet Lemmatizer'''
    vol_list = []
    for page in vol.pages():
        df = page.tokenlist('body', case=False, pos=False)
        dicty = df.to_dict()
        count = dicty['count']
        clean_list = []
        for key in count.keys():
            w = key[2]
            if w not in STOPWORDS and len(w) > 2: # removing two character words
                w = w.translate(PUNCDIG_TRANSLATOR)
                # The PUNCDIG_TRANSLATOR should have been placed before the test
                if w != '':
                    clean_list += [lemmatizer.lemmatize(w)] * count[key]
        vol_list.append(clean_list)
    return vol_list


def get_topic_average(sorted_list):
    '''This averages the topics in a list of pages, topics, and percentages'''
    dicty = {}
    for (_, x, y) in sorted_list:
        dicty.setdefault(x, []).append(y)
    topic_averages = [(x, mean(y)) for x, y in dicty.items()]
    return topic_averages

        
def analyze_corpus_with_model(other_coprus, lda_model):
    '''Filters an unseen corpus on the original LDA Model'''
    pot_match = []
    for doc_num, doc in enumerate(other_corpus):
        vector = lda_model[doc]
        # row = sorted(vector[0], key=lambda x: x[1], reverse=True)
        # topic_num, prop_topic = row[0]
        topic_num, prop_topic = max(vector[0], key=lambda x: x[1])
        if topic_num in (0, 1, 3, 5, 6, 11) and prop_topic > .04:
            pot_match.append((doc_num, topic_num, prop_topic))
    return pot_match 
    
    
def corpus_parser(corpus_list, corpus_dict, ldamodel):
    '''This takes a new corpus and return the best matches, 
    the most common topics and the top topics in a new corpus. 
    `corpus_list` needs to be a list of lists of the words on a page'''
    other_corpus = [corpus_dict.doc2bow(text) for text in vol_list]
    sorted_list = analyze_corpus_with_model(other_corpus, ldamodel)
    best_match = max(sorted_list, key=lambda x: x[-1])
    most_common_topic = Counter([x[1] for x in sorted_list]).most_common(1).pop()
    top_topic = max(get_topic_average(sorted_list), key=lambda x: x[-1])
    return best_match, most_common_topic, top_topic
    

def file_parser(feature_reader, corpus_dict, analyzed_dict, ldamodel):
    '''returns a dictionary of each volume, and the topics represented
    in that volume.'''
    for vol in feature_reader.volumes():
        corpus_list = volume_parser(vol)
        best_match, most_common_topic, top_topic = corpus_parser(corpus_list, corpus_dict, ldamodel)
        analyzed_dict[vol.id] = Book(vol.id, Topic(*top_topic), BestMatch(*best_match), Topic(*most_common_topic))
    return analyzed_dict  
```

We can try this code on the following 10 items. But making these calls to HathiTrust will take a few minutes to run over all of the code. 


```python
files = ['chi.086834843',
 'chi.096807539',
 'chi.098001406',
 'chi.100957606',
 'chi.19073766',
 'chi.096733853',
 'chi.098001359',
 'chi.098383507',
 'chi.11963941',
 'chi.19080474'
]
```


```python
fr = FeatureReader(ids=files)
analyzed_dict = file_parser(fr, corpus_dict, {}, lda_model)
```

Here is the results of the code. 


```python
analyzed_dict
```




    {'chi.086834843': Book(ht_id='chi.086834843', top_topic=Topic(top_num=1, perc=0.5824437936147054), best_match=BestMatch(page=375, top_num=5, perc=0.81989264), most_common_topic=Topic(top_num=5, perc=47)),
     'chi.096807539': Book(ht_id='chi.096807539', top_topic=Topic(top_num=1, perc=0.5351911102022443), best_match=BestMatch(page=375, top_num=5, perc=0.81989455), most_common_topic=Topic(top_num=5, perc=44)),
     'chi.098001406': Book(ht_id='chi.098001406', top_topic=Topic(top_num=1, perc=0.5570442179838816), best_match=BestMatch(page=375, top_num=5, perc=0.8198953), most_common_topic=Topic(top_num=5, perc=44)),
     'chi.100957606': Book(ht_id='chi.100957606', top_topic=Topic(top_num=1, perc=0.5560638954242071), best_match=BestMatch(page=375, top_num=5, perc=0.81989455), most_common_topic=Topic(top_num=5, perc=43)),
     'chi.19073766': Book(ht_id='chi.19073766', top_topic=Topic(top_num=1, perc=0.6181326270103454), best_match=BestMatch(page=375, top_num=5, perc=0.8198934), most_common_topic=Topic(top_num=5, perc=47)),
     'chi.096733853': Book(ht_id='chi.096733853', top_topic=Topic(top_num=1, perc=0.645752027630806), best_match=BestMatch(page=375, top_num=5, perc=0.8198932), most_common_topic=Topic(top_num=5, perc=45)),
     'chi.098001359': Book(ht_id='chi.098001359', top_topic=Topic(top_num=1, perc=0.5671495258808136), best_match=BestMatch(page=375, top_num=5, perc=0.81989396), most_common_topic=Topic(top_num=5, perc=44)),
     'chi.098383507': Book(ht_id='chi.098383507', top_topic=Topic(top_num=1, perc=0.5658926725387573), best_match=BestMatch(page=375, top_num=5, perc=0.81989247), most_common_topic=Topic(top_num=5, perc=44)),
     'chi.11963941': Book(ht_id='chi.11963941', top_topic=Topic(top_num=1, perc=0.5824437886476517), best_match=BestMatch(page=375, top_num=5, perc=0.8198917), most_common_topic=Topic(top_num=5, perc=46)),
     'chi.19080474': Book(ht_id='chi.19080474', top_topic=Topic(top_num=1, perc=0.6459993571043015), best_match=BestMatch(page=375, top_num=5, perc=0.8198934), most_common_topic=Topic(top_num=5, perc=46))}



## Measuring Performance 

The performance of this code is a little dissapointing. Part of this is because NLTK has not been optimized for performance. SpaCy would have provided a faster way of Lemmantizing. But also LDA is a costly compute. To explore how long the code actually takes to run, and to see where the cost hits are, the following two sections use `timeit` and `cProfile`. 


```python
# Timeit
import timeit
def wrapper(func, *args, **kwargs):
    def wrapped():
        return func(*args, **kwargs)
    return wrapped

wrapped = wrapper(file_parser, fr, corpus_dict, {}, lda_model)
```


```python
import cProfile 
cProfile.run('file_parser(fr, corpus_dict, {}, lda_model)')
```


```python
timeit.timeit(wrapped, number=1)
```

### Navigation 

- [Back](exploring_topic_distribution.md)
- [Home](../README.md)
- [Next](analysing_the_output.md)
