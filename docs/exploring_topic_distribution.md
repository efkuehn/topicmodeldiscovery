

# Find the Dominant Document For Each Topic

Once we have a topic model that has pretty good distribution and the bags of words have fairly coherent topics, we needed to explore the specific topics in the corpus. To do this, we created a pandas dataframe that works similarly to a spreadsheet, but allows all of the functionality of python on top of it. 

The firest two cells import the necessary modules, and load the data. 



```python
import pandas as pd
import json
from gensim import corpora 
from gensim.models.ldamodel import LdaModel 
from gensim.corpora.dictionary import Dictionary
```


```python
lda_model = LdaModel.load('./models/PrelimTOpicModel2') 
corpus_dict = Dictionary.load_from_text('./models/corpus_dictionary_2')
with open('./models/corpus.json', 'r') as fp:
    corpus = json.load(fp)
with open('./models/text_list.json', 'r') as fp:
    text_list = json.load(fp)
with open('./models/corpus_list.json', 'r') as fp:
    corpus_list = json.load(fp)
```

The following code is the primary function that creates the dataframe. This dataframe has a row for each page in the document. Which topic is dominant for the words on the page, and what the distinctive words are for the given topic. It also includes the pdf and page number for the document we are analyzing. 

This allowed us to go back and look at the page for further context, in order to better understand the topics. 


```python
# this creates a pandas DataFrame that orders all of the topics and shows the dominant topic for each document
def format_topics_sent(ldamodel, corpus, texts):
    sent_topics_df = pd.DataFrame()
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row[0], key=lambda x: x[1], reverse=True)
        
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_topic', 'Perc_Contrib', 'Topic_Keywords']
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    sent_topics_df.rename(columns={0: "Text"}, inplace=True)
    return sent_topics_df
```

## Exploring the Dominant Topic Models

In order to better understand the specifics of this code, we can explore each particular row, by creating a generator to look at the rows. 


```python
def format_topics_sent_gen(ldamodel, corpus, texts):
    for i, row in enumerate(ldamodel[corpus]):
        yield row
```


```python
row_generator = format_topics_sent_gen(lda_model, corpus, corpus_list)
```


```python
row = next(row_generator)
```


```python
row
```




    ([(0, 0.010000001),
      (1, 0.010000001),
      (2, 0.010000001),
      (3, 0.010000001),
      (4, 0.010000001),
      (5, 0.010000001),
      (6, 0.010000001),
      (7, 0.010000001),
      (8, 0.010000001),
      (9, 0.76),
      (10, 0.010000001),
      (11, 0.010000001),
      (12, 0.010000001),
      (13, 0.010000001),
      (14, 0.010000001),
      (15, 0.010000001),
      (16, 0.010000001),
      (17, 0.010000001),
      (18, 0.010000001),
      (19, 0.010000001),
      (20, 0.010000001),
      (21, 0.010000001),
      (22, 0.010000001),
      (23, 0.010000001),
      (24, 0.010000001)],
     [(0, [9])],
     [(0, [(9, 2.9999995)])])



For looking at the details of a specific topic, and its word distribution, you can query the lda_model directly. The `topn` variable shows how many items to display


```python
lda_model.show_topic(21, topn=30)
```




    [('god', 0.20524834),
     ('people', 0.08997392),
     ('power', 0.0812579),
     ('faith', 0.057230312),
     ('christian', 0.05547319),
     ('life', 0.05419738),
     ('word', 0.049899396),
     ('world', 0.045320693),
     ('way', 0.031713385),
     ('human', 0.030553361),
     ('reality', 0.025296446),
     ('experience', 0.023166098),
     ('doe', 0.021962931),
     ('tulud', 0.019804804),
     ('need', 0.016158376),
     ('especially', 0.013886022),
     ('like', 0.01281783),
     ('sense', 0.012462943),
     ('particularly', 0.011948879),
     ('fact', 0.011376779),
     ('just', 0.01132918),
     ('make', 0.011165283),
     ('time', 0.0108135445),
     ('g', 0.010585245),
     ('relation', 0.009754408),
     ('good', 0.00956607),
     ('example', 0.009151671),
     ('culture', 0.008596516),
     ('context', 0.008572249),
     ('challenge', 0.007773363)]




```python
sent_topics_df = format_topics_sent(lda_model, corpus, text_list)
```


```python
sent_topics_df
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
      <th>Dominant_topic</th>
      <th>Perc_Contrib</th>
      <th>Topic_Keywords</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>9.0</td>
      <td>0.7600</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Davidson 2018.pdf, 0]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>9.0</td>
      <td>0.6080</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Davidson 2018.pdf, 1]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>9.0</td>
      <td>0.6800</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Davidson 2018.pdf, 2]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21.0</td>
      <td>0.5200</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Davidson 2018.pdf, 3]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.0</td>
      <td>0.6040</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Davidson 2018.pdf, 4]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>9.0</td>
      <td>0.8629</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Davidson 2018.pdf, 5]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>9.0</td>
      <td>0.8080</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Davidson 2018.pdf, 6]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>14.0</td>
      <td>0.5200</td>
      <td>right, human, word, reality, state, world, tim...</td>
      <td>[../pdfs/Davidson 2018.pdf, 7]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.0</td>
      <td>0.0400</td>
      <td>black, experience, life, mean, like, make, poi...</td>
      <td>[../pdfs/Davidson 2018.pdf, 8]</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.0</td>
      <td>0.6800</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Davidson 2018.pdf, 9]</td>
    </tr>
    <tr>
      <th>10</th>
      <td>9.0</td>
      <td>0.8400</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Davidson 2018.pdf, 10]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>9.0</td>
      <td>0.7600</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Davidson 2018.pdf, 11]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.0</td>
      <td>0.0400</td>
      <td>black, experience, life, mean, like, make, poi...</td>
      <td>[../pdfs/Davidson 2018.pdf, 12]</td>
    </tr>
    <tr>
      <th>13</th>
      <td>17.0</td>
      <td>0.5200</td>
      <td>research, form, study, mean, need, way, order,...</td>
      <td>[../pdfs/Davidson 2018.pdf, 13]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>9.0</td>
      <td>0.5200</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Davidson 2018.pdf, 14]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>5.0</td>
      <td>0.5200</td>
      <td>social, political, economic, immigrant, societ...</td>
      <td>[../pdfs/Davidson 2018.pdf, 15]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>13.0</td>
      <td>0.4080</td>
      <td>new, press, ed, york, power, study, global, pe...</td>
      <td>[../pdfs/Davidson 2018.pdf, 16]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>14.0</td>
      <td>0.4080</td>
      <td>right, human, word, reality, state, world, tim...</td>
      <td>[../pdfs/Davidson 2018.pdf, 17]</td>
    </tr>
    <tr>
      <th>18</th>
      <td>0.0</td>
      <td>0.0400</td>
      <td>black, experience, life, mean, like, make, poi...</td>
      <td>[../pdfs/Davidson 2018.pdf, 18]</td>
    </tr>
    <tr>
      <th>19</th>
      <td>7.0</td>
      <td>0.3448</td>
      <td>theology, experience, theological, tulud, cont...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Six. Expanding ...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>8.0</td>
      <td>0.2205</td>
      <td>group, community, religious, social, role, tim...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Six. Expanding ...</td>
    </tr>
    <tr>
      <th>21</th>
      <td>20.0</td>
      <td>0.4337</td>
      <td>struggle, woman, life, oppression, feminist, e...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Six. Expanding ...</td>
    </tr>
    <tr>
      <th>22</th>
      <td>18.0</td>
      <td>0.6215</td>
      <td>woman, feminist, oppression, tulud, particular...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Six. Expanding ...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>20.0</td>
      <td>0.4992</td>
      <td>struggle, woman, life, oppression, feminist, e...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Six. Expanding ...</td>
    </tr>
    <tr>
      <th>24</th>
      <td>24.0</td>
      <td>0.2211</td>
      <td>filipino, philippine, hk, migrant, tulud, just...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Six. Expanding ...</td>
    </tr>
    <tr>
      <th>25</th>
      <td>20.0</td>
      <td>0.4348</td>
      <td>struggle, woman, life, oppression, feminist, e...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Six. Expanding ...</td>
    </tr>
    <tr>
      <th>26</th>
      <td>20.0</td>
      <td>0.4400</td>
      <td>struggle, woman, life, oppression, feminist, e...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Six. Expanding ...</td>
    </tr>
    <tr>
      <th>27</th>
      <td>20.0</td>
      <td>0.2912</td>
      <td>struggle, woman, life, oppression, feminist, e...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Six. Expanding ...</td>
    </tr>
    <tr>
      <th>28</th>
      <td>20.0</td>
      <td>0.5509</td>
      <td>struggle, woman, life, oppression, feminist, e...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Six. Expanding ...</td>
    </tr>
    <tr>
      <th>29</th>
      <td>3.0</td>
      <td>0.2221</td>
      <td>migrant, country, home, community, family, exp...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Six. Expanding ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1089</th>
      <td>7.0</td>
      <td>0.5100</td>
      <td>theology, experience, theological, tulud, cont...</td>
      <td>[../pdfs/Ahn 2018.pdf, 16]</td>
    </tr>
    <tr>
      <th>1090</th>
      <td>5.0</td>
      <td>0.4263</td>
      <td>social, political, economic, immigrant, societ...</td>
      <td>[../pdfs/Ahn 2018.pdf, 17]</td>
    </tr>
    <tr>
      <th>1091</th>
      <td>0.0</td>
      <td>0.0400</td>
      <td>black, experience, life, mean, like, make, poi...</td>
      <td>[../pdfs/Ahn 2018.pdf, 18]</td>
    </tr>
    <tr>
      <th>1092</th>
      <td>9.0</td>
      <td>0.2080</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Haug 2018.pdf, 0]</td>
    </tr>
    <tr>
      <th>1093</th>
      <td>13.0</td>
      <td>0.6800</td>
      <td>new, press, ed, york, power, study, global, pe...</td>
      <td>[../pdfs/Haug 2018.pdf, 1]</td>
    </tr>
    <tr>
      <th>1094</th>
      <td>9.0</td>
      <td>0.3467</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Haug 2018.pdf, 2]</td>
    </tr>
    <tr>
      <th>1095</th>
      <td>9.0</td>
      <td>0.3467</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Haug 2018.pdf, 3]</td>
    </tr>
    <tr>
      <th>1096</th>
      <td>4.0</td>
      <td>0.5095</td>
      <td>migration, context, study, challenge, communit...</td>
      <td>[../pdfs/Haug 2018.pdf, 4]</td>
    </tr>
    <tr>
      <th>1097</th>
      <td>11.0</td>
      <td>0.5200</td>
      <td>religion, religious, culture, cultural, christ...</td>
      <td>[../pdfs/Haug 2018.pdf, 5]</td>
    </tr>
    <tr>
      <th>1098</th>
      <td>7.0</td>
      <td>0.5200</td>
      <td>theology, experience, theological, tulud, cont...</td>
      <td>[../pdfs/Haug 2018.pdf, 6]</td>
    </tr>
    <tr>
      <th>1099</th>
      <td>9.0</td>
      <td>0.3467</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Haug 2018.pdf, 7]</td>
    </tr>
    <tr>
      <th>1100</th>
      <td>9.0</td>
      <td>0.5771</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Haug 2018.pdf, 8]</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>0.0</td>
      <td>0.0400</td>
      <td>black, experience, life, mean, like, make, poi...</td>
      <td>[../pdfs/Haug 2018.pdf, 9]</td>
    </tr>
    <tr>
      <th>1102</th>
      <td>4.0</td>
      <td>0.5200</td>
      <td>migration, context, study, challenge, communit...</td>
      <td>[../pdfs/Haug 2018.pdf, 10]</td>
    </tr>
    <tr>
      <th>1103</th>
      <td>0.0</td>
      <td>0.0400</td>
      <td>black, experience, life, mean, like, make, poi...</td>
      <td>[../pdfs/Haug 2018.pdf, 11]</td>
    </tr>
    <tr>
      <th>1104</th>
      <td>9.0</td>
      <td>0.3467</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Haug 2018.pdf, 12]</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>12.0</td>
      <td>0.5200</td>
      <td>work, place, family, home, like, case, make, m...</td>
      <td>[../pdfs/Haug 2018.pdf, 13]</td>
    </tr>
    <tr>
      <th>1106</th>
      <td>21.0</td>
      <td>0.5100</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Haug 2018.pdf, 14]</td>
    </tr>
    <tr>
      <th>1107</th>
      <td>0.0</td>
      <td>0.0400</td>
      <td>black, experience, life, mean, like, make, poi...</td>
      <td>[../pdfs/Haug 2018.pdf, 15]</td>
    </tr>
    <tr>
      <th>1108</th>
      <td>4.0</td>
      <td>0.3467</td>
      <td>migration, context, study, challenge, communit...</td>
      <td>[../pdfs/Cruz - 2010 - Preliminary Material.pd...</td>
    </tr>
    <tr>
      <th>1109</th>
      <td>7.0</td>
      <td>0.5100</td>
      <td>theology, experience, theological, tulud, cont...</td>
      <td>[../pdfs/Cruz - 2010 - Preliminary Material.pd...</td>
    </tr>
    <tr>
      <th>1110</th>
      <td>23.0</td>
      <td>0.6080</td>
      <td>hong, kong, tulud, filipina, filipino, g, huma...</td>
      <td>[../pdfs/Cruz - 2010 - Preliminary Material.pd...</td>
    </tr>
    <tr>
      <th>1111</th>
      <td>17.0</td>
      <td>0.3276</td>
      <td>research, form, study, mean, need, way, order,...</td>
      <td>[../pdfs/Cruz - 2010 - Preliminary Material.pd...</td>
    </tr>
    <tr>
      <th>1112</th>
      <td>23.0</td>
      <td>0.5200</td>
      <td>hong, kong, tulud, filipina, filipino, g, huma...</td>
      <td>[../pdfs/Cruz - 2010 - Preliminary Material.pd...</td>
    </tr>
    <tr>
      <th>1113</th>
      <td>23.0</td>
      <td>0.3738</td>
      <td>hong, kong, tulud, filipina, filipino, g, huma...</td>
      <td>[../pdfs/Cruz - 2010 - Preliminary Material.pd...</td>
    </tr>
    <tr>
      <th>1114</th>
      <td>5.0</td>
      <td>0.2232</td>
      <td>social, political, economic, immigrant, societ...</td>
      <td>[../pdfs/Cruz - 2010 - Preliminary Material.pd...</td>
    </tr>
    <tr>
      <th>1115</th>
      <td>20.0</td>
      <td>0.3811</td>
      <td>struggle, woman, life, oppression, feminist, e...</td>
      <td>[../pdfs/Cruz - 2010 - Preliminary Material.pd...</td>
    </tr>
    <tr>
      <th>1116</th>
      <td>20.0</td>
      <td>0.5257</td>
      <td>struggle, woman, life, oppression, feminist, e...</td>
      <td>[../pdfs/Cruz - 2010 - Preliminary Material.pd...</td>
    </tr>
    <tr>
      <th>1117</th>
      <td>2.0</td>
      <td>0.4688</td>
      <td>worker, domestic, migrant, filipina, condition...</td>
      <td>[../pdfs/Cruz - 2010 - Preliminary Material.pd...</td>
    </tr>
    <tr>
      <th>1118</th>
      <td>24.0</td>
      <td>0.2966</td>
      <td>filipino, philippine, hk, migrant, tulud, just...</td>
      <td>[../pdfs/Cruz - 2010 - Preliminary Material.pd...</td>
    </tr>
  </tbody>
</table>
<p>1119 rows × 4 columns</p>
</div>



The following code was used, and reused to show the details of a specific topic. This allowed us to see the parallels between the different documents. 


```python
sent_topics_df[sent_topics_df['Dominant_topic'] == 21.0].sort_values('Perc_Contrib', ascending=False)
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
      <th>Dominant_topic</th>
      <th>Perc_Contrib</th>
      <th>Topic_Keywords</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>647</th>
      <td>21.0</td>
      <td>0.7600</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Thompson 2017.pdf, 3]</td>
    </tr>
    <tr>
      <th>1047</th>
      <td>21.0</td>
      <td>0.7127</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Izuzquiza - 2011 - Breaking bread not...</td>
    </tr>
    <tr>
      <th>992</th>
      <td>21.0</td>
      <td>0.7060</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Five. A Differe...</td>
    </tr>
    <tr>
      <th>1054</th>
      <td>21.0</td>
      <td>0.6937</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Izuzquiza - 2011 - Breaking bread not...</td>
    </tr>
    <tr>
      <th>548</th>
      <td>21.0</td>
      <td>0.6903</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Nnamani 2015.pdf, 3]</td>
    </tr>
    <tr>
      <th>520</th>
      <td>21.0</td>
      <td>0.6800</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Strine 2018.pdf, 1]</td>
    </tr>
    <tr>
      <th>810</th>
      <td>21.0</td>
      <td>0.6260</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Three. Expandin...</td>
    </tr>
    <tr>
      <th>1053</th>
      <td>21.0</td>
      <td>0.6158</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Izuzquiza - 2011 - Breaking bread not...</td>
    </tr>
    <tr>
      <th>153</th>
      <td>21.0</td>
      <td>0.6090</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Frederiks and Nagy - 2016 - Religion,...</td>
    </tr>
    <tr>
      <th>546</th>
      <td>21.0</td>
      <td>0.6017</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Nnamani 2015.pdf, 1]</td>
    </tr>
    <tr>
      <th>791</th>
      <td>21.0</td>
      <td>0.5950</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Three. Expandin...</td>
    </tr>
    <tr>
      <th>1052</th>
      <td>21.0</td>
      <td>0.5829</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Izuzquiza - 2011 - Breaking bread not...</td>
    </tr>
    <tr>
      <th>646</th>
      <td>21.0</td>
      <td>0.5767</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Thompson 2017.pdf, 2]</td>
    </tr>
    <tr>
      <th>615</th>
      <td>21.0</td>
      <td>0.5324</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Four. Exploring...</td>
    </tr>
    <tr>
      <th>618</th>
      <td>21.0</td>
      <td>0.5295</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Four. Exploring...</td>
    </tr>
    <tr>
      <th>602</th>
      <td>21.0</td>
      <td>0.5227</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Four. Exploring...</td>
    </tr>
    <tr>
      <th>807</th>
      <td>21.0</td>
      <td>0.5206</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Three. Expandin...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21.0</td>
      <td>0.5200</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Davidson 2018.pdf, 3]</td>
    </tr>
    <tr>
      <th>1023</th>
      <td>21.0</td>
      <td>0.5200</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Rowlands 2018.pdf, 3]</td>
    </tr>
    <tr>
      <th>1087</th>
      <td>21.0</td>
      <td>0.5200</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Ahn 2018.pdf, 14]</td>
    </tr>
    <tr>
      <th>530</th>
      <td>21.0</td>
      <td>0.5200</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Strine 2018.pdf, 11]</td>
    </tr>
    <tr>
      <th>1106</th>
      <td>21.0</td>
      <td>0.5100</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Haug 2018.pdf, 14]</td>
    </tr>
    <tr>
      <th>251</th>
      <td>21.0</td>
      <td>0.4899</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Frederiks and Nagy - 2016 - Religion,...</td>
    </tr>
    <tr>
      <th>625</th>
      <td>21.0</td>
      <td>0.4835</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Four. Exploring...</td>
    </tr>
    <tr>
      <th>772</th>
      <td>21.0</td>
      <td>0.4811</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Three. Expandin...</td>
    </tr>
    <tr>
      <th>215</th>
      <td>21.0</td>
      <td>0.4675</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Frederiks and Nagy - 2016 - Religion,...</td>
    </tr>
    <tr>
      <th>610</th>
      <td>21.0</td>
      <td>0.4641</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Four. Exploring...</td>
    </tr>
    <tr>
      <th>616</th>
      <td>21.0</td>
      <td>0.4612</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Four. Exploring...</td>
    </tr>
    <tr>
      <th>773</th>
      <td>21.0</td>
      <td>0.4561</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Three. Expandin...</td>
    </tr>
    <tr>
      <th>1081</th>
      <td>21.0</td>
      <td>0.4557</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Ahn 2018.pdf, 8]</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1061</th>
      <td>21.0</td>
      <td>0.3035</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Izuzquiza - 2011 - Breaking bread not...</td>
    </tr>
    <tr>
      <th>244</th>
      <td>21.0</td>
      <td>0.3016</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Frederiks and Nagy - 2016 - Religion,...</td>
    </tr>
    <tr>
      <th>981</th>
      <td>21.0</td>
      <td>0.2992</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Five. A Differe...</td>
    </tr>
    <tr>
      <th>228</th>
      <td>21.0</td>
      <td>0.2988</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Frederiks and Nagy - 2016 - Religion,...</td>
    </tr>
    <tr>
      <th>1050</th>
      <td>21.0</td>
      <td>0.2878</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Izuzquiza - 2011 - Breaking bread not...</td>
    </tr>
    <tr>
      <th>770</th>
      <td>21.0</td>
      <td>0.2849</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Three. Expandin...</td>
    </tr>
    <tr>
      <th>281</th>
      <td>21.0</td>
      <td>0.2819</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Jimenez 2019.pdf, 5]</td>
    </tr>
    <tr>
      <th>991</th>
      <td>21.0</td>
      <td>0.2799</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Five. A Differe...</td>
    </tr>
    <tr>
      <th>463</th>
      <td>21.0</td>
      <td>0.2774</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/cruz2010.pdf, 31]</td>
    </tr>
    <tr>
      <th>50</th>
      <td>21.0</td>
      <td>0.2774</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Six. Expanding ...</td>
    </tr>
    <tr>
      <th>155</th>
      <td>21.0</td>
      <td>0.2710</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Frederiks and Nagy - 2016 - Religion,...</td>
    </tr>
    <tr>
      <th>572</th>
      <td>21.0</td>
      <td>0.2698</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Soares et al 2017.pdf, 4]</td>
    </tr>
    <tr>
      <th>74</th>
      <td>21.0</td>
      <td>0.2673</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Frederiks and Nagy - 2016 - Religion,...</td>
    </tr>
    <tr>
      <th>212</th>
      <td>21.0</td>
      <td>0.2673</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Frederiks and Nagy - 2016 - Religion,...</td>
    </tr>
    <tr>
      <th>968</th>
      <td>21.0</td>
      <td>0.2542</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Five. A Differe...</td>
    </tr>
    <tr>
      <th>233</th>
      <td>21.0</td>
      <td>0.2530</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Frederiks and Nagy - 2016 - Religion,...</td>
    </tr>
    <tr>
      <th>573</th>
      <td>21.0</td>
      <td>0.2505</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Soares et al 2017.pdf, 5]</td>
    </tr>
    <tr>
      <th>214</th>
      <td>21.0</td>
      <td>0.2483</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Frederiks and Nagy - 2016 - Religion,...</td>
    </tr>
    <tr>
      <th>620</th>
      <td>21.0</td>
      <td>0.2476</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Four. Exploring...</td>
    </tr>
    <tr>
      <th>511</th>
      <td>21.0</td>
      <td>0.2406</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Two. Frontiers ...</td>
    </tr>
    <tr>
      <th>806</th>
      <td>21.0</td>
      <td>0.2403</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Three. Expandin...</td>
    </tr>
    <tr>
      <th>578</th>
      <td>21.0</td>
      <td>0.2400</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Four. Exploring...</td>
    </tr>
    <tr>
      <th>292</th>
      <td>21.0</td>
      <td>0.2394</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Introduction.pdf, 0]</td>
    </tr>
    <tr>
      <th>621</th>
      <td>21.0</td>
      <td>0.2291</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Four. Exploring...</td>
    </tr>
    <tr>
      <th>568</th>
      <td>21.0</td>
      <td>0.2285</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Soares et al 2017.pdf, 0]</td>
    </tr>
    <tr>
      <th>970</th>
      <td>21.0</td>
      <td>0.2233</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Five. A Differe...</td>
    </tr>
    <tr>
      <th>35</th>
      <td>21.0</td>
      <td>0.2217</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Six. Expanding ...</td>
    </tr>
    <tr>
      <th>448</th>
      <td>21.0</td>
      <td>0.2217</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/cruz2010.pdf, 16]</td>
    </tr>
    <tr>
      <th>340</th>
      <td>21.0</td>
      <td>0.2045</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter One. Geographie...</td>
    </tr>
    <tr>
      <th>498</th>
      <td>21.0</td>
      <td>0.1989</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Two. Frontiers ...</td>
    </tr>
  </tbody>
</table>
<p>109 rows × 4 columns</p>
</div>



To explore each topic was helpful, but one of the things we wanted to see was a shorter dataframe that had the topics and which document best exemplified those documents. The next cell groups the dataframe by the dominant topic, and the next cell creates a new dataframe so that just the best exemplified topics are portrayed. 


```python
grpd_df = sent_topics_df.groupby('Dominant_topic')
```


```python
# This code creates a pandas DataFrame that shows which document is exemplified by which topic
new_df = pd.DataFrame()

for i, grp in grpd_df:
    new_df = pd.concat([new_df, grp.sort_values(['Perc_Contrib'], ascending=[0]).head(1)], axis=0)

new_df.reset_index(drop=True, inplace=True)
new_df.columns = ['Topic_Num', 'Topic_Perc_Contrib', 'Keywords', 'Text']
new_df
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
      <th>Topic_Num</th>
      <th>Topic_Perc_Contrib</th>
      <th>Keywords</th>
      <th>Text</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.0</td>
      <td>0.5200</td>
      <td>black, experience, life, mean, like, make, poi...</td>
      <td>[../pdfs/Rowlands 2018.pdf, 16]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>0.5854</td>
      <td>identity, challenge, term, experience, context...</td>
      <td>[../pdfs/Frederiks and Nagy - 2016 - Religion,...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>0.4688</td>
      <td>worker, domestic, migrant, filipina, condition...</td>
      <td>[../pdfs/Cruz - 2010 - Preliminary Material.pd...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>3.0</td>
      <td>0.5200</td>
      <td>migrant, country, home, community, family, exp...</td>
      <td>[../pdfs/Snyder 2018.pdf, 16]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>4.0</td>
      <td>0.5200</td>
      <td>migration, context, study, challenge, communit...</td>
      <td>[../pdfs/Snyder 2018.pdf, 5]</td>
    </tr>
    <tr>
      <th>5</th>
      <td>5.0</td>
      <td>0.7828</td>
      <td>social, political, economic, immigrant, societ...</td>
      <td>[../pdfs/Jimenez 2019.pdf, 7]</td>
    </tr>
    <tr>
      <th>6</th>
      <td>6.0</td>
      <td>0.6744</td>
      <td>church, christian, american, immigrant, commun...</td>
      <td>[../pdfs/Nnamani 2015.pdf, 5]</td>
    </tr>
    <tr>
      <th>7</th>
      <td>7.0</td>
      <td>0.6938</td>
      <td>theology, experience, theological, tulud, cont...</td>
      <td>[../pdfs/cruz2010.pdf, 34]</td>
    </tr>
    <tr>
      <th>8</th>
      <td>8.0</td>
      <td>0.4646</td>
      <td>group, community, religious, social, role, tim...</td>
      <td>[../pdfs/Frederiks and Nagy - 2016 - Religion,...</td>
    </tr>
    <tr>
      <th>9</th>
      <td>9.0</td>
      <td>0.9751</td>
      <td>œ, dorottya, martha, human, order, case, g, st...</td>
      <td>[../pdfs/Frederiks and Nagy - 2016 - Religion,...</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10.0</td>
      <td>0.6800</td>
      <td>say, way, make, problem, time, life, mean, peo...</td>
      <td>[../pdfs/Strine 2018.pdf, 8]</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11.0</td>
      <td>0.5710</td>
      <td>religion, religious, culture, cultural, christ...</td>
      <td>[../pdfs/Nnamani 2015.pdf, 14]</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12.0</td>
      <td>0.6930</td>
      <td>work, place, family, home, like, case, make, m...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Two. Frontiers ...</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13.0</td>
      <td>0.7600</td>
      <td>new, press, ed, york, power, study, global, pe...</td>
      <td>[../pdfs/Snyder 2018.pdf, 17]</td>
    </tr>
    <tr>
      <th>14</th>
      <td>14.0</td>
      <td>0.5200</td>
      <td>right, human, word, reality, state, world, tim...</td>
      <td>[../pdfs/Davidson 2018.pdf, 7]</td>
    </tr>
    <tr>
      <th>15</th>
      <td>15.0</td>
      <td>0.5200</td>
      <td>service, relationship, sense, work, good, espe...</td>
      <td>[../pdfs/Rowlands 2018.pdf, 0]</td>
    </tr>
    <tr>
      <th>16</th>
      <td>16.0</td>
      <td>0.5200</td>
      <td>theological, book, human, faith, case, g, life...</td>
      <td>[../pdfs/Thompson 2017.pdf, 0]</td>
    </tr>
    <tr>
      <th>17</th>
      <td>17.0</td>
      <td>0.5653</td>
      <td>research, form, study, mean, need, way, order,...</td>
      <td>[../pdfs/Frederiks and Nagy - 2016 - Religion,...</td>
    </tr>
    <tr>
      <th>18</th>
      <td>18.0</td>
      <td>0.6215</td>
      <td>woman, feminist, oppression, tulud, particular...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter Six. Expanding ...</td>
    </tr>
    <tr>
      <th>19</th>
      <td>19.0</td>
      <td>0.6800</td>
      <td>mission, world, international, dorottya, marth...</td>
      <td>[../pdfs/Frederiks and Nagy - 2016 - Religion,...</td>
    </tr>
    <tr>
      <th>20</th>
      <td>20.0</td>
      <td>0.6381</td>
      <td>struggle, woman, life, oppression, feminist, e...</td>
      <td>[../pdfs/cruz2010.pdf, 27]</td>
    </tr>
    <tr>
      <th>21</th>
      <td>21.0</td>
      <td>0.7600</td>
      <td>god, people, power, faith, christian, life, wo...</td>
      <td>[../pdfs/Thompson 2017.pdf, 3]</td>
    </tr>
    <tr>
      <th>22</th>
      <td>23.0</td>
      <td>0.7664</td>
      <td>hong, kong, tulud, filipina, filipino, g, huma...</td>
      <td>[../pdfs/Cruz - 2010 - An intercultural theolo...</td>
    </tr>
    <tr>
      <th>23</th>
      <td>24.0</td>
      <td>0.5721</td>
      <td>filipino, philippine, hk, migrant, tulud, just...</td>
      <td>[../pdfs/Cruz - 2010 - Chapter One. Geographie...</td>
    </tr>
  </tbody>
</table>
</div>



# Details of the Topic Model

One of the problems with topic modeling is that because it is an unsupervised clustering method, sometimes the computer sees connections that are not obvious, or at the vary least, are not _semantic_ clusters. Topic model is a blunt tool, but we picked six of these topics that we thought might be helpful in discovering books over the past 100 years that might build on the topic we had chosen. 

These topics are:

* _topic number_: 0
   * _heading_: *Black Experience*
   * _key terms_: 'black, experience, life, mean, like, make, point, american, challenge, relation'
* _topic number_: 1 
   * _heading_: *Context of Migrant Experience* 
   * _key terms_: 'identity, challenge, term, experience, context, question, migrant, people, state, dorottya'
*  _topic number_: 3
   * _heading_: *Communal Experience*
   * _key terms_: 'migrant, country, home, community, family, experience, life, economic, new, reality'
* _topic number_: 5
   * _heading_: *Social, Political, Economic Migrations*
   * _key terms_: 'social, political, economic, immigrant, society, cultural, perspective, issue, people, life'
* _topic number_: 6
   * _heading_: *Immigration and American Christianity*
   * _key terms_: 'church, christian, american, immigrant, community, role, dorottya, martha, state, faith'
* _topic number_: 11
   * _heading_: *Religion and Culture*
   * _key terms_: 'religion, religious, culture, cultural, christian, identity, faith, experience, example, time'
   
   
These topics were analysed in the context of the pdfs that generated them. These where the topics that we thought were both coherent, and might provide interesting analysis when looked at the political theology corpus generated from HathiTrust. 

These are the only six topics we looked for in the HathiTrust corpus that we had identified. 

