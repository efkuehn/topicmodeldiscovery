
# Charts and Graphs for Data Analysis

The script `pool_process.py` runs on 7 different CPU cores, to spits out the text into `./output/analalyzed_corpus.json`. This script creates text file that needs a bit of massaging to create a python readable dictionary. Part of the goal of this script is to make a pandas dataframe. 


```python
# thanks to @ninjaaron for help pulling this script section together 
from collections import namedtuple
import json

encode = json.JSONEncoder(ensure_ascii=False).encode

Book = namedtuple("Book", "ht_id, top_topic, best_match, most_common_topic")
Topic = namedtuple("Topic", "top_num, perc")
BestMatch = namedtuple("BestMatch", "page, top_num, perc")

with open("./output/analyzed_corpus3.json") as fh:
    books = eval(fh.read())

analyzed_list = []
for book in books.values():
    out_dict = {}
    dct = book._asdict()
    out_dict['ht_id'] = dct['ht_id']
    for key, value in dct.items():
        try:
            for inner_key, inner_value in value._asdict().items():
                out_dict[key+'_'+inner_key] = inner_value    
            # dct[key] = value._asdict()
        except AttributeError:
            pass
    analyzed_list.append(out_dict)
```


```python
# This is what each line looks like after we open in it. 
analyzed_list[0]
```




    {'ht_id': 'mdp.39015011261867',
     'top_topic_top_num': 0,
     'top_topic_perc': 0.4302787482738495,
     'best_match_page': 227,
     'best_match_top_num': 6,
     'best_match_perc': 0.71999705,
     'most_common_topic_top_num': 6,
     'most_common_topic_perc': 70}



Next we create the dataframe we will use, and reorder the columns. 


```python
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
df = pd.DataFrame(analyzed_list)
columns = [
     'ht_id',
     'top_topic_top_num',
     'top_topic_perc',
     'best_match_page',
     'best_match_top_num',
     'best_match_perc',
     'most_common_topic_top_num',
     'most_common_topic_perc',
]
df = df[columns]
```

# Supplementing The Dataframe with Data about each record

The year publication date used for this, can easily be retrieved from the volume output. As it was, I reused the data from the paper on Political Theology that we presented at Atla's conference in Indianapolis. 

The titles and dates for each publication could be retrieved inside the `pool_process.py` script. The Subject headings used for this were gathered through OCLC's connection. 


```python
topic_num2name = {
    0: 'Black Experience',
    1: 'Context of Migrant Experience',
    3: 'Communal Experience',
    5: 'Social, Political, Economic Migrations',
    6: 'Immigration and American Christianity',
    11: 'Religion and Culture',
}
```


```python
import sqlite3
```


```python
conn = sqlite3.connect('../data/politheo.db')
cur = conn.cursor()
```


```python
def get_year(row):
    ht_id = row['ht_id']
    query = 'SELECT date FROM hathitrust_rec WHERE htitem_id = ?'
    cur.execute(query, (ht_id, ))
    year = cur.fetchone()
    year = int(year[0].split('-')[0])
    return year
    
```


```python
df['date'] = df.apply(get_year, axis=1)
df.columns
```




    Index(['ht_id', 'top_topic_top_num', 'top_topic_perc', 'best_match_page',
           'best_match_top_num', 'best_match_perc', 'most_common_topic_top_num',
           'most_common_topic_perc', 'date'],
          dtype='object')



Sorting the date by the decade. 


```python
df['decade'] = df.apply(lambda x: (x['date']//10)*10, axis=1)
```

## Number of Pages Dominated by a Particular Topic 

This chart shows how many pages are dominated by a particular topic over time. This is grouped by decade.  


```python
most_common_df = df.groupby(['most_common_topic_top_num', 'decade'])['most_common_topic_perc'].sum().unstack('most_common_topic_top_num')
```


```python
most_common_df.plot(kind='bar', stacked=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x11adba110>




![png](analysing_the_output_files/analysing_the_output_16_1.png)


This final part saves the script in a png file. 


```python
most_common_plot = most_common_df.plot(kind='bar', stacked=True).get_figure()
most_common_plot.savefig('./output/most_common_plot.png')
```


![png](analysing_the_output_files/analysing_the_output_18_0.png)


## Top Topic Averages 

The top_topic column is an average rating of how well the topics match a particular volume. This chart takes an average of those averages by decade to see if there are any decades which have particular highlights on which books where were. 


```python
top_topic_df = df.groupby(['top_topic_top_num', 'decade'])['top_topic_perc'].mean().unstack('top_topic_top_num')
```


```python
top_topic_df.plot(kind='bar', stacked=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x116381310>




![png](analysing_the_output_files/analysing_the_output_21_1.png)



```python
# This code saves the chart to a graph. 
top_topic_fig = top_topic_df.plot(kind='bar', stacked=True).get_figure()
top_topic_fig.savefig('./output/top_topic_fig.png')
```


![png](analysing_the_output_files/analysing_the_output_22_0.png)


## Best Match 

I tried a couple of approaches to find the best match. The Best match was calculated by finding the topic that best matched the model, and showing which page in the volume that came from. I thought it might be interesting to see which decade had the best, best match. But it turns out that this wasn't a particularly helpful measure. 


```python
best_match_df = df.groupby(['best_match_top_num', 'decade'])['best_match_perc'].max().unstack('best_match_top_num')
```


```python
best_match_df.plot(kind='bar', stacked=True)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x1163cb5d0>




![png](analysing_the_output_files/analysing_the_output_25_1.png)


Digging into the actual dataframe that produces the chart is a little more interesting. For the decades that only have one or two matches, it is because there are only one or two books that represent that decade. This did lead to the realization that we need a more percise way of looking at the best match data. 


```python
best_match_df
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
      <th>best_match_top_num</th>
      <th>0</th>
      <th>1</th>
      <th>3</th>
      <th>5</th>
      <th>6</th>
      <th>11</th>
    </tr>
    <tr>
      <th>decade</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.961600</td>
      <td>0.931428</td>
      <td>0.880000</td>
      <td>0.954286</td>
      <td>0.972571</td>
      <td>0.953334</td>
    </tr>
    <tr>
      <th>1740</th>
      <td>NaN</td>
      <td>0.759997</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1760</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.696676</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1780</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.808000</td>
    </tr>
    <tr>
      <th>1790</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.679999</td>
      <td>0.862856</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1810</th>
      <td>0.985231</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.906667</td>
      <td>0.893333</td>
      <td>0.839999</td>
    </tr>
    <tr>
      <th>1820</th>
      <td>0.951826</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.931428</td>
      <td>0.931428</td>
      <td>0.755001</td>
    </tr>
    <tr>
      <th>1830</th>
      <td>0.840000</td>
      <td>0.807999</td>
      <td>NaN</td>
      <td>0.893333</td>
      <td>0.833935</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1840</th>
      <td>0.842599</td>
      <td>0.807999</td>
      <td>NaN</td>
      <td>0.804002</td>
      <td>0.862857</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1850</th>
      <td>0.840000</td>
      <td>0.879999</td>
      <td>NaN</td>
      <td>0.687734</td>
      <td>0.917567</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1860</th>
      <td>0.679998</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.839999</td>
      <td>0.760000</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1870</th>
      <td>NaN</td>
      <td>0.807998</td>
      <td>0.807998</td>
      <td>0.879999</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1880</th>
      <td>0.840000</td>
      <td>0.839999</td>
      <td>NaN</td>
      <td>0.879999</td>
      <td>0.880000</td>
      <td>0.804000</td>
    </tr>
    <tr>
      <th>1890</th>
      <td>0.673333</td>
      <td>0.904000</td>
      <td>0.808000</td>
      <td>0.958261</td>
      <td>NaN</td>
      <td>0.804000</td>
    </tr>
    <tr>
      <th>1900</th>
      <td>0.760000</td>
      <td>0.880000</td>
      <td>NaN</td>
      <td>0.940000</td>
      <td>0.862856</td>
      <td>0.804000</td>
    </tr>
    <tr>
      <th>1910</th>
      <td>0.807999</td>
      <td>0.740874</td>
      <td>0.519999</td>
      <td>0.903999</td>
      <td>0.893333</td>
      <td>0.840000</td>
    </tr>
    <tr>
      <th>1920</th>
      <td>0.839999</td>
      <td>0.807999</td>
      <td>0.893333</td>
      <td>0.903999</td>
      <td>0.868341</td>
      <td>0.671111</td>
    </tr>
    <tr>
      <th>1930</th>
      <td>0.903999</td>
      <td>0.829560</td>
      <td>0.734630</td>
      <td>0.949473</td>
      <td>0.893333</td>
      <td>0.881154</td>
    </tr>
    <tr>
      <th>1940</th>
      <td>0.862857</td>
      <td>0.863058</td>
      <td>0.617596</td>
      <td>0.943529</td>
      <td>0.970402</td>
      <td>0.680000</td>
    </tr>
    <tr>
      <th>1950</th>
      <td>0.879999</td>
      <td>0.880000</td>
      <td>0.903999</td>
      <td>0.954286</td>
      <td>0.896471</td>
      <td>0.912727</td>
    </tr>
    <tr>
      <th>1960</th>
      <td>0.912727</td>
      <td>0.929932</td>
      <td>0.903999</td>
      <td>0.954286</td>
      <td>0.936000</td>
      <td>0.931429</td>
    </tr>
    <tr>
      <th>1970</th>
      <td>0.952000</td>
      <td>0.912727</td>
      <td>0.956364</td>
      <td>0.959167</td>
      <td>0.971765</td>
      <td>0.961600</td>
    </tr>
    <tr>
      <th>1980</th>
      <td>0.903999</td>
      <td>0.926154</td>
      <td>0.987027</td>
      <td>0.982545</td>
      <td>0.979575</td>
      <td>0.977674</td>
    </tr>
    <tr>
      <th>1990</th>
      <td>0.958261</td>
      <td>0.961600</td>
      <td>0.912727</td>
      <td>0.969032</td>
      <td>0.979130</td>
      <td>0.974737</td>
    </tr>
    <tr>
      <th>2000</th>
      <td>0.913540</td>
      <td>0.931428</td>
      <td>0.986426</td>
      <td>0.989327</td>
      <td>0.972000</td>
      <td>0.954286</td>
    </tr>
    <tr>
      <th>2010</th>
      <td>0.679998</td>
      <td>0.862856</td>
      <td>NaN</td>
      <td>0.880000</td>
      <td>NaN</td>
      <td>0.720004</td>
    </tr>
  </tbody>
</table>
</div>



### Best Match Spread Sheet 

I thought it would be worth digging into the data to see which books had high matches. As well as a high representative of that topic across the book. The first script creates a data frame where the most common topic and the best match topic are the same. This will help promote the aboutness of a particular work will be about the topic we are interseted in. 


```python
# The Top Five Pages for Each topic, Title, and Year

# df.groupby('best_match_top_num')['best_match_top_num', 'ht_id', 'best_match_page', 'best_match_perc'].head()
bmatch_df = df[df['most_common_topic_top_num'] == df['best_match_top_num']].sort_values(by=['best_match_top_num', 'best_match_perc'], ascending=False)
```


```python
# The colomuns of the topic are still all the ones we have
bmatch_df.columns
```




    Index(['ht_id', 'top_topic_top_num', 'top_topic_perc', 'best_match_page',
           'best_match_top_num', 'best_match_perc', 'most_common_topic_top_num',
           'most_common_topic_perc', 'date', 'decade'],
          dtype='object')



These two functions can add additional information from the Hathi Trust database created in the Political Theological project. The title could also be added to the dataframes when the entire corpus is run over the data. 


```python
def find_title(row):
    query = 'SELECT title FROM hathitrust_rec WHERE htitem_id = ?'
    cur.execute(query, (row['ht_id'], ))
    title = cur.fetchone()
    return title[0]

def find_subjects(row):
    query = 'SELECT subject_heading FROM htitem2subjhead WHERE htitem_id = ?'
    cur.execute(query, (row['ht_id'], ))
    subjects = cur.fetchall()
    return ' | '.join([x[0] for x in subjects])
```


```python
# apply find_title to the dataframe
bmatch_df['title'] = bmatch_df.apply(find_title, axis=1)
```


```python
# add the topic name to the dataframe
bmatch_df['top_nam'] = bmatch_df.apply(lambda x: topic_num2name[x['best_match_top_num']], axis=1)
```


```python
# add the subjects to the dataframe. 
# Subjects likewise are available in the hathitrust record reader
bmatch_df['subjects'] = bmatch_df.apply(find_subjects, axis=1)
```


```python
# This cell reorders the columns in a more intuitive order
bmatch_df = bmatch_df[
    ['top_nam',
     'best_match_top_num',
     'ht_id', 
     'title',
     'date',
     'subjects',
     'best_match_page',                     
     'best_match_perc',      
     'most_common_topic_top_num', 
     'most_common_topic_perc'
    ]
].reset_index(drop=True)
```


```python
bmatch_df.head(20)
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
      <th>top_nam</th>
      <th>best_match_top_num</th>
      <th>ht_id</th>
      <th>title</th>
      <th>date</th>
      <th>subjects</th>
      <th>best_match_page</th>
      <th>best_match_perc</th>
      <th>most_common_topic_top_num</th>
      <th>most_common_topic_perc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>uva.x030453390</td>
      <td>Bulletin signalétique. 527, Histoire et scien...</td>
      <td>1984</td>
      <td></td>
      <td>926</td>
      <td>0.977674</td>
      <td>11</td>
      <td>453</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>mdp.39015064409652</td>
      <td>Francis bulletin signalétique. 527, Histoire e...</td>
      <td>1994</td>
      <td></td>
      <td>494</td>
      <td>0.974737</td>
      <td>11</td>
      <td>206</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>mdp.39015079907781</td>
      <td>Bulletin signalétique. 527: Sciences religieus...</td>
      <td>1970</td>
      <td></td>
      <td>78</td>
      <td>0.961600</td>
      <td>11</td>
      <td>101</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>uva.x030785791</td>
      <td>Theologische revue. Jahrg.70-72 1974-76</td>
      <td>1976</td>
      <td></td>
      <td>216</td>
      <td>0.960000</td>
      <td>11</td>
      <td>172</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>inu.30000087051045</td>
      <td>Das Bild des Menschen in den Wissenschaften / ...</td>
      <td>2002</td>
      <td>Philosophical anthropology | Human beings | Bo...</td>
      <td>66</td>
      <td>0.954286</td>
      <td>11</td>
      <td>43</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>mdp.39015062902328</td>
      <td>Die Säkularisation im Prozess der Säkularisier...</td>
      <td>2005</td>
      <td>SaÌˆkularisierung.$0(DE-588)4051238-1$2gnd | S...</td>
      <td>446</td>
      <td>0.954286</td>
      <td>11</td>
      <td>174</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>mdp.39015063320546</td>
      <td>New approaches to the study of religion / edit...</td>
      <td>0</td>
      <td></td>
      <td>501</td>
      <td>0.953334</td>
      <td>11</td>
      <td>166</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>uva.x006167926</td>
      <td>Communio viatorum. v.43-44 2001-2002</td>
      <td>2002</td>
      <td>Theology--Periodicals | Theologie | Theology</td>
      <td>563</td>
      <td>0.949474</td>
      <td>11</td>
      <td>64</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>uc1.b3365232</td>
      <td>Transzendenz und Immanenz : Philosophie und Th...</td>
      <td>1977</td>
      <td>Philosophie et theÌologie--CongreÌ€s | Foi et...</td>
      <td>92</td>
      <td>0.949474</td>
      <td>11</td>
      <td>132</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>ien.35556034292441</td>
      <td>Politik und Politeia : Formen und Probleme pol...</td>
      <td>2000</td>
      <td>Political sociology | Bibliografie.$0(DE-588)4...</td>
      <td>116</td>
      <td>0.949474</td>
      <td>11</td>
      <td>78</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>mdp.39015021492742</td>
      <td>Troeltsch-Studien / herausgegeben von Horst Re...</td>
      <td>0</td>
      <td>Theologians--Germany--Biography | Theologians</td>
      <td>23</td>
      <td>0.946667</td>
      <td>11</td>
      <td>152</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>mdp.39015059876949</td>
      <td>Deutsche Nationalbibliographie und Bibliograph...</td>
      <td>1995</td>
      <td>Bibliography, National.$0(DNLM)D001637 | Tijds...</td>
      <td>755</td>
      <td>0.946667</td>
      <td>11</td>
      <td>83</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>mdp.39015078401489</td>
      <td>Beiträge zur Geschichte des Bistums Regensburg...</td>
      <td>1983</td>
      <td>Zeitschrift.$2swd | Geschichte.$2swd</td>
      <td>282</td>
      <td>0.943529</td>
      <td>11</td>
      <td>123</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>uva.x000043098</td>
      <td>Civilisation noire et Église catholique : col...</td>
      <td>1978</td>
      <td>Blacks--Africa--Congresses | Noirs--Afrique--C...</td>
      <td>125</td>
      <td>0.943529</td>
      <td>11</td>
      <td>111</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>uva.x001488980</td>
      <td>Civil Religion : die religiöse Dimension der ...</td>
      <td>1987</td>
      <td>Begriff.$2swd | Civil religion | Zivilreligion...</td>
      <td>11</td>
      <td>0.940000</td>
      <td>11</td>
      <td>277</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>mdp.39015051374331</td>
      <td>Die Flucht in den Begriff : Materialien zu Heg...</td>
      <td>1982</td>
      <td></td>
      <td>358</td>
      <td>0.936000</td>
      <td>11</td>
      <td>211</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>mdp.39015053573773</td>
      <td>Der Mensch als Bild Gottes.</td>
      <td>1969</td>
      <td>Noise-prevention and control | Homme (theÌolo...</td>
      <td>332</td>
      <td>0.931429</td>
      <td>11</td>
      <td>39</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>uva.x006090923</td>
      <td>Post-theism : reframing the Judeo-Christian tr...</td>
      <td>2000</td>
      <td>Judaism--Relations--Christianity | Christianit...</td>
      <td>102</td>
      <td>0.926154</td>
      <td>11</td>
      <td>122</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>mdp.39015066107940</td>
      <td>Göttinger Miszellen. no.198-203(2004)</td>
      <td>2004</td>
      <td>Antiquities | Civilization | Egyptologie</td>
      <td>501</td>
      <td>0.926154</td>
      <td>11</td>
      <td>49</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Religion and Culture</td>
      <td>11</td>
      <td>mdp.39015063166550</td>
      <td>Religion und Politik : zu Theorie und Praxis d...</td>
      <td>2004</td>
      <td>Politische Theologie.$0(DE-588)4046562-7$2gnd ...</td>
      <td>368</td>
      <td>0.926154</td>
      <td>11</td>
      <td>187</td>
    </tr>
  </tbody>
</table>
</div>




```python
# This cell saves the dataframe to a 
bmatch_df.to_excel('./output/best_match.xlsx', index=False)
```


```python

```
