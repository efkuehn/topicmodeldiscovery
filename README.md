# topicmodeldiscovery
This is a space for notes on "Topic Modeling as a Tool for Resource Discovery", for the ATLA 75th anniversary volume.

[The Bibliography that @efkuehn *will* use is on
Zotero](https://www.zotero.org/groups/2198779/theologicopolitical/items)


## The Order of the notebooks 

1. [Creating A Topic Model](notebooks/create_topic_models.ipynb) 
2. [Visualising the Topic Model](notebooks/visualising_topic_model.ipynb)
3. [Exploring Topic Distribution](notebooks/exploring_topic_distribution.ipynb)
4. [Exploring Hathi Trust](notebooks/exploring_hathi_trust.ipynb)
5. [Analysing the Output](notebooks/graphs_and_plots_for_data.ipynb) 


## Running the Process 

The Jupyter Notebooks describe the different steps and stages that we went
through and developing the work. But most of the heavy compute was actually
run through some scripts that were called directly. 

## The Data 

The larger corpus that we wanted to explore is the [Political Theology
Corpus](https://babel.hathitrust.org/cgi/mb?a=listis&c=1154484) that we
created on HathiTrust. 

The IDs for this corpus can be found in the file
[data/newpathlist.txt](data/newpathlist.txt). On the system that we ran the
topic model, we ran the following rsync command to get a zipped local copy of
all of these files: 
```
rsync -av --no-relative --files-from newpathlist.txt data.analytics.hathitrust.org::features/ hathitrust/
``` 
The `--no-relative` flattened the files into the same directory which allowed
the script to run better. 


## The Model 

The version of the model that we ran could use a bit of tweaking and
correcting based on a better defined corpus, and changes to the paremters. But
as a proof of concept, [this is the model we ran](notebooks/models/PrelimTopicModel2).

## The Script 

We used [Anaconda](https://www.anaconda.com/) to install all of the
dependencies this script needed: 
```
conda install -c conda-forge nltk
conda install -c conda-forge htrc-feature-reader
conda install -c htrc htrc-feature-reader
conda install gensim
```

Also, two nltk datasets need to be downloaded:
```
nltk.download('wordnet')
nltk.download('punkt') 
```

The machine that we ran this on had 8 CPUs. If you run this on a different
machine, you may want to change the max number of workers. The current script
is set at 7. The script that runs over all of the downloaded HathiTrust data
is [pool_process.py](notebooks/pool_processing.py). 
