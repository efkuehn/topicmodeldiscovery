# topicmodeldiscovery
This is a space for notes on "Topic Modeling as a Tool for Resource Discovery", for the ATLA 75th anniversary volume.

[The Bibliography we used to create the topic model is linked here.](https://docs.google.com/document/d/1sXHkN6WsW_SwG5xLSRPLS-DPqbwbQCHcB8ErrIbxu0M/edit?usp=sharing)


## The Order of the notebooks 

1. [Creating A Topic Model](docs/creating_topic_models.md) 
2. [Visualising the Topic Model](docs/visualising_topic_model.md)
3. [Exploring Topic Distribution](docs/exploring_topic_distribution.md)
4. [Exploring Hathi Trust](docs/exploring_hathi_trust.md)
5. [Analysing the Output](docs/analysing_the_output.md) 


## Running the Process 

The Jupyter Notebooks describe the different steps and stages that we went
through and developing the work. But most of the heavy compute was actually
run through some scripts that were called directly. 

To recreate our work flow, either clone this repo, or download and unpack the
zip file. 

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
correcting based on a better defined corpus, and changes to the parameters. But
as a proof of concept, [the  model we ran is in the `notebooks/models`
directory](notebooks/models/). The model can be accessed as it is in many of
the linked notebooks: 
```
lda_model = LdaModel.load('./models/PrelimTOpicModel2') 
corpus_dict = Dictionary.load_from_text('./models/corpus_dictionary_2')
with open('./models/corpus.json', 'r') as fp:
    corpus = json.load(fp)
```


## The Script 


We used [Anaconda](https://www.anaconda.com/) to install all of the
dependencies this script needed: 
```
conda install -c conda-forge nltk
conda install -c conda-forge htrc-feature-reader
conda install -c htrc htrc-feature-reader
conda install gensim
```

Anaconda has some of the other libraries used like Pandas, Sci-kit learn, and
numpy. 

Also, two nltk datasets need to be downloaded:
```
nltk.download('wordnet')
nltk.download('punkt') 
```

The machine that we ran this on had 8 CPUs. If you run this on a different
machine, you may want to change the max number of workers. The current script
is set at 7. The script that runs over all of the downloaded HathiTrust data
is [pool_process.py](notebooks/pool_processing.py). 
