# Graph Context Attention Knowledge Embedding

## Instractions

Install requirements

```sh
pip3 install -r requirements.txt
```

Get data

```sh
git lfs pull
```

Run

```sh
python3 manage.py --model [GCAKE] --dataset [FB15K-237, WN18RR]
./train_all_dataset.sh [GCAKE]
```

> Get raw data:
>
> ```sh
> ./data_downloader.sh
> ```
>
> Get trainable data
>
> ```sh
> ./data_preprocessing.sh
> ```

## Dataset

* Knowledge Graph with Entity Description
  * [xrb92/DKRL: Representation Learning of Knowledge Graphs with Entity Descriptions (AAAI'16)](https://github.com/xrb92/DKRL) - FB15k with description, FB20k-new
  * [villmow/datasets_knowledge_embedding: Datasets for Knowledge Graph Completion with textual information about the entities](https://github.com/villmow/datasets_knowledge_embedding) - FB15K, FB15k-237, WN18, WN18RR
* Knowledge Graph
  * [AGENDA Dataset](https://github.com/rikdz/GraphWriter#agenda-dataset)
* Relation Extraction/Classification
  * [davidsbatista/Annotated-Semantic-Relationships-Datasets](https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets)
    * SemEval 2007 Task 4
    * SemEval 2010 Task 8
    * ReRelEM
    * BioNLP Shared Task
    * DBpediaRelations-PT

### Knowledge Graph with Entity Description

> [想问一下WN18 和 WN18RR是什么关系 · Issue #76 · thunlp/OpenKE](https://github.com/thunlp/OpenKE/issues/76)

* [Papers With Code : Link Prediction](https://paperswithcode.com/task/link-prediction)
* [Relation Prediction | NLP-progress](http://nlpprogress.com/english/relation_prediction.html)

> `head\trelation\ttail\t[cls] head_description [sep] relation_text [sep] tail_description [sep]\n`

#### Freebase

`data/FB15K-237/*.tsv`

* [State-of-the-art table for Link Prediction on FB15k](https://paperswithcode.com/sota/link-prediction-on-fb15k)
* [State-of-the-art table for Link Prediction on FB15k-237](https://paperswithcode.com/sota/link-prediction-on-fb15k-237)

#### WordNet

`data/WN18RR/*.tsv`

* [State-of-the-art table for Link Prediction on WN18](https://paperswithcode.com/sota/link-prediction-on-wn18)
* [State-of-the-art table for Link Prediction on WN18RR](https://paperswithcode.com/sota/link-prediction-on-wn18rr)

---

### Knowledge Graph

#### AGENDA Dataset

> all-in-one json

* Train: 38720
* Valid: 1000
* Test: 1000

### Relation Extraction/Classification

#### SemEval 2007 Task 4

> each relation (total 7) a file

* Train (0-140): 140 x 7
* Test (141-?)
  * relation 1: 220
  * relation 2: 218
  * relation 3: 233
  * relation 4: 221
  * relation 5: 211
  * relation 6: 212
  * relation 7: 214

#### SemEval 2010 Task 8

> all-in-one txt

* Train (1-8000): 8000
* Test (8001-10717): 2717

#### ReRelEM

> all-in-one xml

* Total
  * Element 'DOC': 129

#### BioNLP Shared Task

> each sample three file (rel, a1, txt)

* Total: 798

#### DBpediaRelations-PT

> all-in-one txt

`grep -o 'SENTENCE' data/DBpediaRelations-PT/DBpediaRelations-PT-0.2.txt | wc -l`

* Total: 98023

## Coding Stytle

* [styleguide | Style guides for Google-originated open-source projects](https://google.github.io/styleguide/pyguide.html)
* [google/seq2seq pylintrc](https://github.com/google/seq2seq/blob/master/pylintrc)
