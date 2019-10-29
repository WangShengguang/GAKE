# Graph Self-Attention Knowledge Embedding

## Instractions

Get raw data:

```sh
./data_downloader.sh
```

## Dataset

* Knowledge Graph
  * [AGENDA Dataset](https://github.com/rikdz/GraphWriter#agenda-dataset)
* Knowledge Graph with Entity Description
  * [xrb92/DKRL: Representation Learning of Knowledge Graphs with Entity Descriptions (AAAI'16)](https://github.com/xrb92/DKRL)
* Relation Extraction/Classification
  * [davidsbatista/Annotated-Semantic-Relationships-Datasets](https://github.com/davidsbatista/Annotated-Semantic-Relationships-Datasets)
    * SemEval 2007 Task 4
    * SemEval 2010 Task 8
    * ReRelEM
    * BioNLP Shared Task
    * DBpediaRelations-PT

### Knowledge Graph

#### AGENDA Dataset

> all-in-one json

* Train: 38720
* Valid: 1000
* Test: 1000

### Knowledge Graph with Entity Description

#### Freebase 2015

#### WordNet 2018

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
