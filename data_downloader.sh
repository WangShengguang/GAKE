#!/bin/bash
extract_dataset_from_targz () {
    mkdir -p data/$1
    tar xvzf data/$2 -C data/$1
}

mkdir data

# Update dataset from davidsbatista/Annotated-Semantic-Relationships-Datasets (deprecated)
# Update dataset from villmow/datasets_knowledge_embedding
git submodule update --init --recursive

# ============ Deprecated ============ #

# AGENDA dataset
# wget https://github.com/rikdz/GraphWriter/raw/master/data/unprocessed.tar.gz -P data
# extract_dataset_from_targz AGENDA unprocessed.tar.gz
# rm data/unprocessed.tar.gz

# SemEval 2007 Task 4
#extract_dataset_from_targz SemEval2007Task4 Annotated-Semantic-Relationships-Datasets/datasets/SemEval2007-Task4.tar.gz
#ls data/SemEval2007Task4/SemEval2007-Task4/*.tar.gz | xargs -n1 tar -C data/SemEval2007Task4 -xvzf 
# SemEval 2010 Task 8
# extract_dataset_from_targz SemEval2010Task8 Annotated-Semantic-Relationships-Datasets/datasets/SemEval2010_task8_all_data.tar.gz
# ReRelEM
# extract_dataset_from_targz ReRelEM Annotated-Semantic-Relationships-Datasets/datasets/ReRelEM.tar.gz
# BioNLP Shared Task
# extract_dataset_from_targz BioNLPSharedTask Annotated-Semantic-Relationships-Datasets/datasets/BioNLP.tar.gz

# DBpediaRelations-PT
# mkdir -p data/DBpediaRelations-PT
# bzip2 -dk data/Annotated-Semantic-Relationships-Datasets/datasets/DBpediaRelations-PT-0.2.txt.bz2
# mv data/Annotated-Semantic-Relationships-Datasets/datasets/DBpediaRelations-PT-0.2.txt data/DBpediaRelations-PT
