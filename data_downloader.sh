#!/bin/bash
# AGENDA dataset
mkdir -p data/AGENDA
wget https://github.com/rikdz/GraphWriter/raw/master/data/unprocessed.tar.gz -P data
tar xvzf data/unprocessed.tar.gz -C data/AGENDA
rm data/unprocessed.tar.gz
