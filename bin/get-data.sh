#!/usr/bin/env bash
set -e

mkdir -p tmp
cd tmp
wget https://raw.githubusercontent.com/explosion/projects/master/ner-fashion-brands/fashion_brands_training.jsonl
wget https://raw.githubusercontent.com/explosion/projects/master/ner-fashion-brands/fashion_brands_eval.jsonl

cd ..

mkdir -p examples/fashion-ner2
python -m spacy convert tmp/fashion_brands_training.jsonl examples/fashion-ner --lang en
python -m spacy convert tmp/fashion_brands_eval.jsonl examples/fashion-ner --lang en
