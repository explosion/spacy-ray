#!/bin/bash

./bin/get-data.sh

TRAIN_DATA=examples/fashion-ner/fashion_brands_training.json
TEST_DATA=examples/fashion-ner/fashion_brands_eval.json
CFG=examples/fashion-ner/ner-cnn.cfg
# Run the training on CPU. Use -g 0 for GPU 0. --help should give more usage.
python -m spacy train -j 2 --strategy ps $TRAIN_DATA $TEST_DATA $CFG

python -m spacy train -j 3 --strategy ps $TRAIN_DATA $TEST_DATA $CFG

python -m spacy train -j 2 --strategy allreduce $TRAIN_DATA $TEST_DATA $CFG

python -m spacy train -j 3 --strategy allreduce $TRAIN_DATA $TEST_DATA $CFG


# if have GPU
python -m spacy train -j 2 -g 0 --strategy ps $TRAIN_DATA $TEST_DATA $CFG

python -m spacy train -j 3 -g 0 --strategy ps $TRAIN_DATA $TEST_DATA $CFG

python -m spacy train -j 2 -g 0 --strategy allreduce $TRAIN_DATA $TEST_DATA $CFG

python -m spacy train -j 3 -g 0 --strategy allreduce $TRAIN_DATA $TEST_DATA $CFG
