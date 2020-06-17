# spacy train example

```
# Set up the data. Downloads a small NER file and runs spacy convert.
./bin/get-data.sh

# Run the training on CPU. Use -g 0 for GPU 0. --help should give more usage.
python -m spacy train examples/fashion-ner/fashion_brands_training.json examples/fashion-ner/fashion_brands_eval.json examples/fashion-ner/ner-cnn.cfg

python -m spacy train -g 0 -j 2 --strat allreduce examples/fashion-ner/fashion_brands_training.json examples/fashion-ner/fashion_brands_eval.json examples/fashion-ner/ner-cnn.cfg
```
