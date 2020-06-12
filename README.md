# spacy train example

```
# Set up the data. Downloads a small NER file and runs spacy convert.
./bin/get-data.sh

# Run the training on CPU. Use -g 0 for GPU 0. --help should give more usage.
python -m spacy train examples/fashion-ner/training.json examples/fashion-ner/eval.json examples/fashion-ner ner-cnn.cfg
```
