# spacy train example

```
# Set up the data. Downloads a small NER file and runs spacy convert.
./bin/get-data.sh

# This doesn't quite work yet (we need to set up an entry-point to make
# the import), but once the packaging is done you'll be able to do:
pip install spacy-ray
spacy ray train ...
# We should add the following commands as well:
spacy ray pretrain ...
spacy ray parse ...
spacy ray evaluate ...
```

All of the code for the integration can stay in this package, we don't need to add anything to spaCy or Thinc.
