# spacy ray command

This repo contains the work in progress on the Ray integration for spaCy v3. You'll be able to install this package:

```
pip install spacy-ray
```

Which will add the `spacy ray` commands to your spaCy CLI. The main command will be `spacy ray train` for 
parallel and distributed training, but we expect to add `spacy ray pretrain` and `spacy ray parse` as well.

All of the code for the integration will be in this package, we don't need to add anything to spaCy or Thinc.
