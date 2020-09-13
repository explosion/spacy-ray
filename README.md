<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spacy-ray: Parallel and distributed training with spaCy and Ray

> ⚠️ This repo is still a work in progress and requires the new **spaCy v3.0**.

[Ray](https://ray.io/) is a fast and simple framework for building and running **distributed applications**. This very lightweight extension package lets you use Ray for parallel and distributed training with [spaCy](https://spacy.io). If `spacy-ray` is installed in the same environment as spaCy, it will automatically add `spacy ray` commands to your spaCy CLI.

The main command is `spacy ray train` for
parallel and distributed training, but we expect to add `spacy ray pretrain` and `spacy ray parse` as well.

## 🚀 Quickstart

You can install `spacy-ray` from pip:

```bash
pip install spacy-ray
```

To check if the command has been registered successfully:

```bash
python -m spacy ray --help
```

Train a model using the same API as `spacy train`:

```bash
python -m spacy ray train config.cfg --n-workers 2
```
