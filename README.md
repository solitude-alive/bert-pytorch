# BERT implementation with pytorch

## 1. Install the environment
Install the environment from `environment.yml`
```commandline
conda env create -f environment.yml
```
Then active your environment.

## 2.Prepare dataset
The dataset should be like
```text
This is an \t example in dataset.\n
```
You can download the dataset from [Wiki Dataset](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/) and put it under directory `data`.

Then run `dataset/gen_data.py`

## 3. Generate the vocab file
Run `dataset/WordVocab.py`

## 4. Pretrain your BERT
Run `main.py`


## Reference
[BERT-pytorch](https://github.com/codertimo/BERT-pytorch)
