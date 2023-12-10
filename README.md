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
i am about to s ##cre ##am ma ##dly in the office / especially \t when they bring more papers to pi ##le higher on my des ##k . \n
```
You can download the raw dataset from [Wiki Dataset](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/) and put it under directory `data`. \
Then run `dataset/gen_data.py` to generate the dataset data, or you can use your own dataset.

> The `tokenization.py` is referenced from [BERT-Official](https://github.com/google-research/bert/tokenization.py)

## 3. Generate the vocab file
Run `dataset/WordVocab.py`

## 4. Pretrain your BERT
Run `main.py`


## Reference
[[BERT-pytorch]](https://github.com/codertimo/BERT-pytorch) 
[[BERT-Official]](https://github.com/google-research/bert)
