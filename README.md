# adv-def-text-dist

## Requirements: 

- Python 3.6+ 
- TensorFlow 1.3+
- spacy
- transformers
- tensorflow_hub

Please refer to requirement.txt for detailed packages

## Introduction

This is an implementation for the paper: Grey-box Adversarial Attack and Defence for Text. 
The paper is currently submitted to EMNLP'20. 

## Run training and test

1, Please download the Yelp review dataset from the official website [link](https://www.yelp.com/dataset). 

2, Download the GloVe embeddings and the counter-fitted embeddings using 

```
./download.sh.
```

3, Run dataset preprecessing using 

```
python yelp_preprocessing.py --data_dir YELP_DATASET_PATH --embed_file GLOVE_EMB_PATH
```

4, Train target models using the scripts 
 
```
./train_cls.sh
``` 

5, Pre-train Auto-encoder for reconstruction using the scripts

```
./train_ae.sh.
```

6, Train adversarial attack/defence models using the scripts (multiple variants of our model are available and commented out in the script)

```
./train_adv.sh
```

7, Perform independent test for adversarial attack/defence using 

```
./test_adv.sh
```
