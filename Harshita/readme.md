## About this project

- This project contains Task 1 a) binary classification of humorous and non-humorous text 
- Following models are presented:
  1. LSTM
  2. GRU with GolVe Embeddings


## Manifest

- A list of the top-level files in this project with a description of what each file is.

```
- Datasets/                              ----> train/validation/gold test dataset in csv format 
    - dev.csv
    - gold-test
    - train.csv
    
- GRU+GloVe/                          ----> Folder for GRU with GloVe embeddings file
    - GRU+Glove.ipynb
    - gru+glove.py

- readme.md                           ----> This markdown file you are reading.

```

## Results

1. Vanilla LSTM
```
Hyperparameters:
------------------------------------------------------------
# Input Layer -> hidden layer (128) -> output layer (2)
optimiser = sgd
learning_rate= 0.01
epochs = 7
------------------------------------------------------------


Results:
------------------------------------------------------------
train loss:  0.0086
------------------------------------------------------------
val loss:  0.4903  F1 score:  92.04  Accuracy:  90.10
------------------------------------------------------------
test loss:  0.3768  F1 score:  93.02  Accuracy:  91.50
------------------------------------------------------------

```
3. GRU with Glove
```
Hyperparameters:
------------------------------------------------------------
Input Layer-> GloVe 300d embeddings -> GRU(100)->hidden layer (128) -> output layer (1)
optimiser= adam
learning_rate= 0.001
epochs = 10
------------------------------------------------------------


Results:
------------------------------------------------------------
train loss: 0.1122
------------------------------------------------------------
test loss:  0.543 F1 score:  0.862  Accuracy:  0.820
------------------------------------------------------------
gold loss:  0.484  F1 score:  0.876  Accuracy:  0.846
------------------------------------------------------------


```

- Colab notebooks contains link to Google Colab and can be executed with runtime GPU directly

1.Change the path of the dataset files
2.Run all cells


