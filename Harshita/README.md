## About this project

- This project contains Task 1 a) binary classification of humorous and non-humorous text 
- Following models are presented:
  1. Vanilla LSTM
  2. GRU with GolVe Embeddings


## Manifest

- A list of the top-level files in this project with a description of what each file is.

```
- Datasets/                        ----> train/test/gold dataset in csv format 
    - dev.csv
    - gold-test
    - train.csv
    
- GRU+GloVe/                       ----> Folder for GRU with GloVe embeddings file
    - GRU+Glove.ipynb
    - gru+glove.py

- readme.md                        ----> This markdown file you are reading.

```

## Results

1. Vanilla LSTM
```
Hyperparameters:
------------------------------------------------------------
# Input Layer ->GRU(100)-> hidden layer (128) -> output layer (1)
optimiser = sgd
learning_rate= 0.02
epochs = 7
------------------------------------------------------------


Results:
------------------------------------------------------------
train loss:  0.666
------------------------------------------------------------
test loss:  0.658  F1 score:  0.771  Accuracy:  0.632
------------------------------------------------------------
gold loss:  0.666  F1 score:  0.762  Accuracy:  0.615
------------------------------------------------------------

```
2. GRU with Glove
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
test loss:  0.543  F1 score:  0.862  Accuracy:  0.820
------------------------------------------------------------
gold loss:  0.484  F1 score:  0.876  Accuracy:  0.846
------------------------------------------------------------


```

- Colab notebooks contains link to Google Colab and can be executed with runtime GPU directly

1. Change the path of the dataset files
2. Execute cells


