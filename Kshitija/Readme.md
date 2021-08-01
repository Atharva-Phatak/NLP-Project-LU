## About this project

- This project contains Task 1 a) binary classification of humorous and non-humorous text 
- Following models are presented:
  1. Bert-base with 2 layered MLP
  2. Bert-base with 2 layered LSTM


## Manifest

- A list of the top-level files in this project with a description of what each file is.

```
- data/                              ----> train/validation/gold test dataset in csv format 
    - dev.txt
    - train.csv
    - gold-test-27446.csv
- Notebooks/                          ----> Folder for all colab notebooks of following .py files
    - Notebooks/bert_lstm_main.ipynb
    - Notebooks/my_bert_main.ipynb
- bert_lstm_main.py                   -----> This file contains bert-base with 2 layered LSTM
- my_bert_main.py                    -----> This file contains bert-base with 2 layered MLP 
- requirements.txt                   ----> This contains all the dependencies to execute .py files
- README.md                           ----> This markdown file you are reading.

```

## Results

1. Bert-base with 2 layered MLP
```
Hyperparameters:
------------------------------------------------------------
# BASENET(768) -> hidden layer (256) -> output layer (2)
BASENET = 'bert-base-uncased'
learning_rate= 2e-5
epochs = 10
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
3. Bert-base with 2 layered LSTM
```
Hyperparameters:
------------------------------------------------------------
# BASENET(768) -> LSTM with internal dropout (256) -> hidden layer (128) -> output layer (2)
BASENET = 'bert-base-uncased'
learning_rate= 2e-5
epochs = 10
------------------------------------------------------------


Results:
------------------------------------------------------------
train loss:  0.0093
------------------------------------------------------------
 val loss:  0.4482  F1 score:  92.59  Accuracy:  90.70
------------------------------------------------------------
test loss:  0.3656  F1 score:  93.83  Accuracy:  92.50
------------------------------------------------------------


```

## Usage

- Clone repository 
```
git clone 'https://github.com/Atharva-Phatak/NLP-Project-LU.git' && cd NLP-Project-LU/Kshitija/

```
- Install Dependencies to run .py files

```
pip install -r requirements.txt

```

- An example to run .py file
```
python bert_lstm_main.py 

```

- Colab notebooks contains link to Google Colab and can be executed with runtime GPU directly

- Following changes are required to load model weights

1. Google drive link - [drive](https://drive.google.com/drive/folders/17e-Hu1aw9srRg1sIJoUuh-KiGtxHIoOn?usp=sharing)
2. Save at location Kshitija/my_model_weights/
3. Uncomment and execute 'LOAD SAVED MODEL' section of code 
4. execute 'EVALUATE GOLD TEST' section of code


