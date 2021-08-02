### ***The directory structure is as follows***

1) ***Notebooks*** directory has all the final notebooks which are as follows
    * Lakehead_NLP_Humor_Detection_GloVe_LSTM.ipynb contains the training and evaluation code for training LSTM + GloVe semval data 
    * NLP_Humor_Detection_Augemented_Data.ipynb contains training and evaluation code for training LSTM + GloVe on Augemented Data
   
2) ***Dataset*** directory has all the data used for training, augemented, validation and testing

***
### Follow these step to setup environment 

* Create a virtualenv by whatever means you are comfortable with.
* Clone this repo using git clone.
* cd into the Girijesh directory using ```cd Girijesh```
* open terminal and run ```pip install -r requirements.txt```

***
### Change path of following before running code

* Change path for all files
* Download glove model from https://www.kaggle.com/thanakomsn/glove6b300dtxt and change path of variable ```glove_path``` to load glove model
* cd into the Girijesh directory using ```cd Girijesh```
* open terminal and run ```pip install -r requirements.txt```
***
### To train the models use the below commands

* To train LSTM with Glove run : ```python lakehead_nlp_humor_detection_glove_lstm.py```
* To train LSTM with Glove on augmented data run : ```python nlp_humor_detection_augemented_data.py```
***



