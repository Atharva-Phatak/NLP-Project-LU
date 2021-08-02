### This directory structure is as follows

1) ***Notebooks*** directory has all the final notebooks which are as follows
    * Bert-single-task.ipynb contains the training and evaluation for bert-small on task 1a
    * MultiTask_Roberta.ipynb contains training and evaluation for roberta-base trained in MultiTask fashion using task 1a,1b,1c.
2) ***Datasets*** directory has all the data used for training,validation and testing

***
### Follow these step to setup environment 

* Create a virtualenv by whatever means you are comfortable with.
* Clone this repo using git clone.
* cd into the Atharva directory using ```cd Atharva```
* open terminal and run ```pip install -r requirements.txt```
***
### To train the models use the below commands

* To train Bert-small(Single-Task) run : ```python train_singletask.py```
* To train roberta-base(Multi-Task) run : ```python train_multitask.py```
***
### For evaluation and reproduction of results you must download the finetuned weights of the model 
* Use this link to download models weights : https://drive.google.com/drive/folders/1J21PXTOAgQn3Ov4KEYSTQ-dJEguZxX7I?usp=sharing
* To evaluate models run the below command
  * ***For Single Task(Bert-Small)*** run : ```python evaluate.py --task stm --model_path path/to/saved/model-weights```
  * For Multi Task(Roberta-Base) run : ```python evaluate.py --task mtm --model_path path/to/saved/model-weights```

* ```stm``` : single-task , ```mtm``` : multitask
* The argument ```model_path``` expects full path to the **.bin** file.  
***
## Results Multi-Task setting on Gold-test-data
|Model|Task-1a(F1-Score)|Task-1b(RMSE)|Task-1c(F1-score)|
|-----|-----------------|-------------|-----------------|
|Roberta-base|0.9465|1.127|0.5326|
|Bert-small|0.9129| 1.113| 0.4918|


