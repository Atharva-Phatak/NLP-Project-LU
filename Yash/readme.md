# Directory Sturcture 

1. #### _Notebooks_ directory contains all the notebooks.
    * Humour_Detection_Baseline_models.ipynb contains the baseline models for task 1a.
    * Humour_Detection_Roberta.ipynb contains the training and evaluation for roberta-base for task 1a.
    * Humour_Detection_Roberta(Data_Augmentation).ipynb contains the training and evaluation for roberta-base using data agumentation for task 1a.

---

# Enviornment Setup
* Clone the repository.
* Set current working direcotry as Yash - ```cd Yash ```
* open terminal and run 
```bash
pip install -r requirements.txt
pip install -r requirements_baseline.txt
```

---

# Running the models ( While loading the dataset provide the respective path of the dataset files)
* For baseline models run : ``` python humour_detection_baseline_models.py ```
* For Roberta-base run : ``` python humour_detection_roberta.py ```
* For Roberta-base with data augmentation : ``` python humour_detection_roberta(data_augmentation).py ```

----

# For loading and evaluating the results:
1 Download the weights of the model
   * For Roberta-base : https://drive.google.com/file/d/1tvLLybpewPerGqH3cGRSfZC6LcS8EYAd/view?usp=sharing
   * For Roberta-base with data augmentation : https://drive.google.com/file/d/1gkToE3duIjDlzZ6IavJhKKxgeLqzCYWz/view?usp=sharing
 
2 Run test_after_loading_model() function(change the path variable in the function to the model path)

---

## Model Results on Gold-test-data
|Model|Task-1a(F1-Score)|
|-----|-----------------|
|Roberta-base|0.950|
|Roberta-base-with-augmentation|0.945|
|Multinomial Naive Bayes|0.871|
|Support Vector Machine|0.850|