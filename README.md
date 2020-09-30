# Loan dataset prediciton
The main purpose of this repository is to focus more on code organization and test some additional functionalities, both related and not related to Machine Learning.<br> 
Hence we will use only a few columns for the predicting purposes. 
Planned functionalities:
- implementing CV (DONE)
- implementing hyperopt (DONE)
- logging (DONE)
- error and exception handling (partially)
- using dash to present data (+ as an iterface for model training) (in progress)
- using distributed computing

We will use the [lending-club dataset](https://www.kaggle.com/wordsforthewise/lending-club), specifically the **accepted** loans since for those we have the `loan_status` which will be the base for our target variable.
     
 

## Package installation

```commandline
git clone git@github.com:zbipio/loan-dataset.git
cd loan-dataset
python3.8 -m venv venv
source venv/bin/activate
./install.sh
```

## Run the app
```commandline
source venv/bin/activate
python main.py
```
