
![image](https://user-images.githubusercontent.com/76450761/118816778-d1e3f880-b8b2-11eb-8931-a831072c99e1.png)
![image](https://user-images.githubusercontent.com/76450761/118816921-f8a22f00-b8b2-11eb-8e7a-b0a7d45f3929.png)



In this project, we tackle the Kaggle Rossman challenge.
The goal is to predict the Sales of a given store on a given day.
Submissions are evaluated on the root mean square percentage error (RMSPE).

![image](https://user-images.githubusercontent.com/76450761/118817147-31420880-b8b3-11eb-9050-71c27473c5a2.png)

Please install the requirements.txt to reproduce the results:
pip install -r requirements.txt
It is recommended to use a new environment before running the pip install.

The dataset consists of three csv files: store.csv, train.csv and holdout.csv

Data Files:
- train.csv holds info about each store.
- store.csv holds the sales info per day for each store.
- holdout. csv holds "unseen" data that the model is going to be evaluated on


#### The repo contains main.py that runs the main script from step one until the end. 
#### The script can be run after cloning since all data used is in the repo.

Notes on main.py:

By default, the hyperparameter section is uncommented due to long completion time. 
The other sections will print out intermediate results for demonstration purposes on how the score looks like applying different models.
The script can be ran individually and the last print out will be the RMSPE for the predictions of the holdout set. 
This is the final evalution method for the challenge.

The function.py file contains utility functions that are called in main.py.

The rossman_model.sav contains the pickled hypertuned model with the least RMSPE.

The single steps of training the model are as follows:

1. Exploring data: EDA and visualization

2. Cleaning data:
   - drop data with no store
   - drop data with no DayOfWeek
   - drop data when store in NOT open
   - drop data where promo is NaN
   - drop SchoolHoliday data where promo is NaN
   - drop parameters that don't seem useful:   
          ('CompetitionOpenSinceMonth','CompetitionOpenSinceYear',/
          'Promo2SinceWeek', 'Promo2SinceYear', 'PromoInterval')
    - drop all rows with NaNs - Approximately 3% of rows
    - convert all the columns to int when necessary

3. Encoding:   
    - add Month as dummies
    - add a feature for scaled CompetitionDistance
    - convert DayOfWeek to dummies
    - convert StateHoliday to dummies
    - convert StoreType to dummies
    - convert Assortment to dummies


4. Looking at the correlations of the Sales with different parameters. Sales have significant correlation with:
      - Customers
      - DayOfWeek
      - StateHoliday
      - StoreType
      - Scaled CompetitionDistance
      etc.

5. Test / Train Split

6. Baseline Model and RandomForestRegression

7. Feature Selection and Engineering

8. Training data via Pipelines (RandomForestRegression, KNR and XGBoost Regressors)

9. Hyperparameter Tuning of the best model from step 8
