I was inspired to do this project by **[Nikolaos Christoforidis](https://www.datacamp.com/profile/NickChristoforidis) in the DataCamp Competition.**

1. BUSINESS QUESTION
- Business Problem / Baseline Definition

The purpose is to gain useful insight from the available data and train a model to predict the probability of a loan not being paid in full.

2. BUSINESS UNDERSTANDING
- Solution Planing  / Data Collection and CleaningExploratory Data Analysis / Data Preparation

The exploratory analysis will provide information and insights on the data. Tables and visualizations will be used to better understand the available data and answer relevant questions

The data will be used to predict the probability of a loan not being paid in full. 

Different models will be trained and the best-performing one will be selected.

Question that we will get the answers:

**QUESTION 1** - **Find out what kind of people take a loan for what purposes.**

**QUESTION 2: Predict the probability a user will be able to pay back their loan**

3. DATA PREPARATION 
- Exploratory Data Analysis / Data Preparation

Now i will show a lot of graphics to explore the data.

**THE NATURE OF LOAN**

![image](https://user-images.githubusercontent.com/72289622/140429424-2b165d10-41e9-4bee-934a-1d6b2b3d3ec4.png)


**INTEREST RATE BY LOAN PURPOSE**

![image](https://user-images.githubusercontent.com/72289622/140429449-57496813-856e-4511-bb54-973bfde8d733.png)


The graph shows that the purpose of a loan does not affect its interest rate much,
since we see the boxes at the same place for each purpose.
Loans for small businesses tend to have higher interest rates than the rest -> their average is higher than the 75th percentile.

**MONTHLY INSTALLMENT AMOUNT BY LOAN PURPOSE**

![image](https://user-images.githubusercontent.com/72289622/140429473-b08167e4-5e1d-4dbf-a65f-8d8e3762ab8d.png)

We can see that the installment amount changes related to the pupose of the loan, with loans for small businesses, debt consolidation & home improvement having the highest range of installements.

**FICO SCORE BY PAID STATUS**

![image](https://user-images.githubusercontent.com/72289622/140429490-ad6300ab-a217-47a1-bb73-d459a8b2865d.png)


It seems like people paying back their loans have higher FICO scores

Conclusion of this analysis:

People that do not meet credit criteria are more prone to not pay back a loan
The purpose of a loan is related to paid-back status
Loans for small businesses tend to have higher interest rates
Loans for small businesses and debt consolidation have the highest installements
Small business loans have $120 higher installments than other loans, on average
Log of income does not relate to fully-paid status
Dti does not greatly relate to fully-paid status
Higher FICO scores indicate fully-paid back loans

4. MACHINE LEARNING APPLICATION
- Training Algorithms / Performance Algorithms

Q1:

We have applied a several Algorithms.

KMeans - to cluster the data and define the types of customers.

From the cluster we divide the customers in 3 differents types: 

CLUSTER 0 - 'TRUSTHWORTHY' - Highest income in the set, also the highest fico and normal dti
CLUSTER 1 - 'DEBTORS' - Highest dti and average annual income and fico score
CLUSTER 2 - 'HIGH RISK' - Lowest income in the set, lowest dti and fico

![image](https://user-images.githubusercontent.com/72289622/140429513-0d4cf681-46fb-479c-88be-7d8885870228.png)

DEBTORS - take loans mostly to consolidate and fund theis credit card;
TRUSTWORTHY - expand their business or enjoy their good status with home
improvements and major purchase, almost twice as much as the other clusters;
HIGH RISK - taking loans for a mix purpose, but much more about educational
purposes.

**QUESTION 2: Predict the probability a user will be able to pay back their loan**

To resolve this questions we train four models for classification:

Logistic Regression
Random Forest Classifier
SVC
XGBClassifier

This model is imbalanced sets, so accuracy is not a good metric to consider a model. Because a model will always predict 'class 0' (fully paid loan), in our case this model will be correct 85% of the time.

So we used the standard score to this set is roc_auc.

We use the MinMaxScaler because ultimately the training set will include several binary variables.

To evaluete the best parameterswe use the GridSearchCV.

The best parameters:

```python
best_params_lr = {'C': 0.4, 'max_iter': 50, 'penalty': 'l1', 'solver': 'saga'}
best_params_rf =  {'max_depth': 6, 'max_leaf_nodes': 8, 'min_samples_leaf': 3}
best_params_svc = {'gamma': 'scale', 'kernel': 'linear', 'max_iter': -1}
best_params_xgb = {'colsample_bytree': 0.5, 'gamma': 0.6, 'learning_rate': 0.01, 
										'max_depth': 4, 'reg_lambda': 10, 'scale_pos_weight': 4, 
										'subsample': 0.7}
```

Evaluation the model:

The roc_auc:

![image](https://user-images.githubusercontent.com/72289622/140429541-ef93fd7e-1d3b-4594-a620-8dcd28d7d9f5.png)

The roc curve, the more area there is under a curve, the better the model perfoms; the area 

under the curve is the roc_auc score we used before. All models seem to perform equally well.

![image](https://user-images.githubusercontent.com/72289622/140429549-67263e66-cc2b-4819-8ec7-b3ea72abc79e.png)

The best model to predict the probability of a loan not being fully paid
seems to be: LogisticRegression Classifier, with marginally better
performance than SVC and Random Forest.

5. MODEL PUBLICATION
- Conversion to Business / Production Model

6. CONCLUSION

Finally, the performance of even the best model is not very good; this was
more or less expected from the beginning, when we saw absence of explanatory
strength for any features, no clear grouping, no correlation to the target
feature, etc.

But we get a roc_auc score of 0,71.
