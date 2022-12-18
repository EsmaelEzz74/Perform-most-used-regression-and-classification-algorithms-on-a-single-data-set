# Perform-most-used-regression-and-classification-algorithms-on-a-single-data-set
For a single data set I will use 8 different machine learning algorithms and compare the results
- <h5> Using a weather data set which contains 12 features for almost hundred thousand records I do the following :
- To examine the data set, I first check the column types if they are ready for preprocessing and duplicate rows and null values
- I found 24 duplicate rows to deal with and 500+ null values in a column. So I discarded them all to start using the dataset
- I needed some insight from the data set so I used  matplotlib.pyplot to plot a pie chart on the PRECIP TYPE column so I know  the data is not fully balanced
- <img src = "https://user-images.githubusercontent.com/85246622/208267595-527f625e-36b9-4daa-b6b7-0d5bdd5fb39d.png" width = "400" height = "400" />
---------------------------------------------
- Using the seaborn countplot, I get the most common values in the SUMMARY and the 20 most common values in the DAILY SUMMARY columns  
- <img src = "https://user-images.githubusercontent.com/85246622/208268026-3020baf5-0593-40f9-8cfc-d728abf9dcb5.png" width = "800" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/208268203-c398fe77-25c7-4646-877b-98b2b70abf07.png" width = "800" height = "400" />
---------------------------------------------
- Then to get the most correlated numerical columns, I used the seaborn heatmap as follows
- <img src = "https://user-images.githubusercontent.com/85246622/208268296-77bd1286-4266-421b-a46d-905ae76eff91.png" width = "400" height = "400" />
---------------------------------------------
- For the three categorical columns, I used Sklearn's LabelEncoder  to convert them to numeric values for easy handling in future models
<h1>------------------------------------------------------------------------------</h1>
<h2> Machine Learning Regression Algorithms </h2>
<h3> Single Linear Regression implementation : </h3> 

- First, I selected the only row to handle, which is TEMPERATURE next to the APPARENT TEMPERATURE columns as the label  
- Then, I split the data to train and test sets and apply data normalization by the StandardScaler from sklearn 
- Then initializing the model to fit the data then predict the test set and get the R2_score and for more explanation, i draw a scatter plot between the two columns and the model fitted line and print the statistical models for the Sklearn's single linear regression model
- <img src = "https://user-images.githubusercontent.com/85246622/208268990-d24c0a19-9043-41e8-9975-7b415303b667.png" width = "400" height="400" /> <img src ="https://user-images.githubusercontent.com/85246622/208268996-814caf1e-da63-4c82-9e62-ba6680e3e83d.png" width = "400" height="400" />
---------------------------------------------
How I coded my unique linear regression to compare the  algorithm above:
- I first coded a function that computes the gradient descent by first computing the result predicted  by the formula with theta-0 and theta-1 initialization, then computing the cost function, then the gradients and finally updating the two continues thetas up to the best values so they reach the minimum to break  this loop condition with their last values for later use
- In the linear regression function, I split the data to train and test sets and perform data normalization then fit the data and calculated the hypothesis to get the R2_score and draw the fitted line versus the scattered data
<img src = "https://user-images.githubusercontent.com/85246622/208269646-1597ab70-8f67-4ddf-92cd-34545b6a683a.png" width = "400" height="400" />
<h2>------------------------------------------------------------------------------</h2>
<h3> Multi Linear Regression implementation : </h3> 

- In the data selection stage, I select all columns except the date column and set the TEMPERATURE column as the label 
- For checking the outliers i used the boxplot then use the Z-score technique to remove them and almost remove about 2500 rows 
- <img src = "https://user-images.githubusercontent.com/85246622/208271924-47545468-8913-4c14-96c9-fef7c6d4c757.png" width = "800" height = "600" />
- <img src = "https://user-images.githubusercontent.com/85246622/208271928-799b33b7-4e71-4a6c-94d7-54b0c059ed81.png" width = "800" height = "600" />

