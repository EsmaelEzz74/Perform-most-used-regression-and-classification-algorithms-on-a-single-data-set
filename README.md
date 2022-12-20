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
<h1>----------------------------------------------------------------</h1>
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
---------------------------------------------
- To fit the data to  sklearn's multilinear regression, I split the data into training and test sets and apply data normalization by StandardScaler, then  fit the data to the model and predict the test set and print the R2_score and the statistical models for  Sklearn's multilinear regression model
- <img src = "https://user-images.githubusercontent.com/85246622/208651224-55f246ae-13b5-4ef0-aa3e-d39300dd9675.png" width ="400" height = "400" />
---------------------------------------------
- How I coded my unique multilinear regression to compare the  algorithm above:
- I coded a function that includes all the steps to build the model from scratch, split the data and apply data normalization, to then  calculate the gradient descent, I first predicted the predicted values and then the error calculated to calculate the cost function and  gradient descent and then finally updating the two continuous thetas  to the best values so they reach the minimum to break this loop condition with their latest values for later use like using the cost function list to plot the cost functions versus number of iterations 
- <img src = "https://user-images.githubusercontent.com/85246622/208652938-7148d3e2-d68f-4ded-8a37-fbfa459b875f.png" width ="400" height = "400" />
<h2>------------------------------------------------------------------------------</h2>
<h3> Polynomial Regression implementation : </h3> 

- For the data selection phase, I select data for linear polynomial regression and other data for multipolynomial regression, and a set of fixed Y-labels
- In linear polynomial  regression, I select the third degree for polynomial  features, then transform  data to split into training and test sets and apply data normalization by StandardScaler, then use a linear regression model to fit the data and predict the test set and print the R2_score
- <img src = "https://user-images.githubusercontent.com/85246622/208655184-b94c3c20-e07f-4a31-a175-049ae4d4b24f.png" width ="400" height = "400" />
---------------------------------------------
- In multipolynomial regression, I select the second degree for polynomial  features, then transform  data to split into training and test sets and apply data normalization by StandardScaler, then use a multiregression model to fit the data and predict the test set and print the R2_score
- <img src = "https://user-images.githubusercontent.com/85246622/208655504-99bb7a17-a274-4afd-84fe-11b5a5c77d0e.png" width ="400" height = "400" />
---------------------------------------------
- In my polynomial regression, I just implemented the linear model, So i first coded a function named create_features and return the data to its second degree for polynomial features  
- <img src = "https://user-images.githubusercontent.com/85246622/208655937-6fdbf0b1-4f28-4680-b160-219b2aa1de5c.png" width ="400" height = "400" />
- Then the normal_equ function to use the new data from the above function after i splited it to training and test sets and i applied the formula  <h1> (((X.T) X ) ** -1 ) X.T Y </h1> then predict the test set and print the R2_score 
- <img src = "https://user-images.githubusercontent.com/85246622/208656525-8830724f-3f2f-485c-9606-f837dc787d00.png" width ="400" height = "400" />
<h2>------------------------------------------------------------------------------</h2>
<h3> SVR Regression implementation : </h3> 

- For the three different kernels of the SVR model, I first split the data into training and test sets and apply data normalization by StandardScaler, then I initialize the three kernels with different names 
- <img src = "https://user-images.githubusercontent.com/85246622/208657562-13949f5b-6510-4fe0-86ec-f658098ebb5d.png" width ="400" height = "400" />
---------------------------------------------
- The SVR linear kernel and its R2_score
- <img src = "https://user-images.githubusercontent.com/85246622/208657751-8dd24b40-3301-47b7-a6be-5f8b91bf546b.png" width ="400" height = "400" />
---------------------------------------------
- The SVR rbf kernel and its R2_score
- <img src = "https://user-images.githubusercontent.com/85246622/208657775-e96fa438-eee0-4b19-9471-74f9ed23062d.png" width ="400" height = "400" />
---------------------------------------------
- The SVR poly kernel and its R2_score
- <img src = "https://user-images.githubusercontent.com/85246622/208657797-575b7341-566d-4d85-93fc-b430bf41cb48.png" width ="400" height = "400" />
<h1>----------------------------------------------------------------</h1>
<h2> Machine Learning Classification Algorithms </h2>
<h3> Logistic Regression implementation : </h3> 

- 




