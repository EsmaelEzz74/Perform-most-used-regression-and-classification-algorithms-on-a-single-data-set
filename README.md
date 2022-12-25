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
<h3> SVM Regression implementation : </h3> 

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

- In the classification problem, the labeled column is a binary value column consisting of only two values, like the Precip Type column, and the training data are all other columns
- Using sklearn's train_test_split, I split the data into training and test sets, then used StandardScaler to perform the data normalization, and then used the logistic regression model to fit the training data to predict the test set and print the R2_score 
- From the Sklearn metrics I used first the confusion matrix to print  the model's confusion matrix and using the Seaborn heatmap I got this chart and with classification_report i print the report showing the precision, recall, and f1-score
- <img src = "https://user-images.githubusercontent.com/85246622/209470687-86056203-2d03-4f36-b85a-b63b50cdbe1b.png" width = "400" height = "400" />- <img src = "https://user-images.githubusercontent.com/85246622/209470826-ae823799-f3fc-4a6a-8859-82721501802e.png" width = "400" height = "400" />
---------------------------------------------
- When coding the logistic regression, I'm writing a function that computes the sigmoid, the cost function, and a function to predict the test set and puts them all into one function, so after splitting the data and I used the computed sigmoid and the cost function  to calculate the gradient to update thetas to reach the minimum value and then print out the R2_score,  confusion matrix and  classification report
- <img src = "https://user-images.githubusercontent.com/85246622/209471443-cfabe662-ebdc-443e-8e27-d6a0cb11f7d4.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209471450-3f539328-dd5f-482b-a792-d34cfa473d9a.png" width = "400" height = "400" />
<h2>------------------------------------------------------------------------------</h2>
<h3> SVM Classifier implementation : </h3> 

- I split the data into training and test sets and used the StandardScaler to perform data normalization
- In the LinearSVC function, I made a list of C containing four numbers for C and for each C, fitted the training data to the model with a fixed tolerance number, and predicted the test set to get the score and then print out a classification report for each C and a compilation table containing the four merging matrices for the four C numbers
- <img src = "https://user-images.githubusercontent.com/85246622/209472440-5413d8d6-fa99-47b8-8d1b-c227931ea74a.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209472359-e00053b8-b19c-4a61-bcc8-bbea82f092d6.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209472379-2830b914-09b5-4cec-b857-7967ef3e8b76.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209472385-b759ad20-a8d3-4d86-986d-4488b904ee81.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209472404-dc959f14-77c5-4bb8-b389-797ef48c0965.png" width = "800" height = "800" /> 
---------------------------------------------
- For a general purpose function to compare between SVC kernels, I used the splitted training abd test sets and for a list of C, for each C I fitted the training set to the model and predict the test set to print the score then print out the classification report and a compilation table containing the four merging matrices for the four C numbers
- For the SVC RBF kernel
- <img src = "https://user-images.githubusercontent.com/85246622/209474010-596f7f49-ac05-452f-8431-5e772606c437.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474014-775b1152-ec03-4614-a4ba-c81e15bb38e1.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474016-e34b9180-e997-4451-945a-d1ebca1c3109.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474018-f1118f03-965b-48a4-a3e5-92c45eedf3b4.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474020-ac68512e-02c9-46a3-bfa0-5e67fab6ec12.png" width = "800" height = "800" />
---------------------------------------------
- For the SVC Linear kernel
- <img src = "https://user-images.githubusercontent.com/85246622/209474154-cd8ff6e0-4902-4257-8787-47b86a5c9b93.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474158-cbc3ee28-d55b-444f-80c3-5edf65261ac6.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474162-d75041ac-2357-4990-b3fb-913ad0638006.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474164-62467d91-df31-49d7-9d02-ba8808431d43.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474168-d2d32096-c64e-4acc-b73e-156b714ed959.png" width = "800" height = "800" />
---------------------------------------------
- For the SVC Poly kernel
- <img src = "https://user-images.githubusercontent.com/85246622/209474172-e311fa40-1744-4bff-ae96-3f417f6c0cc4.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474175-32a1b9ca-122d-4ca6-83ff-f130f6136d5e.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474176-ba3528f4-72fe-4526-bb08-c350d2a84cdf.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474180-25b2c0dd-25ab-4177-9382-baa9df7cbdf7.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474183-32199c37-5f90-456c-ba7c-36d136941aeb.png" width = "800" height = "800" />
<h2>------------------------------------------------------------------------------</h2>
<h3> KNN Classifier implementation : </h3> 

- In the KNN function there are two variables that can be changed as nearest neighbors, so I put a list  of four different numbers in the neighbors and  model weights, which is a function parameter
- Using train_test_split I split the data into test and training  sets and used  StandardScaler to do the data normalization
- For each number in the  list of neighbors, I initialize the model with that neighbor and the weight parameter, then fit the training data to the model and predict the test set to print the score and  classification report and a build table for the four mixing matrices for the four K numbers
- For the UNIFORM weights parameter
- <img src = "https://user-images.githubusercontent.com/85246622/209474811-06cb4e28-e754-4587-8fed-cd255f94b988.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474812-04146882-27f4-4bd8-b5a8-966b97eea807.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474814-1df92d22-e9e5-47de-bd6e-001d32e41c6c.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474817-a91c1fb5-50df-4949-8633-872fe9faa073.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474820-3745e2a0-4e78-4f1e-8679-cce7a7113bbb.png" width = "800" height = "800" />
---------------------------------------------
- For the DISTANCE weights parameter
- <img src = "https://user-images.githubusercontent.com/85246622/209474823-d10d0442-7be1-4fdd-aafe-47ffa98f8fb1.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474832-38ae5494-abfe-4f09-b30c-4cad36a57c7f.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474839-c503d55d-d61b-4788-ac69-2c5e3d4a072b.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474845-1826cc30-4eda-4642-b285-f123194f4d71.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474849-b833953f-d948-44b8-978c-d6d94463f0c4.png" width = "800" height = "800" />
<h2>------------------------------------------------------------------------------</h2>
<h3> Decision Tree Classifier implementation : </h3> 

- There are 3 modifiable variables in the decision tree, like criterion, splitter and maximum depth, so these are my function parameters
- Using train_test_split I split the data into test and training  sets and used  StandardScaler to do the data normalization
- To initialize the model, I used all three parameters to fit the training set, then predicted the test set and printed the score and classification report and plotted the confusion matrix as a seaborn heatmap
- For a GINI criterion and BEST as splitter with maximum depth of 7 
- <img src = "https://user-images.githubusercontent.com/85246622/209475337-bffeeced-fa39-4429-940e-7ba9348d5a54.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209475343-7898e66a-2676-40dd-8db5-fb05188342e5.png" width = "400" height = "400" />
---------------------------------------------
- For a GINI criterion and BEST as splitter with maximum depth of 5 
- <img src = "https://user-images.githubusercontent.com/85246622/209475446-509b3933-66ce-4aaa-bb37-60f8aefdcde9.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209475362-be8088f3-f2fe-4564-aecf-86da0fcb14b4.png" width = "400" height = "400" />
---------------------------------------------
- For a GINI criterion and RANDOM as splitter with maximum depth of 7 
- <img src = "https://user-images.githubusercontent.com/85246622/209475375-5e072904-ce42-4b6b-8212-947cc72396aa.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209475381-d167488a-2ef0-4b1e-adc2-8c122051c48b.png" width = "400" height = "400" />
---------------------------------------------
- For a GINI criterion and RANDOM as splitter with maximum depth of 5
- <img src = "https://user-images.githubusercontent.com/85246622/209475388-c4df705b-873b-4b2f-9b48-db1e41c1757e.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209475393-4b6ab334-b355-423a-b5e4-16324e978413.png" width = "400" height = "400" />
---------------------------------------------
- For a GINI criterion and RANDOM as splitter with maximum depth of NONE 
- <img src = "https://user-images.githubusercontent.com/85246622/209475403-bb4bb7c3-d0e7-4613-879b-1ac817113d52.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209475404-bd4f83d8-10f7-43f8-b277-fa21a39dfc20.png" width = "400" height = "400" />
---------------------------------------------
- For a ENTROPY criterion and BEST as splitter with maximum depth of 7
- <img src = "https://user-images.githubusercontent.com/85246622/209476257-b8159f1b-4113-4d3c-98de-8e07e30065f0.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209476201-60dfd0cc-859c-44b0-9466-f26f2b8c1962.png" width = "400" height = "400" />
---------------------------------------------
- For a ENTROPY criterion and BEST as splitter with maximum depth of 5 
- <img src = "https://user-images.githubusercontent.com/85246622/209476167-61cecd46-3b8c-4e9d-bcc8-ad6ec4369e11.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209476171-49545e3a-92a4-4d6f-95b2-6fe8537ea9ab.png" width = "400" height = "400" />
---------------------------------------------
- For a ENTROPY criterion and RANDOM as splitter with maximum depth of 7 
- <img src = "https://user-images.githubusercontent.com/85246622/209476158-5663d171-a74d-4209-9e5b-3b986f591edf.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209476161-8cd45ab9-6f6b-498b-b28f-e760d776035a.png" width = "400" height = "400" />
---------------------------------------------
- For a ENTROPY criterion and RANDOM as splitter with maximum depth of 5 
- <img src = "https://user-images.githubusercontent.com/85246622/209476151-24094e35-ac4e-45d2-8a95-fd0593c27d83.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209476153-ecfe3272-a099-45b0-b55f-5d7944d7e2bb.png" width = "400" height = "400" />
---------------------------------------------
- For a ENTROPY criterion and RANDOM as splitter with maximum depth of NONE 
- <img src = "https://user-images.githubusercontent.com/85246622/209476140-842ec623-fbe1-4624-8a22-279cf5aaa3bc.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209476142-b88b8540-4d97-447e-a97d-9b67d01601b5.png" width = "400" height = "400" />
