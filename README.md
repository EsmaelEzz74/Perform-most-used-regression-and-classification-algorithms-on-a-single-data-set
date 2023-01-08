# Exploring the Power of Regression and Classification on Predicting the Weather
- <h4>The data set being used contains 12 features and almost hundred thousand records : 
- <h4>The data set was checked for column types, duplicate rows, and null values. Some duplicate rows and null values were found and removed.
-	<h4>The data set was explored using visualizations like pie charts and countplots. 
- <h4>The most correlated numerical columns were identified using a heatmap.
- <h4>Categorical columns were converted to numeric values using LabelEncoder.
- <img src = "https://user-images.githubusercontent.com/85246622/208267595-527f625e-36b9-4daa-b6b7-0d5bdd5fb39d.png" width = "400" height = "400" />  <img src = "https://user-images.githubusercontent.com/85246622/208268296-77bd1286-4266-421b-a46d-905ae76eff91.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/208268026-3020baf5-0593-40f9-8cfc-d728abf9dcb5.png" width = "800" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/208268203-c398fe77-25c7-4646-877b-98b2b70abf07.png" width = "800" height = "400" />
---------------------------------------------
<h1>----------------------------------------------------------------</h1>
<h2> Machine Learning Regression Algorithms </h2>
<h3> Single Linear Regression implementation : </h3> 

- <h4> Single linear regression was implemented using sklearn's single linear regression model and a custom implementation. The data was split into training and test sets, normalized, and fit to the model. The model's performance was evaluated using the R2 score and visualized with a scatter plot and fitted line.
- <img src = "https://user-images.githubusercontent.com/85246622/208268990-d24c0a19-9043-41e8-9975-7b415303b667.png" width = "400" height="400" /> <img src ="https://user-images.githubusercontent.com/85246622/208268996-814caf1e-da63-4c82-9e62-ba6680e3e83d.png" width = "400" height="400" />
---------------------------------------------
<img src = "https://user-images.githubusercontent.com/85246622/208269646-1597ab70-8f67-4ddf-92cd-34545b6a683a.png" width = "400" height="400" />
<h2>------------------------------------------------------------------------------</h2>
<h3> Multi Linear Regression implementation : </h3> 

-	<h4>Outliers were identified and removed in the data used for multi-linear regression using boxplots and the Z-score technique. 
- <img src = "https://user-images.githubusercontent.com/85246622/208271924-47545468-8913-4c14-96c9-fef7c6d4c757.png" width = "800" height = "600" />
- <img src = "https://user-images.githubusercontent.com/85246622/208271928-799b33b7-4e71-4a6c-94d7-54b0c059ed81.png" width = "800" height = "600" />
---------------------------------------------
- <h4>Multi-linear regression was implemented using sklearn's multi-linear regression model and a custom implementation. The data was split into training and test sets, normalized, and fit to the model. The model's performance was evaluated using the R2 score and visualized with a plot of the cost function versus the number of iterations.
- <img src = "https://user-images.githubusercontent.com/85246622/208651224-55f246ae-13b5-4ef0-aa3e-d39300dd9675.png" width ="400" height = "400" />  <img src = "https://user-images.githubusercontent.com/85246622/208652938-7148d3e2-d68f-4ded-8a37-fbfa459b875f.png" width ="400" height = "400" />
---------------------------------------------
<h2>------------------------------------------------------------------------------</h2>
<h3> Polynomial Regression implementation : </h3> 

-	<h4>Polynomial regression was implemented using linear and multi-linear models with third and second degree polynomial features, respectively. The data was split into training and test sets, normalized, and fit to the model. The model's performance was evaluated using the R2 score.
- <img src = "https://user-images.githubusercontent.com/85246622/208655184-b94c3c20-e07f-4a31-a175-049ae4d4b24f.png" width ="400" height = "400" />  <img src = "https://user-images.githubusercontent.com/85246622/208655504-99bb7a17-a274-4afd-84fe-11b5a5c77d0e.png" width ="400" height = "400" />
---------------------------------------------
-	<h4>A custom implementation of polynomial regression was also created using linear regression with second degree polynomial features. The data was handled just like above.
- <h4>Then the normal_equ function to use the new data from the above function and i applied the formula  <h1> (((X.T) X ) ** -1 ) X.T Y </h1> <h4>then predict the test set and print the R2_score </h4>
- <img src = "https://user-images.githubusercontent.com/85246622/208655937-6fdbf0b1-4f28-4680-b160-219b2aa1de5c.png" width ="400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/208656525-8830724f-3f2f-485c-9606-f837dc787d00.png" width ="400" height = "400" />
<h2>------------------------------------------------------------------------------</h2>
<h3> SVM Regression implementation : </h3> 

- •	SVM regression was implemented using linear, rbf, and poly kernels. The data was split into training and test sets, normalized, and fit to the model. The model's performance was evaluated using the R2 score .
- <img src = "https://user-images.githubusercontent.com/85246622/208657562-13949f5b-6510-4fe0-86ec-f658098ebb5d.png" width ="400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/208657751-8dd24b40-3301-47b7-a6be-5f8b91bf546b.png" width ="400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/208657775-e96fa438-eee0-4b19-9471-74f9ed23062d.png" width ="400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/208657797-575b7341-566d-4d85-93fc-b430bf41cb48.png" width ="400" height = "400" />
<h1>----------------------------------------------------------------</h1>
<h2> Machine Learning Classification Algorithms </h2>
<h3> Logistic Regression implementation : </h3> 

- <h4>Logistic regression was implemented using sklearn's logistic regression model and a custom implementation. The data was split into training and test sets, normalized, and fit to the model. The model's performance was evaluated using the confusion matrix, classification report, and R2 score. The results were visualized using a heatmap .
- <img src = "https://user-images.githubusercontent.com/85246622/209470687-86056203-2d03-4f36-b85a-b63b50cdbe1b.png" width = "400" height = "400" />- <img src = "https://user-images.githubusercontent.com/85246622/209470826-ae823799-f3fc-4a6a-8859-82721501802e.png" width = "400" height = "400" />
---------------------------------------------
- <h4>The custom implementation of logistic regression included functions to compute the sigmoid function, cost function, and predictions. The data was split into training and test sets, and the model's performance was evaluated using the confusion matrix, classification report, and R2 score.
- <img src = "https://user-images.githubusercontent.com/85246622/209471443-cfabe662-ebdc-443e-8e27-d6a0cb11f7d4.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209471450-3f539328-dd5f-482b-a792-d34cfa473d9a.png" width = "400" height = "400" />
<h2>------------------------------------------------------------------------------</h2>
<h3> SVM Classifier implementation : </h3> 

- <h4>A Support Vector Machine (SVM) classifier was implemented and data was split into training and test sets, then normalized using StandardScaler
- <h4>The LinearSVC function was used with a list of 4 values for the parameter C, and the model was fit to the training data with a fixed tolerance value. The test set was then predicted and a classification report and compilation table of 4 merging matrices were printed for each value of C
- <img src = "https://user-images.githubusercontent.com/85246622/209472440-5413d8d6-fa99-47b8-8d1b-c227931ea74a.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209472359-e00053b8-b19c-4a61-bcc8-bbea82f092d6.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209472379-2830b914-09b5-4cec-b857-7967ef3e8b76.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209472385-b759ad20-a8d3-4d86-986d-4488b904ee81.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209472404-dc959f14-77c5-4bb8-b389-797ef48c0965.png" width = "800" height = "800" /> 
---------------------------------------------
- <h4>A general purpose function was created to compare SVC kernels using the splitted training and test sets. The SVC RBF and Linear kernels were used, and for each kernel and each value of C, the model was fit to the training data, the test set was predicted, and a classification report and compilation table of 4 merging matrices were printed
-	<h4>Images were included to visualize the results of the SVM classifier implementation.

- <h4>For the SVC RBF kernel
- <img src = "https://user-images.githubusercontent.com/85246622/209474010-596f7f49-ac05-452f-8431-5e772606c437.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474014-775b1152-ec03-4614-a4ba-c81e15bb38e1.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474016-e34b9180-e997-4451-945a-d1ebca1c3109.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474018-f1118f03-965b-48a4-a3e5-92c45eedf3b4.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474020-ac68512e-02c9-46a3-bfa0-5e67fab6ec12.png" width = "800" height = "800" />
---------------------------------------------
- <h4>For the SVC Linear kernel
- <img src = "https://user-images.githubusercontent.com/85246622/209474154-cd8ff6e0-4902-4257-8787-47b86a5c9b93.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474158-cbc3ee28-d55b-444f-80c3-5edf65261ac6.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474162-d75041ac-2357-4990-b3fb-913ad0638006.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474164-62467d91-df31-49d7-9d02-ba8808431d43.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474168-d2d32096-c64e-4acc-b73e-156b714ed959.png" width = "800" height = "800" />
---------------------------------------------
- <h4>For the SVC Poly kernel
- <img src = "https://user-images.githubusercontent.com/85246622/209474172-e311fa40-1744-4bff-ae96-3f417f6c0cc4.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474175-32a1b9ca-122d-4ca6-83ff-f130f6136d5e.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474176-ba3528f4-72fe-4526-bb08-c350d2a84cdf.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474180-25b2c0dd-25ab-4177-9382-baa9df7cbdf7.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474183-32199c37-5f90-456c-ba7c-36d136941aeb.png" width = "800" height = "800" />
<h2>------------------------------------------------------------------------------</h2>
<h3> KNN Classifier implementation : </h3> 

- <h4>A K-Nearest Neighbors (KNN) classifier was implemented and data was split into training and test sets, then normalized using StandardScaler
-	<h4>The KNN function was used with a list of 4 values for the parameters "neighbors" and "weights". For each value of "neighbors", the model was initialized with that value and the "weights" parameter, fit to the training data, and used to predict the test set. A score, classification report, and table of 4 mixing matrices were printed for each value of "neighbors"
-	<h4>The "weights" parameter was set to "UNIFORM" and "DISTANCE" and the results were visualized using images.

- <h4>For the UNIFORM weights parameter
- <img src = "https://user-images.githubusercontent.com/85246622/209474811-06cb4e28-e754-4587-8fed-cd255f94b988.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474812-04146882-27f4-4bd8-b5a8-966b97eea807.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474814-1df92d22-e9e5-47de-bd6e-001d32e41c6c.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474817-a91c1fb5-50df-4949-8633-872fe9faa073.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474820-3745e2a0-4e78-4f1e-8679-cce7a7113bbb.png" width = "800" height = "800" />
---------------------------------------------
- <h4>For the DISTANCE weights parameter
- <img src = "https://user-images.githubusercontent.com/85246622/209474823-d10d0442-7be1-4fdd-aafe-47ffa98f8fb1.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474832-38ae5494-abfe-4f09-b30c-4cad36a57c7f.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474839-c503d55d-d61b-4788-ac69-2c5e3d4a072b.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209474845-1826cc30-4eda-4642-b285-f123194f4d71.png" width = "400" height = "400" />
- <img src = "https://user-images.githubusercontent.com/85246622/209474849-b833953f-d948-44b8-978c-d6d94463f0c4.png" width = "800" height = "800" />
<h2>------------------------------------------------------------------------------</h2>
<h3> Decision Tree Classifier implementation : </h3> 

- <h4> The given text describes the implementation of a Decision Tree Classifier on a dataset. 
- <h4>The model has three modifiable variables: criterion, splitter, and maximum depth. The dataset was split into training and test sets, and normalized using StandardScaler. 
- <h4>The model was then fit on the training set and used to predict the test set. The model's performance was evaluated using a score, classification report, and confusion matrix, which were plotted as a seaborn heatmap.
- <h4>The model was run with various combinations of the criterion, splitter, and maximum depth parameters, and the resulting heatmaps were plotted.

- <h4>For a GINI criterion and BEST as splitter with maximum depth of 7 
- <img src = "https://user-images.githubusercontent.com/85246622/209475337-bffeeced-fa39-4429-940e-7ba9348d5a54.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209475343-7898e66a-2676-40dd-8db5-fb05188342e5.png" width = "400" height = "400" />
---------------------------------------------
- <h4>For a GINI criterion and BEST as splitter with maximum depth of 5 
- <img src = "https://user-images.githubusercontent.com/85246622/209475446-509b3933-66ce-4aaa-bb37-60f8aefdcde9.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209475362-be8088f3-f2fe-4564-aecf-86da0fcb14b4.png" width = "400" height = "400" />
---------------------------------------------
- <h4>For a GINI criterion and RANDOM as splitter with maximum depth of 7 
- <img src = "https://user-images.githubusercontent.com/85246622/209475375-5e072904-ce42-4b6b-8212-947cc72396aa.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209475381-d167488a-2ef0-4b1e-adc2-8c122051c48b.png" width = "400" height = "400" />
---------------------------------------------
- <h4>For a GINI criterion and RANDOM as splitter with maximum depth of 5
- <img src = "https://user-images.githubusercontent.com/85246622/209475388-c4df705b-873b-4b2f-9b48-db1e41c1757e.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209475393-4b6ab334-b355-423a-b5e4-16324e978413.png" width = "400" height = "400" />
---------------------------------------------
- <h4>For a GINI criterion and RANDOM as splitter with maximum depth of NONE 
- <img src = "https://user-images.githubusercontent.com/85246622/209475403-bb4bb7c3-d0e7-4613-879b-1ac817113d52.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209475404-bd4f83d8-10f7-43f8-b277-fa21a39dfc20.png" width = "400" height = "400" />
---------------------------------------------
- <h4>For a ENTROPY criterion and BEST as splitter with maximum depth of 7
- <img src = "https://user-images.githubusercontent.com/85246622/209476257-b8159f1b-4113-4d3c-98de-8e07e30065f0.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209476201-60dfd0cc-859c-44b0-9466-f26f2b8c1962.png" width = "400" height = "400" />
---------------------------------------------
- <h4>For a ENTROPY criterion and BEST as splitter with maximum depth of 5 
- <img src = "https://user-images.githubusercontent.com/85246622/209476167-61cecd46-3b8c-4e9d-bcc8-ad6ec4369e11.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209476171-49545e3a-92a4-4d6f-95b2-6fe8537ea9ab.png" width = "400" height = "400" />
---------------------------------------------
- <h4>For a ENTROPY criterion and RANDOM as splitter with maximum depth of 7 
- <img src = "https://user-images.githubusercontent.com/85246622/209476158-5663d171-a74d-4209-9e5b-3b986f591edf.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209476161-8cd45ab9-6f6b-498b-b28f-e760d776035a.png" width = "400" height = "400" />
---------------------------------------------
- <h4>For a ENTROPY criterion and RANDOM as splitter with maximum depth of 5 
- <img src = "https://user-images.githubusercontent.com/85246622/209476151-24094e35-ac4e-45d2-8a95-fd0593c27d83.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209476153-ecfe3272-a099-45b0-b55f-5d7944d7e2bb.png" width = "400" height = "400" />
---------------------------------------------
- <h4>For a ENTROPY criterion and RANDOM as splitter with maximum depth of NONE 
- <img src = "https://user-images.githubusercontent.com/85246622/209476140-842ec623-fbe1-4624-8a22-279cf5aaa3bc.png" width = "400" height = "400" /> <img src = "https://user-images.githubusercontent.com/85246622/209476142-b88b8540-4d97-447e-a97d-9b67d01601b5.png" width = "400" height = "400" />
