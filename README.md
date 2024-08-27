This is a comprehensive end-to-end performed Data Science project that solves a regression problem using supervised machine learning models.
In this project, I have very closely implemented all the steps of a Data Science Project lifecycle.

# Problem Statement: Predict the prices for used vehicles in the market based on vehicle features.

#	My Approach:

i.	Data Collection: Dummy data on Used Vehicle Price Prediction from Kaggle.

ii.	Performed Data Pre-processing techniques:
	Removed Duplicate records
	Performed Feature Engineering Techniques: Aggregation by combing multiple features into a single one, Deriving new features from existing ones & Bucketing by grouping numeric values into discreet bins
	Dropped columns that had the most % of missing values
	Dropped records from the columns that had missing values more than 35%
	Label Encoded multi-class columns using Label Encoder
	Imputed the rest of the missing values by using the KNN Imputer

iii.	Performed EDA:
a.	Data Summarization: To understand the data type of each column, unique values in each column, etc.
b.	Plotted Box Plot for Outlier Detection & got rid of potential outliers using IQR concept
c.	Plotted bar plot to understand the price of vehicles given their brand, model, type & other factors
d.	Plotted Line Chart to Understand the Trends in vehicle price over the years given its brand, model, type & other features
e.	Plotted Scatter Plot to Understand the Relation between mileage & years for different vehicle brands, models, body types, drivetrain, fuel types & transmission type of the vehicle
f.	Plotted a Correlation Matrix to understand the Relation Between Different Features in the dataset & to understand If Multi Collinearity exists.

iv.	Model Selection:
a. Tried & A/B Tested Multiple Regression Models like Linear, Lasso & Ridge Regression, SVM Regressor, DT Regressor, Random Forest Regressor, etc to find the best model.
b. Tried Hyperparameter tuning using GridSearchCV wrt DT & RF to find the best model hyperparameters.
c. Found out the Feature Importance of each attribute in the dataset wrt the target variable 
d. Used Performance Metrics: R Square & Adjusted R Square to Evaluate the Model's Performance
e. Chose the Model with best R square & Adjusted R square value


v. Final Model:
a. Dataset Transformation: Transformed the dataset using the StandardScalar technique.
b. Model Training: Split the dataset using train-test split.
c. Trained & Fitted the Model on Regression Algorithm that uses the Bagging technique: Random Forest Regressor
d. Output Prediction & Visualization: Predicted the output & performed some visualization.
e. Model Deployment: Created a flask interface/UI to test the vehicle price prediction model.

