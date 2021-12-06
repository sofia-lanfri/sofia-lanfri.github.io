## **INTERNSHIP GRIP (December2021): The Sparks Foundation**

### **Data Science and Business Analytics Internship**

 ![logo.png](attachment:logo.png)


##                               *Author: Sofia Lanfri*





# **Task 1: Linear regression with Python Scikit Learn**

## In this presentation, we will see how the Python Scikit-Learn library for machine learning can be used to implement regression functions.
 
In this regression task, we will predict the percentage of marks that a student is expected to score based upon the number of hours he studied. 

We use the data available at http://bit.ly/w-data.




## Table of Contents

* [Importing the libraries required](#libraries)
* [Importing the dataset: Reading and Preparing Data ](#data)
* [Visualizing the dataset: plotting the Scores vs Hours of study](#plotting)
* [Checking assumptions of Linear regression](#assumptions)
* [Applying Sklearn Linear Regression](#regression)
* [Training the algorithm](#training)
* [Making predictions](#predictions)
* [Evaluating the model](#evaluation)

## **Importing the required libraries:** <a class="anchor" id="libraries"></a>


```python
# Data manipulation, reading and writing data, 
import pandas as pd
# NUMERICAL COMPUTING TOOLS,  ARRAYS management
import numpy as np  
# PLOTTING
import matplotlib.pyplot as plt
# Data visualization
import seaborn as sns 
# Linear regression model
import sklearn


```

## **Importing the dataset: reading and preparing Data** <a class="anchor" id="data"></a>

In this step, we will import the dataset through the link with pandas library and then we will observe the data.


```python
data = pd.read_csv('http://bit.ly/w-data')
print('Shape of the dataset is: ', data.shape)
data.head()
```

    Shape of the dataset is:  (25, 2)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2.5</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5.1</td>
      <td>47</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3.2</td>
      <td>27</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.5</td>
      <td>75</td>
    </tr>
    <tr>
      <th>4</th>
      <td>3.5</td>
      <td>30</td>
    </tr>
  </tbody>
</table>
</div>




```python
#checking for any missing values

data.isnull().sum()
```




    Hours     0
    Scores    0
    dtype: int64




```python
# obtaining data descriptors

data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>25.000000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>5.012000</td>
      <td>51.480000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.525094</td>
      <td>25.286887</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.100000</td>
      <td>17.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.700000</td>
      <td>30.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>4.800000</td>
      <td>47.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>7.400000</td>
      <td>75.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>9.200000</td>
      <td>95.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Checking the attributes of the data

data.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 25 entries, 0 to 24
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype  
    ---  ------  --------------  -----  
     0   Hours   25 non-null     float64
     1   Scores  25 non-null     int64  
    dtypes: float64(1), int64(1)
    memory usage: 528.0 bytes


## Visualizing the dataset: plotting the Scores vs Hours of study <a class="anchor" id="plotting"></a>



If we plot the independent variable (hours) on the x-axis and dependent variable (percentage) on the y-axis,
we can observe that there is a linear relationship between "hours studied" and "percentage score". So, we can use the linear regression supervised machine model on it to predict further values.




```python
plt.rcParams["figure.figsize"] = [10,5]
data.plot( x ='Hours', y ='Scores', style='o', color='blue', markersize=5)
plt.title('Hours vs Percentage of marks')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage of marks')  
plt.grid()
plt.show()
```


    
![png](output_11_0.png)
    


y = mx + b

Where b is the intercept and m is the slope of the line.

Basically what the linear regression algorithm does is it fits multiple lines on the data points and returns the line that results in the least error.


```python
#Calculate correlation between variables

data.corr(method = 'pearson')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Hours</th>
      <th>Scores</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Hours</th>
      <td>1.000000</td>
      <td>0.976191</td>
    </tr>
    <tr>
      <th>Scores</th>
      <td>0.976191</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Plot histogram of data

hours = data['Hours']
scores = data['Scores']

hours.plot(kind = "hist", density = True)
hours.plot(kind = "kde")
plt.title('Histogram of variable Hours')  
```




    Text(0.5, 1.0, 'Histogram of variable Hours')




    
![png](output_14_1.png)
    



```python
# Plot histogram of data
 
scores.plot(kind = "hist", density = True)
scores.plot(kind = "kde")
plt.title('Histogram of variable Scores')  
```




    Text(0.5, 1.0, 'Histogram of variable Scores')




    
![png](output_15_1.png)
    


## Checking assumptions of Linear regression <a class="anchor" id="assumptions"></a>

Using the code developed by Jeff Macaluso

https://jeffmacaluso.github.io/post/LinearRegressionAssumptions/



```python
# Linear Regression
def linear_regression_assumptions(features, label, feature_names=None):
    """
    Tests a linear regression on the model to see if assumptions are being met
    """
    from sklearn.linear_model import LinearRegression
    
    # Setting feature names to x1, x2, x3, etc. if they are not defined
    if feature_names is None:
        feature_names = ['X'+str(feature+1) for feature in range(features.shape[1])]
    
    print('Fitting linear regression')
    # Multi-threading if the dataset is a size where doing so is beneficial
    if features.shape[0] < 100000:
        model = LinearRegression(n_jobs=-1)
    else:
        model = LinearRegression()
        
    model.fit(features, label)
    
    # Returning linear regression R^2 and coefficients before performing diagnostics
    r2 = model.score(features, label)
    print()
    print('R^2:', r2, '\n')
    print('Coefficients')
    print('-------------------------------------')
    print('Intercept:', model.intercept_)
    
    for feature in range(len(model.coef_)):
        print('{0}: {1}'.format(feature_names[feature], model.coef_[feature]))

    print('\nPerforming linear regression assumption testing')
    
    # Creating predictions and calculating residuals for assumption tests
    predictions = model.predict(features)
    df_results = pd.DataFrame({'Actual': label, 'Predicted': predictions})
    df_results['Residuals'] = abs(df_results['Actual']) - abs(df_results['Predicted'])

    
    
    
    def linear_assumption():
        """
        Linearity: Assumes there is a linear relationship between the predictors and
                   the response variable. If not, either a polynomial term or another
                   algorithm should be used.
        """
        print('\n=======================================================================================')
        print('Assumption 1: Linear Relationship between the Target and the Features')
        
        print('Checking with a scatter plot of actual vs. predicted. Predictions should follow the diagonal line.')
        
        # Plotting the actual vs predicted values
        sns.lmplot(x='Actual', y='Predicted', data=df_results, fit_reg=False, height=5)
        
        # Plotting the diagonal line
        line_coords = np.arange(df_results.min().min(), df_results.max().max())
        plt.plot(line_coords, line_coords,  # X and y points
                 color='darkorange', linestyle='--')
        plt.title('Actual vs. Predicted')
        plt.show()
        print('If non-linearity is apparent, consider adding a polynomial term')
        
        
    def normal_errors_assumption(p_value_thresh=0.05):
        """
        Normality: Assumes that the error terms are normally distributed. If they are not,
        nonlinear transformations of variables may solve this.
               
        This assumption being violated primarily causes issues with the confidence intervals
        """
        from statsmodels.stats.diagnostic import normal_ad
        print('\n=======================================================================================')
        print('Assumption 2: The error terms are normally distributed')
        print()
    
        print('Using the Anderson-Darling test for normal distribution')

        # Performing the test on the residuals
        p_value = normal_ad(df_results['Residuals'])[1]
        print('p-value from the test - below 0.05 generally means non-normal:', p_value)
    
        # Reporting the normality of the residuals
        if p_value < p_value_thresh:
            print('Residuals are not normally distributed')
        else:
            print('Residuals are normally distributed')
    
        # Plotting the residuals distribution
        plt.subplots(figsize=(12, 6))
        plt.title('Distribution of Residuals')
        sns.histplot(df_results['Residuals'], kde=True, bins=20)
        plt.show()
    
        print()
        if p_value > p_value_thresh:
            print('Assumption satisfied')
        else:
            print('Assumption not satisfied')
            print()
            print('Confidence intervals will likely be affected')
            print('Try performing nonlinear transformations on variables')
        
              
        
    def autocorrelation_assumption():
        """
        Autocorrelation: Assumes that there is no autocorrelation in the residuals. If there is
                         autocorrelation, then there is a pattern that is not explained due to
                         the current value being dependent on the previous value.
                         This may be resolved by adding a lag variable of either the dependent
                         variable or some of the predictors.
        """
        from statsmodels.stats.stattools import durbin_watson
        print('\n=======================================================================================')
        print('Assumption 4: No Autocorrelation')
        print('\nPerforming Durbin-Watson Test')
        print('Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data')
        print('0 to 2< is positive autocorrelation')
        print('>2 to 4 is negative autocorrelation')
        print('-------------------------------------')
        
        durbinWatson = durbin_watson(df_results['Residuals'])
        print('Durbin-Watson:', durbinWatson)
        
        if durbinWatson < 1.5:
            print('Signs of positive autocorrelation', '\n')
            print('Assumption not satisfied', '\n')
            print('Consider adding lag variables')
        elif durbinWatson > 2.5:
            print('Signs of negative autocorrelation', '\n')
            print('Assumption not satisfied', '\n')
            print('Consider adding lag variables')
        else:
            print('Little to no autocorrelation', '\n')
            print('Assumption satisfied')

            
    def homoscedasticity_assumption():
        """
        Homoscedasticity: Assumes that the errors exhibit constant variance
        """
        print('\n=======================================================================================')
        print('Assumption 5: Homoscedasticity of Error Terms')
        print('Residuals should have relative constant variance')
        
        # Plotting the residuals
        plt.subplots(figsize=(12, 6))
        ax = plt.subplot(111)  # To remove spines
        plt.scatter(x=df_results.index, y=df_results.Residuals, alpha=0.5)
        plt.plot(np.repeat(0, df_results.index.max()), color='darkorange', linestyle='--')
        ax.spines['right'].set_visible(False)  # Removing the right spine
        ax.spines['top'].set_visible(False)  # Removing the top spine
        plt.title('Residuals')
        plt.show() 
        print('If heteroscedasticity is apparent, confidence intervals and predictions will be affected')
        
        
    linear_assumption()   
    normal_errors_assumption()
    autocorrelation_assumption()
    homoscedasticity_assumption()
```


```python
# Prepare data to test assumptions.

x = data.iloc[:,:-1].values
y = data.iloc[:,1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=50)
# Assumptions
linear_regression_assumptions(x_train, y_train, feature_names= None)
```

    Fitting linear regression
    
    R^2: 0.9444169959416537 
    
    Coefficients
    -------------------------------------
    Intercept: 2.8102364628265804
    X1: 9.59563563045602
    
    Performing linear regression assumption testing
    
    =======================================================================================
    Assumption 1: Linear Relationship between the Target and the Features
    Checking with a scatter plot of actual vs. predicted. Predictions should follow the diagonal line.



    
![png](output_18_1.png)
    


    If non-linearity is apparent, consider adding a polynomial term
    
    =======================================================================================
    Assumption 2: The error terms are normally distributed
    
    Using the Anderson-Darling test for normal distribution
    p-value from the test - below 0.05 generally means non-normal: 0.0709274436337001
    Residuals are normally distributed



    
![png](output_18_3.png)
    


    
    Assumption satisfied
    
    =======================================================================================
    Assumption 4: No Autocorrelation
    
    Performing Durbin-Watson Test
    Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data
    0 to 2< is positive autocorrelation
    >2 to 4 is negative autocorrelation
    -------------------------------------
    Durbin-Watson: 1.4570790249690864
    Signs of positive autocorrelation 
    
    Assumption not satisfied 
    
    Consider adding lag variables
    
    =======================================================================================
    Assumption 5: Homoscedasticity of Error Terms
    Residuals should have relative constant variance



    
![png](output_18_5.png)
    


    If heteroscedasticity is apparent, confidence intervals and predictions will be affected


### Linearity

This assumes that there is a linear relationship between the predictors (e.g. independent variables or features) and the response variable (e.g. dependent variable or label). 
 
How to detect it: If there is only one predictor, this is pretty easy to test with a scatter plot. Ideally, the points should lie on or around a diagonal line on the scatter plot.


```python
# Visualization of scatter plot of data
plt.rcParams["figure.figsize"] = [8,4]
data.plot( x ='Hours', y ='Scores', style='o', color='blue', markersize=5)
plt.title('Hours vs Percentage of marks')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage of marks')  
plt.grid()
plt.show()
```


    
![png](output_20_0.png)
    


Linear Relationship between the Target and the Features
Checking with a scatter plot of actual vs. predicted. Predictions should follow the diagonal line.

![linearity.png](attachment:linearity.png)

 ### Normality of the Errors terms
 
 More specifically, this assumes that the error terms of the model are normally distributed.  
 
 How to detect it: There are a variety of ways to do so, but we’ll look at both a histogram and the p-value from the Anderson-Darling test for normality.

 

The error terms are normally distributed

Using the Anderson-Darling test for normal distribution
p-value from the test - below 0.05 generally means non-normal: 0.0709274436337001
Residuals are normally distributed

![errors_normal.png](attachment:errors_normal.png)

### No Autocorrelation of the Error Terms 

This assumes no autocorrelation of the error terms. Autocorrelation being present typically indicates that we are missing some information that should be captured by the model.


How to detect it: We can perform a Durbin-Watson test to determine if either positive or negative correlation is present. Alternatively, you could create plots of residual autocorrelations.


No Autocorrelation

Performing Durbin-Watson Test
Values of 1.5 < d < 2.5 generally show that there is no autocorrelation in the data

0 to 2 < is positive autocorrelation, 

 \> 2 to 4 is negative autocorrelation

-------------------------------------

Durbin-Watson: 1.4570790249690864

Signs of positive autocorrelation 

Assumption not satisfied 

Consider adding lag variables

### Homoscedasticity 


This assumes homoscedasticity, which is the same variance within our error terms. Heteroscedasticity occurs 

when we don’t have an even variance across the error terms.

How to detect it: Plot the residuals and see if the variance appears to be uniform.




Homoscedasticity of Error Terms
Residuals should have relative constant variance

![homocedasticity.png](attachment:homocedasticity.png)

## Applying Sklearn Linear Regression: <a class="anchor" id="regression"></a>

The goal of any linear regression algorithm is to accurately predict an output value from a given set of input features. In python, there are a number of different libraries that can create models to perform this task; of which Scikit-learn is the most popular and robust. 

Scikit-learn is a free open-source machine learning library for Python.

### Preparing the data
 
In this step we will divide the data into "features" (inputs) and "labels" (outputs). 

After that we will split the whole dataset into 2 parts - testing data and training data (using iloc function we will divide the data ).




```python
# To extract the attributes and labels, I execute the following script:


x = data.iloc[:,:-1].values
y = data.iloc[:,1].values
```


Splitting data into training and testing data

Now that we have our attributes and labels, the next step is to split this data into training and test sets. We'll do this by using Scikit-Learn's built-in train_test_split() method:

Scikit learn provides a function called train_test_split to divide your data sets randomly.


The above script splits 80% of the data to training set while 20% of the data to test set. The test_size variable is where we actually specify the proportion of test set.



```python
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=50)
```

## **Training the algorithm** <a class="anchor" id="training"></a>
We have split our data into training and testing sets, and now is finally the time to train our algorithm. 


```python
# Import linear regression from sci-kit learn module
# Store linear regression object in a variable called reg.

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
# fits a linear model
reg.fit(x_train,y_train)

print("Training phase complete.")
print('Intercept:', reg.intercept_)
print('Coeficient', reg.coef_)
```

    Training phase complete.
    Intercept: 2.8102364628265804
    Coeficient [9.59563563]


This means that for every one unit of change in hours studied, the change in the score is about 9.91%. Or in simpler words, if a student studies one hour more than they previously studied for an exam, they can expect to achieve an increase of 9.91% in the score achieved by the student previously.

### Visualizing the model

After training the model, now we can to visualize it.
 


```python
# Estimated coefficients

coef=reg.coef_

# Estimated intercept

interc=reg.intercept_
line = coef*x+interc

plt.scatter(x_train,y_train, color='red')
plt.plot(x,line, color='green')
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.title('Plot of the training data')
plt.grid()
plt.show()
```


    
![png](output_36_0.png)
    



```python
plt.scatter(x_test, y_test, color='red')
plt.plot(x, line, color='green');
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score') 
plt.title('Plot of the testing data')
plt.grid()
plt.show()
```


    
![png](output_37_0.png)
    


## **Making predictions** <a class="anchor" id="predictions"></a>
Now that we have trained our algorithm, it's time to make some predictions.


```python
print(x_test) # Testing data - Hours 
# Predict Y using the linear model with estimated coefficients
y_pred = reg.predict(x_test) # Predicting the scores
```

    [[8.9]
     [2.7]
     [6.9]
     [3.8]
     [1.1]]



```python
actual_predicted = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
actual_predicted

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Actual</th>
      <th>Predicted</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>95</td>
      <td>88.211394</td>
    </tr>
    <tr>
      <th>1</th>
      <td>30</td>
      <td>28.718453</td>
    </tr>
    <tr>
      <th>2</th>
      <td>76</td>
      <td>69.020122</td>
    </tr>
    <tr>
      <th>3</th>
      <td>35</td>
      <td>39.273652</td>
    </tr>
    <tr>
      <th>4</th>
      <td>17</td>
      <td>13.365436</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Histogram of the difference 

difference = np.array(y_test-y_pred)
plt.hist(difference, bins= 12) 
plt.title('Histogram of variable difference (y_test-y_pred)')  
```




    Text(0.5, 1.0, 'Histogram of variable difference (y_test-y_pred)')




    
![png](output_41_1.png)
    



### **What would be the predicted score if a student studies for 9.25 hours/day?**

You can also test with your own data




```python
h=9.25
s=reg.predict([[h]])
print("If a pupil studies for")
print("No of Hours = {} per day".format(h))
print("he/she will have a Predicted Score = {} in exams.".format(s[0]))

```

    If a pupil studies for
    No of Hours = 9.25 per day
    he/she will have a Predicted Score = 91.56986604454477 in exams.


## **Evaluating the model:** <a class="anchor" id="evaluation"></a>

In the last step, we are going to evaluate our trained model by calculating some metrics.

Fit a model  with train data, and calculate metrics with X and Y test data:


```python
from sklearn import metrics
from sklearn.metrics import r2_score


print("Mean Absolute Error:",metrics.mean_absolute_error(y_test,y_pred))


print("Mean Squared Error:",metrics.mean_squared_error(y_test,y_pred))

print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test,y_pred)))

# the larger the R2, the better the regression model fits your observations
print("R2 Score:",r2_score(y_test,y_pred))

```

    Mean Absolute Error: 4.5916495300630285
    Mean Squared Error: 25.58407829653998
    Root Mean Squared Error: 5.0580706100785084
    R2 Score: 0.971014141329942


### Graphics of the residuals

Train and test samples


```python
plt.scatter(reg.predict(x_train), reg.predict(x_train) - y_train, c='b', s = 40, alpha = 0.5)
plt.scatter(reg.predict(x_test), reg.predict(x_test) - y_test, c='g', s = 40 )
plt.hlines(y=0, xmin= 0, xmax = 120)
plt.title('Residuals plot using training (blue) and test (green) data')  
plt.ylabel('Residuals') 
```




    Text(0, 0.5, 'Residuals')




    
![png](output_47_1.png)
    


![maxresdefault.jpg](attachment:maxresdefault.jpg)


```python

```
