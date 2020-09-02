#Predict Price of the computer

# Multilinear Regression
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

# loading the data
Computer_Data = pd.read_csv("C:\\Users\\home\\Desktop\\Data Science Assignments\\Python_codes\\Multiple_linear_regression\\Computer_Data.csv")

# to get top 6 rows
Computer_Data.head(6) # 

Computer_Data.shape  #(6259, 11)
Computer_Data.dtypes

 #number of null values

Computer_Data.info()# so there are no null values in the data

Computer_Data.columns

# number of unique values of column cd
Computer_Data.cd.nunique()# 2 unique datas
Computer_Data.cd.unique()#array(['no', 'yes'], dtype=object)


Computer_Data.multi.nunique()#: 2

Computer_Data.multi.unique()#array(['no', 'yes'], dtype=object)

Computer_Data.premium.nunique()

Computer_Data.premium.unique()#array(['yes', 'no'], dtype=object)



Computer_Data.replace(to_replace=['yes', 'no'],
           value= ['1', '0'], 
           inplace=True)
Computer_Data
Computer_Data.info()

##### or create dummy variables of columns whose dtypes is object #####

#pd.get_dummies(Computer_Data[dummy_vars], prefix=dummy_vars)
dummy_vars=['cd','multi','premium']
Computer_Data_dummy = pd.get_dummies(Computer_Data[dummy_vars])
# drop dummy  columns which are no

Computer_Data_dummy.drop('cd_no', axis=1, inplace=True)
Computer_Data_dummy.drop('multi_no', axis=1, inplace=True)
Computer_Data_dummy.drop('premium_no', axis=1, inplace=True)

# rename  dummy columns
Computer_Data_dummy.rename(columns={'cd_yes': 'cd'}, inplace=True)
Computer_Data_dummy.rename(columns={'multi_yes': 'multi'}, inplace=True)
Computer_Data_dummy.rename(columns={'premium_yes': 'premium'}, inplace=True)


Computer_Data_dummy.head(5)



#New_Data=pd.concat(['Computer_Data', 'Computer_Data_dummy',]sort='False')
#New_Data=np.append(Computer_Data, Computer_Data_dummy)

#Computer_Data.replace(to_replace ="yes", value ="1") 



# Correlation matrix 
correlation=Computer_Data.corr()








# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(Computer_Data)


# columns names
Computer_Data.columns

                             
# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model

         
############ Preparing MLR model  ####################
                
model1 = smf.ols('price~speed+hd+ram+screen+cd+multi+premium+ads+trend',data=Computer_Data).fit() # regression model

# Getting coefficients of variables               
model1.params

# Summary
model1.summary()#Adj. R-squared:                  0.775





#### preparing different models based on each column

# preparing model based only on speed
model_s=smf.ols('price~speed',data = Computer_Data).fit()  
model_s.summary() #  Adj. R-squared:                  0.090
# p-value <0.05 .. It is significant 

# Preparing model based only on hd
model_h=smf.ols('price~hd',data = Computer_Data).fit()  
model_h.summary() # Adj. R-squared:                  0.185

# Preparing model based only on ram
model_r=smf.ols('price~ram',data = Computer_Data).fit()  
model_r.summary() # Adj. R-squared:                  0.388

# Preparing model based only on screen
model_sr=smf.ols('price~screen',data = Computer_Data).fit()  
model_sr.summary() #  Adj. R-squared:                  0.087

# Preparing model based only on cd
model_cd=smf.ols('price~cd',data = Computer_Data).fit()  
model_cd.summary() #  Adj. R-squared:                  0.039

# Preparing model based only on multi
model_m=smf.ols('price~multi',data = Computer_Data).fit()  
model_m.summary() #   Adj. R-squared:                  0.000

# Preparing model based only on premium
model_p=smf.ols('price~premium',data = Computer_Data).fit()  
model_p.summary() #   Adj. R-squared:                  0.006

# Preparing model based only on ads
model_a=smf.ols('price~ads',data = Computer_Data).fit()  
model_a.summary() #   Adj. R-squared:                  0.003

# Preparing model based only on trend
model_t=smf.ols('price~trend',data = Computer_Data).fit()  
model_t.summary() #  Adj. R-squared:                  0.040




# Preparing model based only on hd & ram
model_hdram=smf.ols('price~hd+ram',data =Computer_Data).fit()  
model_hdram.summary() # Adj. R-squared:                  0.395
# Both coefficients p-value is significant... 


# Checking whether data has any influential values 
# influence index plots

import statsmodels.api as sm
sm.graphics.influence_plot(model1)
# index 5960 , 4477, 3783 is showing high influence so we can exclude that entire row
Computer_Data_new=Computer_Data.drop(Computer_Data.index[[5960,4477,3783]],axis=0)


# Studentized Residuals = Residual/standard deviation of residuals





# X => A B C D 
# X.drop(["A","B"],axis=1) # Dropping columns 
# X.drop(X.index[[5,9,19]],axis=0)

#X.drop(["X1","X2"],aixs=1)
#X.drop(X.index[[0,2,3]],axis=0)


# Preparing model                  
model1_new = smf.ols('price~speed+hd+ram+screen+cd+multi+premium+ads+trend',data=Computer_Data_new).fit()   

# Getting coefficients of variables        
model1_new.params

# Summary
model1_new.summary() #  Adj. R-squared:                  0.775

# Confidence values 99%
print(model1_new.conf_int(0.01)) # 99% confidence level


# Predicted values of price
price_pred = model1_new.predict(Computer_Data_new[['speed','hd','ram','screen','cd','multi','premium','ads','trend']])
price_pred

Computer_Data_new.head()


# calculating VIF's values of independent variables
rsq_speed = smf.ols('speed~hd+ram+screen+cd+multi+premium+ads+trend',data=Computer_Data_new).fit().rsquared  
vif_speed = 1/(1-rsq_speed)#1.264437878764203
print(vif_speed) 

rsq_hd = smf.ols('hd~speed+ram+screen+cd+multi+premium+ads+trend',data=Computer_Data_new).fit().rsquared  
vif_hd = 1/(1-rsq_hd) #4.366058002252695

rsq_ram = smf.ols('ram~speed+hd+screen+cd+multi+premium+ads+trend',data=Computer_Data_new).fit().rsquared  
vif_ram = 1/(1-rsq_ram) #3.052066483764764


rsq_screen = smf.ols('screen~speed+hd+ram+cd+multi+premium+ads+trend',data=Computer_Data_new).fit().rsquared  
vif_screen = 1/(1-rsq_screen) #1.0815840321414854

rsq_ads = smf.ols('ads~speed+hd+ram+screen+cd+premium+multi+trend',data=Computer_Data_new).fit().rsquared  
vif_ads = 1/(1-rsq_ads) #1.2197178702738756



rsq_trend = smf.ols('trend~speed+hd+ram+screen+cd+premium+multi+ads',data=Computer_Data_new).fit().rsquared  
vif_trend = 1/(1-rsq_trend) #0.5117421283095649


rsq_speed1 = smf.ols('speed~cd+multi+premium',data=Computer_Data_new).fit().rsquared  
vif_speed1 = 1/(1-rsq_speed1)#0.07116024326964443
 




           # Storing vif values in a data frame
d1 = {'Variables':['speed','hd','ram','screen','ads','trend'],'VIF':[vif_speed,vif_hd,vif_ram,vif_screen,vif_ads,vif_trend]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As there is no  higher VIF value

# Added varible plot 
sm.graphics.plot_partregress_grid(model1_new)

# added varible plot for weight is not showing any significance 

# final model
final_model= smf.ols('price~speed+hd+ram+screen+cd+premium+multi+ads+trend',data = Computer_Data_new).fit()
final_model.params
final_model.summary() # 0.809
# As we can see that r-squared value has increased

price_pred = final_model.predict(Computer_Data_new)

import statsmodels.api as sm
# added variable plot for the final model
sm.graphics.plot_partregress_grid(final_model)


######  Linearity #########
# Observed values VS Fitted values
plt.scatter(Computer_Data_new.price,price_pred,c="r");plt.xlabel("observed_values");plt.ylabel("fitted_values")

# Residuals VS Fitted Values 
plt.scatter(price_pred,final_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")


########    Normality plot for residuals ######
# histogram
plt.hist(final_model.resid_pearson) # Checking the standardized residuals are normally distributed

# QQ plot for residuals 
import pylab          
import scipy.stats as st

# Checking Residuals are normally distributed
st.probplot(final_model.resid_pearson, dist="norm", plot=pylab)


############ Homoscedasticity #######

# Residuals VS Fitted Values 
plt.scatter(price_pred,final_model.resid_pearson,c="r"),plt.axhline(y=0,color='blue');plt.xlabel("fitted_values");plt.ylabel("residuals")



### Splitting the data into train and test data 

from sklearn.model_selection import train_test_split
Computer_Data_train,Computer_Data_test  = train_test_split(Computer_Data_new,test_size = 0.2) # 20% size

# preparing the model on train data 

model_train = smf.ols("price~speed+hd+ram+screen+cd+premium+multi+ads+trend",data=Computer_Data_train).fit()

# train_data prediction
train_pred = model_train.predict(Computer_Data_train)

# train residual values 
train_resid  = train_pred - Computer_Data_train.price

# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid*train_resid))

# prediction on test data set 
test_pred = model_train.predict(Computer_Data_test)

# test residual values 
test_resid  = test_pred - Computer_Data_test.price

# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid*test_resid))
