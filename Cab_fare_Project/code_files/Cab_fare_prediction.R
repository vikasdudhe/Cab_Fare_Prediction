#remove all pre-exisiting data from environment 
rm(list=ls())

#set the path for working directory and data files
setwd("D:\\edwisor_details\\project\\cab_fare_1\\")

#install required libraries 
lib_pack = c('ggplot2', 'corrgram', 'DMwR2', 'usdm', 'caret', 'randomForest', 'e1071',
             'DataCombine', 'doSNOW', 'inTrees', 'rpart.plot', 'rpart', 'MASS', 'xgboost',
             'stats')

#load packages 
install.packages(c('ggplot2', 'corrgram', 'DMwR2', 'usdm', 'caret', 'randomForest', 'e1071',
                   'DataCombine', 'doSNOW', 'inTrees', 'rpart.plot', 'rpart', 'MASS', 'xgboost',
                   'stats'))

#check packages loaded or not 
lapply(lib_pack, require, character.only = TRUE)

#Column information
# The details of data attributes in the dataset are as follows:
# pickup_datetime - timestamp value indicating when the cab ride started.
# pickup_longitude - float for longitude coordinate of where the cab ride started.
# pickup_latitude - float for latitude coordinate of where the cab ride started.
# dropoff_longitude - float for longitude coordinate of where the cab ride ended.
# dropoff_latitude - float for latitude coordinate of where the cab ride ended.
# passenger_count - an integer indicating the number of passengers in the cab ride.

# loading datasets
train = read.csv("train_cab.csv", header = T, na.strings = c(" ", "", "NA"))
test = read.csv("test.csv")
#test_pickup_datetime = test["pickup_datetime"]

# Structure of train data
str(train)
# Structure of test data
str(test)
#summary of train and test
summary(train)
summary(test)

#check the sample data from train and test
head(train,5)
head(test,5)

#missing value counts for columns
sum(is.na(train$fare_amount))
sum(is.na(train$pickup_datetime))
sum(is.na(train$passenger_count))

#from above statments we can conclude we have 79 data points which having missing values

####################Deep DIVE in each column of dataset (EDA) ################################

# Changing the data types of variables
train$fare_amount = as.numeric(as.character(train$fare_amount))
train$passenger_count=round(train$passenger_count)

#removing outliers from basic understanding of data
#1) For fare amount cannot be negative values and zero. so need to remove 

#checking condition for fareamount
train[which(train$fare_amount < 1 ),]

#counting rows of satisfying condition 
nrow(train[which(train$fare_amount < 1 ),])

#removing the rows from training dataset
train = train[-which(train$fare_amount < 1 ),]

#confirming the size of dataframe after remove 
dim(train)

#2) next we check for passenger count variable 
# passenger count for hatchback and suv max capacity is 6 members
# passenger count which having value zero need to remove that doesn't make any sense
for (i in seq(4,20,by=1)){
  print(paste('passenger_count above ' ,i,nrow(train[which(train$passenger_count > i ),])))
}

#we can see that after 6 count we have 20 common so will remove after 6 passanger_count
nrow(train[which(train$passenger_count > 6 ),])

# Also we need to see if there are any passenger_count==0
nrow(train[which(train$passenger_count <1 ),])

#total 78 rows which need to remove for passenger_count
train = train[-which(train$passenger_count < 1 ),]
train = train[-which(train$passenger_count > 6),]

#confirming the size of train
dim(train)

#3) cleaning for latitudes and longitudes valid ranges from dataset
#valid range for latitude -90 to 90
#valid range for longitude -180 to 180

print(paste('pickup_longitude above 180=',nrow(train[which(train$pickup_longitude >180 ),])))
print(paste('pickup_longitude above -180=',nrow(train[which(train$pickup_longitude < -180 ),])))

print(paste('pickup_latitude above 90=',nrow(train[which(train$pickup_latitude > 90 ),])))
print(paste('pickup_latitude above -90=',nrow(train[which(train$pickup_latitude < -90 ),])))

print(paste('dropoff_longitude above 180=',nrow(train[which(train$dropoff_longitude > 180 ),])))
print(paste('dropoff_longitude above -180=',nrow(train[which(train$dropoff_longitude < -180 ),])))

print(paste('dropoff_latitude above -90=',nrow(train[which(train$dropoff_latitude < -90 ),])))
print(paste('dropoff_latitude above 90=',nrow(train[which(train$dropoff_latitude > 90 ),])))

#from above observations we can see that for pickup_latitude above 90 we have 1 observation which is out of range

#also we need to check if those are having zero values or not 
nrow(train[which(train$pickup_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])
nrow(train[which(train$dropoff_longitude == 0 ),])
nrow(train[which(train$pickup_latitude == 0 ),])

#removing zero values for latitudes and longitudes and out of range data points
train = train[-which(train$pickup_latitude > 90),]
train = train[-which(train$pickup_longitude == 0),]
train = train[-which(train$dropoff_longitude == 0),]

#confirming the size of train data
dim(train)

#save a copy of this training clean set of data
df_version1 = train

################################# Missing Value Analysis #######################

#now we will look for missing values analysis and there treatment

#first finding the sum of na under each column
missing_val = data.frame(apply(train,2,function(x){sum(is.na(x))}))
print(missing_val)

#creating another column for missing values 
missing_val$Columns = row.names(missing_val)
names(missing_val)[1] =  "Missing_p"

#converting missing values count to percentage format
missing_val$Missing_p = (missing_val$Missing_p /nrow(train)) * 100

#sort as per percentage
missing_val = missing_val[order(-missing_val$Missing_p),]

#making index values null
row.names(missing_val) = NULL

#creating number indexing
missing_val = missing_val[,c(2,1)]

#check the missing tablulated data
missing_val

#As per analysis we will now work on passenger_count column

#checking unique values in passenger_count in train and test data
unique(train$passenger_count)
unique(test$passenger_count)

#so training data contains NA we need to take care of that and we will convert it to factor to makeit categorical
train[,'passenger_count'] = factor(train[,'passenger_count'], labels=(1:6))
test[,'passenger_count'] = factor(test[,'passenger_count'], labels=(1:6))

#we can't use mean mode imputation for this dataset because data is biased for 1 in passenger count 

#remove the datetime column before imputation

#so we will look for KNN imputations
library('VIM')
#applying kNN for imputation with 19 neibhours
train <- kNN(train,variable = c('passenger_count', 'fare_amount'), k = 19)

#store imputated data in csv
write.csv(train, "imputed_df.csv", row.names = FALSE)

#imputation data from file
imputed_df = read.csv("imputed_data2.csv", header=T)
train = imputed_df
#checking for missing data 
sum(is.na(train))

########################## Outlier Analysis ####################################

#we will use boxplot for outlier in fare amount 
plt1 = ggplot(train,aes(x = factor(passenger_count),y = fare_amount))
plt1 + geom_boxplot(outlier.colour="red", fill = "cyan" ,outlier.shape=18,outlier.size=1, notch=FALSE)+ylim(0,100)

# Replace all outliers with NA and impute
fare_values = train[,"fare_amount"] %in% boxplot.stats(train[,"fare_amount"])$out
train[which(fare_values),"fare_amount"] = NA

#lets check the NA\'s
print(sum(is.na(train$fare_amount)))

#impute the na with knn with value 5
train = knnImputation(train, k=3, scale=TRUE)
  
df_version2 = train

########################## Feature Engineering ################################
# 1.Feature Engineering for timestamp variable
# we will derive new features from pickup_datetime variable
# new features will be year,month,day_of_week,hour
#Convert pickup_datetime from factor to date time
train$pickup_date = as.Date(as.character(train$pickup_datetime))
train$pickup_weekday = as.factor(format(train$pickup_date,"%u"))# Monday = 1
train$pickup_mnth = as.factor(format(train$pickup_date,"%m"))
train$pickup_yr = as.factor(format(train$pickup_date,"%Y"))
pickup_time = strptime(train$pickup_datetime,"%Y-%m-%d %H:%M:%S")
train$pickup_hour = as.factor(format(pickup_time,"%H"))

#Add same features to test set
test$pickup_date = as.Date(as.character(test$pickup_datetime))
test$pickup_weekday = as.factor(format(test$pickup_date,"%u"))# Monday = 1
test$pickup_mnth = as.factor(format(test$pickup_date,"%m"))
test$pickup_yr = as.factor(format(test$pickup_date,"%Y"))
pickup_time = strptime(test$pickup_datetime,"%Y-%m-%d %H:%M:%S")
test$pickup_hour = as.factor(format(pickup_time,"%H"))

# there was 1 'na' in pickup_datetime which created na's in above feature engineered variables.
sum(is.na(train))
# we will remove that 1 row of na's
train = na.omit(train)

train = subset(train,select = -c(pickup_datetime,pickup_date))
test = subset(test,select = -c(pickup_datetime,pickup_date))

# 2.Calculate the distance travelled using longitude and latitude
deg_to_rad = function(deg){
  (deg * pi) / 180
}
haversine = function(long1,lat1,long2,lat2){
  #long1rad = deg_to_rad(long1)
  phi1 = deg_to_rad(lat1)
  #long2rad = deg_to_rad(long2)
  phi2 = deg_to_rad(lat2)
  delphi = deg_to_rad(lat2 - lat1)
  dellamda = deg_to_rad(long2 - long1)
  
  a = sin(delphi/2) * sin(delphi/2) + cos(phi1) * cos(phi2) * 
    sin(dellamda/2) * sin(dellamda/2)
  
  c = 2 * atan2(sqrt(a),sqrt(1-a))
  R = 6371e3
  R * c / 1000 #1000 is used to convert to meters
}
# Using haversine formula to calculate distance fr both train and test
train$dist = haversine(train$pickup_longitude,train$pickup_latitude,train$dropoff_longitude,train$dropoff_latitude)
test$dist = haversine(test$pickup_longitude,test$pickup_latitude,test$dropoff_longitude,test$dropoff_latitude)


# Using haversine formula to calculate distance fr both train and test
train$dist = haversine(train$pickup_longitude,train$pickup_latitude,train$dropoff_longitude,train$dropoff_latitude)
test$dist = haversine(test$pickup_longitude,test$pickup_latitude,test$dropoff_longitude,test$dropoff_latitude)

# We will remove the variables which were used to feature engineer new variables
train = subset(train,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))
test = subset(test,select = -c(pickup_longitude,pickup_latitude,dropoff_longitude,dropoff_latitude))

str(train)
summary(train)

######################## Feature selection #####################################
numeric_index = sapply(train,is.numeric) #selecting only numeric

numeric_data = train[,numeric_index]

cnames = colnames(numeric_data)
#Correlation analysis for numeric variables
corrgram(train[,numeric_index],upper.panel=panel.pie, main = "Correlation Plot")

#ANOVA for categorical variables with target numeric variable

#aov_results = aov(fare_amount ~ passenger_count * pickup_hour * pickup_weekday,data = train)
aov_results = aov(fare_amount ~ passenger_count + pickup_hour + pickup_weekday + pickup_mnth + pickup_yr,data = train)

summary(aov_results)

# pickup_weekdat has p value greater than 0.05 
train = subset(train,select=-pickup_weekday)

#remove from test set
test = subset(test,select=-pickup_weekday)

########################## Feature Scaling #####################################
#Normality check
# qqnorm(train$fare_amount)
# histogram(train$fare_amount)
library(car)
# dev.off()
par(mfrow=c(1,2))
qqPlot(train$fare_amount)                             # qqPlot, it has a x values derived from gaussian distribution, if data is distributed normally then the sorted data points should lie very close to the solid reference line 
truehist(train$fare_amount)                           # truehist() scales the counts to give an estimate of the probability density.
lines(density(train$fare_amount))  # Right skewed      # lines() and density() functions to overlay a density plot on histogram

#Normalisation

print('dist')
train[,'dist'] = (train[,'dist'] - min(train[,'dist']))/
  (max(train[,'dist'] - min(train[,'dist'])))

# #check multicollearity
# library(usdm)
# vif(train[,-1])
# 
# vifcor(train[,-1], th = 0.9)

#################### Splitting train into train and validation subsets ###################
set.seed(1000)
tr.idx = createDataPartition(train$fare_amount,p=0.75,list = FALSE) # 75% in trainin and 25% in Validation Datasets
train_data = train[tr.idx,]
test_data = train[-tr.idx,]

rmExcept(c("test","train","df",'df1','df2','df3','test_data','train_data','test_pickup_datetime'))

############################## Model Selection  ##############################
#Error metric used to select model is RMSE

#############            Linear regression               #################
lm_model = lm(fare_amount ~.,data=train_data)

summary(lm_model)
str(train_data)
plot(lm_model$fitted.values,rstandard(lm_model),main = "Residual plot",
     xlab = "Predicted values of fare_amount",
     ylab = "standardized residuals")


lm_predictions = predict(lm_model,test_data[,2:6])

qplot(x = test_data[,1], y = lm_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],lm_predictions)
# mae        mse       rmse       mape 
# 3.5303114 19.3079726  4.3940838  0.4510407  

#############                             Decision Tree            #####################

Dt_model = rpart(fare_amount ~ ., data = train_data, method = "anova")

summary(Dt_model)
#Predict for new test cases
predictions_DT = predict(Dt_model, test_data[,2:6])

qplot(x = test_data[,1], y = predictions_DT, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],predictions_DT)
# mae       mse      rmse      mape 
# 1.8981592 6.7034713 2.5891063 0.2241461 


#############                             Random forest            #####################
rf_model = randomForest(fare_amount ~.,data=train_data)

summary(rf_model)

rf_predictions = predict(rf_model,test_data[,2:6])

qplot(x = test_data[,1], y = rf_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],rf_predictions)
# mae       mse      rmse      mape 
# 1.9053850 6.3682283 2.5235349 0.2335395

############          Improving Accuracy by using Ensemble technique ---- XGBOOST             ###########################
train_data_matrix = as.matrix(sapply(train_data[-1],as.numeric))
test_data_data_matrix = as.matrix(sapply(test_data[-1],as.numeric))

xgboost_model = xgboost(data = train_data_matrix,label = train_data$fare_amount,nrounds = 15,verbose = FALSE)

summary(xgboost_model)
xgb_predictions = predict(xgboost_model,test_data_data_matrix)

qplot(x = test_data[,1], y = xgb_predictions, data = test_data, color = I("blue"), geom = "point")

regr.eval(test_data[,1],xgb_predictions)
# mae       mse      rmse      mape 
# 1.6183415 5.1096465 2.2604527 0.1861947  

#############                         Finalizing and Saving Model for later use                         ####################
# In this step we will train our model on whole training Dataset and save that model for later use
train_data_matrix2 = as.matrix(sapply(train[-1],as.numeric))
test_data_matrix2 = as.matrix(sapply(test,as.numeric))

xgboost_model2 = xgboost(data = train_data_matrix2,label = train$fare_amount,nrounds = 15,verbose = FALSE)

# Saving the trained model
saveRDS(xgboost_model2, "./final_Xgboost_model_using_R.rds")

# loading the saved model
super_model <- readRDS("./final_Xgboost_model_using_R.rds")
print(super_model)

# Lets now predict on test dataset
xgb = predict(super_model,test_data_matrix2)

xgb_pred = data.frame(test_pickup_datetime,"predictions" = xgb)

# Now lets write(save) the predicted fare_amount in disk as .csv format 
write.csv(xgb_pred,"xgb_predictions_R.csv",row.names = FALSE)