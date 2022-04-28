badmin <- read.csv("E:/Downloads/kevin_gyro1.csv") # Reading the data
head(badmin)

str(badmin)

badmin$output <- as.factor(badmin$output) #Converting character into factors

library(dplyr)
library(FSelector)

library(caret) #Classification and Regression Training
library(rpart) #Recursive partitioning and regression trees
library(rpart.plot) #Plotting the rpart model
library(e1071)
library(randomForest) #Random Forest Classification
#Splitting the data set into Training and Test Set
indexTrain <- createDataPartition(badmin$output, p=0.75, list=F)
data_train <- badmin[indexTrain,]  #Training data
data_test <- badmin[-indexTrain,]  #Test data  

#Configuring the control settings
ctrl_Set <- rpart.control(minsplit = 1, minbucket = 1, maxdepth = 10, cp = 0.01)
#Building the learner model
badmin1 <- rpart(output~., data = data_train, control = ctrl_Set)
badmin1

rpart.plot(badmin1, main = "accelerometer", extra = 108)

predict_dt <- predict(badmin1, newdata = data_test, type = "class")
predict_dt  #Accuracy of Decision Trees

confusionMatrix(predict_dt, factor(data_test$output))  # Decision Trees Classification


#Random Forest
rf_market <- randomForest(output~., data=badmin, ntree = 500)
rf_market


#Making predictions on the test data
predict_test_rf_model <- predict(rf_market, newdata = data_test, type = "class")
predict_test_rf_model

confusionMatrix(predict_dt, factor(data_test$output))  # Random Forest Classification


#SVM Classification

svm_market <- svm(output~., data=data_train)
svm_market
#Making predictions on the test data
predict_test_svm <- predict(svm_market, newdata = data_test, type = "class")
predict_test_svm
##Building the confusion matrix
confusionMatrix(predict_test_svm, factor(data_test$output))



######################## NAIVE BAYES #########################
#Building the learner model
market_nb <- naiveBayes(output~., data=data_train)
market_nb
#Making predictions on the test data
predict_test_nb <- predict(market_nb, newdata = data_test, type = "class")
predict_test_nb
##Building the confusion matrix
confusionMatrix(predict_test_nb, factor(data_test$output))



