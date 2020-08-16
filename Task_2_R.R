# Clear console (Can be done by pressing ctrl/cmd + L)
cat("\014")
# Clear environment
rm(list=ls())
# Get working directory
getwd()


if (!require('ROCR')) install.packages('ROCR');
library(ROCR)
if (!require('e1071')) install.packages('e1071');
library(e1071)
if (!require('caret')) install.packages('caret');
library(caret)
if (!require('dplyr')) install.packages('dplyr');
library(dplyr)
if (!require('dataPreparation')) install.packages('dataPreparation');
library(	dataPreparation)
################## Functions #########################
getCharacterColumnsNames <- function(df){
  
  chrColsNames <- names(df[, sapply(df, is.character)])
  
  return(chrColsNames)
}




################## Main Code #########################

#Reading the dataframe:
train_data <- read.csv2("training.csv")

# Getting to know the data:
summary(train_data)
head(train_data)

# Calculating percentage of our class labels:
# This tells us that 92.5% of our data is biased towards a certain label -class imbalance-
prop.table(table(train_data$classLabel))


# Knowing NA missing values:
colSums(is.na(train_data))
sapply(train_data, function(x) sum(is.na(x)))

# After observing the data, it can be found that variable 18 is almost filled with NA so we should drop it:
train_data_modified <- subset(train_data, select = -c(variable18))

# We can handle NA data in rest of columns in various ways,
# the simplest way is using NA.omit as the number of missing values is relatively low.
# We can also replace the missing numeric values using the mean and/or median.

#CASE 1: Omitting NA values
train_data_NA_Omit <- na.omit(train_data_modified)

# Changing all character columns into factors:
charColumnsNames <- getCharacterColumnsNames(train_data_NA_Omit)
train_data_NA_Omit[charColumnsNames] <- lapply(train_data_NA_Omit[charColumnsNames] , factor)

#################### Binary Classification Techniques ##########################
#Reading the validation dataframe
# while applying same pre-processing sequence as train data:
val_data <- read.csv2("validation.csv")
val_data <- subset(val_data, select = -c(variable18))
val_data[charColumnsNames] <- lapply(val_data[charColumnsNames] , factor)

# Solving difference in number of levels of factors between validation and training:
common <- intersect(names(train_data_NA_Omit), names(val_data)) 
for (p in common) { 
  if (class(train_data_NA_Omit[[p]]) == "factor") { 
    levels(val_data[[p]]) <- levels(train_data_NA_Omit[[p]]) 
  } 
}
# If we observe the data, we can see that the column "variable19" in training data is 
# almost the "classLabel" but in numeric form, this will cause some confusion with the validation set.

# Logistic Regression:
mylogit_1 <- glm(classLabel ~.,data = train_data_NA_Omit,family = binomial("logit"), maxit = 100)
step(mylogit_1, direction = "backward")
# As predicted, the step function concluded that variable19 only should be used in our model
mylogit_1 <- glm(classLabel~ variable19,data = train_data_NA_Omit,family = binomial("logit"), maxit = 100)
summary(mylogit_1)

fitted.results <- predict(mylogit_1,newdata=subset(val_data,select=c(1:17)),type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)

misClasificError <- mean(fitted.results != as.numeric(as.factor(val_data$classLabel))-1)
print(paste('Logistic Regression\'s Correct Classification Rate:',(1-misClasificError)*100))


#Predicting on validation set:
pred_with_var19 = predict(mylogit_1, val_data, type = "response")
pr <- prediction(pred_with_var19, val_data$classLabel)
prf <- performance(pr, measure = "tpr", x.measure = "fpr")
aucObj = performance(pr, measure="auc")
auc = aucObj@y.values[[1]]
auc
plot(prf, main = paste("Area under the curve:", auc))
# Area Under Curve is 49.5% 

# SVM:
model <- svm(classLabel~. ,data=train_data_NA_Omit,kernel = "linear")
predictionSVM <- predict(model, val_data, na.action = na.pass)

cfmSVM<-confusionMatrix(predictionSVM,val_data$classLabel[1:191])
cfmSVM
# Using SVM produced similar results to the Logistic regression


