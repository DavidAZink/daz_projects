#YOU WILL NEED TO SET YOUR WORKING DIRECTORY TO THE LOCATION OF THE "df.rds" DATA FILE IN ORDER TO RUN THIS PROGRAM. You also may need to install the packages listed below:
library(tidyverse)
library(plm)
library(gdata)
library(data.table)
library(mltools)
#Change this working directory to the location of the data file on your local machine!
setwd("C:/Users/dzink/Documents/fannie")

#call in data, create some features, one hot encode factors
df=readRDS('df.rds')
dmy=one_hot(data.table(df[,c('first_home', 'units', 'occupancy', 'property_type', 'l_delinquent')] ))
df=cbind(df[, !colnames(df) %in% c('first_home', 'units', 'occupancy', 'property_type', 'l_delinquent')], dmy)
df$l_estimated_ltv=lag(df$estimated_ltv, 1)
df$l_estimated_ltv=ifelse(df$loan_age==0, df$estimated_ltv, df$l_estimated_ltv)
df$spread_ltv=df$spread*df$l_estimated_ltv
df$spread_fico=df$spread*df$fico

#normalize features
features=c('l_delinquent', 'loan_age', 'months_left', 'l_estimated_ltv', 'fico', 'units_1', 'units_2', 'units_3', 'units_4', 
		'occupancy_I', 'occupancy_P', 'occupancy_S' , 'o_cltv', 'o_dti', 'o_upb' ,'o_ltv', 
		'property_type_CP', 'property_type_MH', 'property_type_CO', 'property_type_PU', 'property_type_SF', 
		'first_home_9', 'first_home_Y', 'first_home_N', 'borrowers', 'spread', 'spread_ltv', 'spread_fico')
normalizer<-function(x)((x-mean(x, na.rm=TRUE))/sd(x, na.rm=TRUE))
select(df, features) %>% apply(2, normalizer) %>% data.frame(cbind(df[, c('sequence', 'month', 'delinquent')])) -> df
head(pdata.frame(df, index=c('sequence', 'month')))

#create train/test split
library(caret)
set.seed(25)
training_idx=createDataPartition(df$delinquent, p=0.8)[[1]]
training=df[training_idx, ]
test=df[-training_idx, ]

summary(training)

#train logistic classifier
library(glmnet)
print(summary(training$delinquent))
weights=c(1, length(which(as.character(training$delinquent)=='current'))/length(which(as.character(training$delinquent)=='delinquent')))
training$weights=ifelse(as.character(training$delinquent)=='delinquent', weights[2], 1)
logistic<-glmnet(y=training$delinquent, x=as.matrix(training[, features]),
			 family='binomial', lambda=0, weights=training$weights)
logistic_predictions=data.frame(predicted=as.vector(predict(logistic, as.matrix(test[, features]), type='response')))
logistic_predictions$actual=ifelse(as.character(test$delinquent)=='delinquent', 1, 0)

#train random forest model
library(randomForest)
rf<-randomForest(x=training[, c(features)], y=training$delinquent, weights=weights, ntrees=500, keep.forest=TRUE)
rf_predictions=data.frame(predicted=as.vector(predict(rf, as.matrix(test[, c(features)]), type='prob')[, 2]))
rf_predictions$actual=ifelse(as.character(test$delinquent)=='delinquent', 1, 0)
plot(rf, main='Random Forest')
legend("topright", colnames(rf$err.rate), col=1:4,cex=0.8,fill=1:4)

#Plot ROC curve for each model and calculate ROC AUC for each model
library(ROCR)

pred_ROCR_rf <- ROCR::prediction(rf_predictions$predicted, rf_predictions$actual)
roc_ROCR_rf <- ROCR::performance(pred_ROCR_rf, measure = "tpr", x.measure = "fpr", pos='1')
plot(roc_ROCR_rf, type='l', col='red', main="Random Forest")
abline(a = 0, b = 1)

pred_ROCR_log <- ROCR::prediction(logistic_predictions$predicted, logistic_predictions$actual)
roc_ROCR_log <- ROCR::performance(pred_ROCR_log, measure = "tpr", x.measure = "fpr", pos='1')
plot(roc_ROCR_log, type='l', col='red', main='Logistic Model')
abline(a=0, b=1)


#Create plot of threshold vs F1 Score to help pick optimal threshold
library(MLmetrics)
F1_logistic=data.frame()
F1_randomforest=data.frame()
for (i in seq(0.05, 0.99, 0.01)){
	F1_logistic=rbind(F1_logistic, cbind(F1_Score(logistic_predictions$actual, ifelse(logistic_predictions$predicted>=i, 1, 0), positive='1'), i))	
	F1_randomforest=rbind(F1_randomforest, cbind(F1_Score(rf_predictions$actual, ifelse(rf_predictions$predicted>=i, 1, 0), positive='1'), i))
	}
colnames(F1_logistic)=c('f1_score', 'threshold')
colnames(F1_randomforest)=c('f1_score', 'threshold')

plot(F1_randomforest$threshold, F1_randomforest$f1_score, col='blue', type='l', xlab='Threshold', ylab='F1 Score')
lines(F1_logistic$threshold, F1_logistic$f1_score, col='red')
legend("bottomleft", c("random forest", "logistic"), lty=c(1, 1), col=c('blue', 'red'))

cm_rf=confusionMatrix(as.factor(ifelse(rf_predictions$predicted>=F1_randomforest$threshold[which(F1_randomforest$f1_score==max(F1_randomforest$f1_score))[1]], 1, 0)), 
			as.factor(rf_predictions$actual), positive="1")
cm_logistic=confusionMatrix(as.factor(ifelse(logistic_predictions$predicted>=F1_logistic$threshold[which(F1_logistic$f1_score==max(F1_logistic$f1_score))[1]], 1, 0)), 
			as.factor(logistic_predictions$actual),  positive="1")

print(cm_rf)
print(cm_logistic)

weights=c(1, length(which(as.character(df$delinquent)=='current'))/length(which(as.character(df$delinquent)=='delinquent')))
rf<-randomForest(x=df[, c(features)], y=df$delinquent, weights=weights, ntrees=500, keep.forest=TRUE)
arrange(data.frame(feature=rownames(rf$importance), mean_decreased_gini=rf$importance), desc(MeanDecreaseGini))


