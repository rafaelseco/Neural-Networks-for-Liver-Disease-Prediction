#install.packages("caret")
#install.packages("nnet")
#install.packages("neuralnet")

# Importing Libraries 
library(readr)
library(neuralnet)
library(nnet)
library(dplyr)
library(pROC)
library(caret)
library(ggplot2)
library(lattice)
library(elmNNRcpp)


#_______________________________ IMPORT DATA ___________________________________

data <- read_csv("___________your path_____________\\Liver Patient Dataset (LPD)_train.csv")

#_______________________________ PREPROCESSING _________________________________

#number of missing values for each column
MissingData <- colSums(is.na(data))
print(MissingData)

#elimination of rows with at least 2 missing values
auxiliary <- apply(data,1,function(x)sum(is.na(x)))
data <- data[which(auxiliary<2),]

MissingData <- colSums(is.na(data))
print(MissingData)

#Creating the variable 'Class'
data["Class"] <- ifelse(data$Result == 2,"not_patient","patient")
data["Class"] <- factor(data$Class)
data$Result <- ifelse(data$Result == 2, 0, 1)
# 1 is patient and 2 is not patient

#Creating two variabels: 'Male' and 'Female'
data["Female"] <- ifelse(data$gender == "Female",1,0)
data["Male"] <- ifelse(data$gender == "Male",1,0)

#Elimination of outliers on the variable 'Sgot'
data <- data[-which(data$Sgot >2000),] ## 514 casos agora

#Assigning 'P' all the patients and 'NP' all the non-patients
P <- data[data$Class == "patient",]
NP <- data[data$Class == "not_patient",]


#Changing NA values for the median of each numeric variable in each subset
for(i in c('Tot_Bilirubin','age','Dir_Bilirubin','Alkphos','Sgpt','Sgot','Tot_Protiens','Albumin','ag_Ratio')){
  
  P[is.na(P[[i]]), i] <- median(P[[i]], na.rm = TRUE)
  NP[is.na(NP[[i]]), i] <- median(NP[[i]], na.rm = TRUE)
}

#Finding the mode for the categorical variable 'gender'
P[is.na(P$gender),'gender'] <- mode(P$gender[-which(is.na(P$gender))])
NP[is.na(NP$gender),'gender'] <- mode(NP$gender[-which(is.na(NP$gender))])

#Assigning binary values for gender
P["Female"] <- ifelse(P$gender == "Female",1,0)
P["Male"] <- ifelse(P$gender == "Male",1,0)
NP["Female"] <- ifelse(NP$gender == "Female",1,0)
NP["Male"] <- ifelse(NP$gender == "Male",1,0)

#Shuffle the patients' observations
P <- P[sample(1:nrow(P)),]


#Binding the subsets in a balanced way
data <- rbind(P[1:8369,],NP)

#Total: 29315 observations
#Patients: 20946 observations
#Not Patients: 8369 observations


#Normalizing the dataset with a min-max scalling
normalize<-function(x){
  return ( (x-min(x))/(max(x)-min(x)))
}
data_norm<-as.data.frame(lapply(data[,-c(2,11:14)],FUN=normalize))
TrainingSet <- data.frame(data_norm,data[,c(13,14,12)]) 

#Dividing the Training Set according to its 'Class' and then shuffle it 
TrainingSet_patient <- TrainingSet[TrainingSet$Class == "patient",]
TrainingSet_not_patient <- TrainingSet[TrainingSet$Class == "not_patient",]
TrainingSet_patient <- TrainingSet_patient[sample(1:nrow(TrainingSet_patient)),]
TrainingSet_not_patient <- TrainingSet_not_patient[sample(1:nrow(TrainingSet_not_patient)),]

#Division of Training Set (80% of all data) and Testing Set (20% of all data)
TestingSet <- rbind(TrainingSet_patient[6695:8368,],TrainingSet_not_patient[6695:8368,])
TrainingSet_patient <- TrainingSet_patient[-c(6695:8369),]
TrainingSet_not_patient <- TrainingSet_not_patient[-c(6695:8369),]



#_____________________________ MODELS' PREPARATION ___________________________________

# Formula to use when calling the models
formula_nn <- Class~age+Tot_Bilirubin+Dir_Bilirubin+Alkphos+Sgpt+Sgot+Tot_Protiens+Albumin+ag_Ratio


for(j in 1:3){
  
  #K-Fold Cross-Validation plus initialization of variables to store error, accuracy and a vector to store the models
  k <- 10 
  CV_error <- NULL
  accuracy <- NULL
  AUXvector <- NULL
  metrics <- NULL
  ValidationProcess <- NULL
  
  
  #Progress Bar
  library(plyr) 
  pbar <- create_progress_bar('text')
  pbar$init(k)
  
  #Process of Cross-Validation
  for (i in 1:k){
    
    #Sampling the data in each iteration
    TrainingSet_patient <- TrainingSet_patient[sample(1:nrow(TrainingSet_patient)),]
    TrainingSet_not_patient <- TrainingSet_not_patient[sample(1:nrow(TrainingSet_not_patient)),]
    
    #Binding the two subsets
    CV_data <-rbind(TrainingSet_not_patient,TrainingSet_patient)
    CV_data["Binary"] <- ifelse(CV_data$Class == "not_patient",0,1)
    
    #Setting Training and Validation subsets  
    index <- sample(1:nrow(CV_data),round(0.9*nrow(CV_data)))
    Train <- CV_data[index,]
    Validation <- CV_data[-index,]
    
    FunctionNNET_NEURALNET<-function(model,Validation,CV_data,CV_error,accuracy){
      
      prediction <- predict(model,Validation[,1:11],type='raw')
      k <- ifelse(j==1,2,1)
      predicted_size_category <- ifelse(prediction[,k] < 0.5,"not_patient","patient")
      
      prediction <- prediction*(max(CV_data$Binary)-min(CV_data$Binary))+min(CV_data$Binary)
      
      results <- (Validation$Binary)*(max(CV_data$Binary)-min(CV_data$Binary))+min(CV_data$Binary)
      
      CV_error[i] <<- sum((results - prediction)^2)/nrow(Validation)
      accuracy[i] <<- mean(predicted_size_category==Validation$Class)
      
      #return(list(a=accuracy,b=CV_error))
    }
    
    FunctionELM<-function(model,Validation,CV_data,CV_error,accuracy){
      
      Validation1 <- as.matrix(Validation[,1:11])
      prediction <- elm_predict(model,Validation1)
      
      predicted_size_category <- ifelse(prediction[,2] < 0.5,"not_patient","patient")
      
      prediction <- prediction*(max(CV_data$Binary)-min(CV_data$Binary))+min(CV_data$Binary)
      
      results <- (Validation$Binary)*(max(CV_data$Binary)-min(CV_data$Binary))+min(CV_data$Binary)
      
      CV_error[i] <<- sum((results - prediction)^2)/nrow(Validation)
      accuracy[i] <<- mean(predicted_size_category==Validation$Class)
      
      #return(list(a=accuracy,b=CV_error))
    }
    
    
    #Setting up all the models
    size <-10
    model <- switch(j,
                    neuralnet(formula_nn,data=Train,hidden=1,stepmax = 1e7),
                    nnet(formula_nn,data=Train,size=10),
                    elm(formula_nn,data=Train,nhid=2000,actfun="satlins",verbose=FALSE))
    
    #plotnet(model)
    #prediction <- predict(model,Validation[,1:11],type='raw')
    AUXvector[i] <- list(model)
    
    ValidationProcess <- switch(j,
                                FunctionNNET_NEURALNET(model,Validation,CV_data,CV_error,accuracy),
                                FunctionNNET_NEURALNET(model,Validation,CV_data,CV_error,accuracy),
                                FunctionELM(model,Validation,CV_data,CV_error,accuracy))
    
    
    pbar$step()
  }
  
  cat("\n")
  string <- switch(j,
                   "RESULTS FOR neuralnet:",
                   "RESULTS FOR nnet:",
                   "RESULTS FOR Extreme Learning Machine:")
  
  print(string)
  
  
  
  print("Vector containing the error values:")
  print(CV_error)
  
  print("Vector containing the accuracy values:")
  print(accuracy)
  
  print("Mean accuracy of this network: ")
  print(mean(accuracy))
  
  print("Mean error of this network: ")
  print(mean(CV_error))
  
  k <- ifelse(j==2,1,2)
  
  if(j==3){
    TestingSet1 <- as.matrix(TestingSet[,1:11])
    model <- AUXvector[[which.max(accuracy)]]
    TestingSet_final <- elm_predict(model,TestingSet1,normalize=FALSE)
    predicted_size_category <- ifelse(TestingSet_final[,2] < 0.5,"not_patient","patient")
    print("The accuracy shown by the best validation phase model is:")
    print(mean(predicted_size_category==TestingSet$Class))
    
    
    model <- AUXvector[[which.min(accuracy)]]
    TestingSet_final <- elm_predict(model,TestingSet1,normalize=FALSE)
    predicted_size_category <- ifelse(TestingSet_final[,2] < 0.5,"not_patient","patient")
    print("The accuracy shown by the worst validation phase model is:")
    print(mean(predicted_size_category==TestingSet$Class))
    
  }else{
    
    model <- AUXvector[[which.max(accuracy)]]
    TestingSet_final <- predict(model,TestingSet[,1:11],type='raw')
    predicted_size_category <- ifelse(TestingSet_final[,k] < 0.5,"not_patient","patient")
    print("The accuracy shown by the best validation phase model is:")
    print(mean(predicted_size_category==TestingSet$Class)) 
    
    
    model <- AUXvector[[which.min(accuracy)]]
    TestingSet_final <- predict(model,TestingSet[,1:11],type='raw')
    predicted_size_category <- ifelse(TestingSet_final[,k] < 0.5,"not_patient","patient")
    print("The accuracy shown by the worst validation phase model is:")
    print(mean(predicted_size_category==TestingSet$Class)) 
    
  }
  
  
}

#TestingSet <- as.data.frame(TestingSet)
trctrl <- trainControl(method = "cv", number = 10)

cat("\n")
print("RESULTS FOR Support Vector Machine:")

svmModel <- train(formula_nn, data = CV_data, method = "svmLinear",
                  trControl=trctrl,tuneLength = 2,preProc = c("center","scale"))

model_results <- predict(svmModel,newdata=TestingSet[,1:11]) #we use predict instead of compute

cat("\n")
print(confusionMatrix(model_results, TestingSet$Class ))


cat("\n")
print("RESULTS FOR Gradient Boosting Machine:")
gbmModel <- train(formula_nn, data = CV_data, method = "gbm",
                  trControl=trctrl,
                  verbose=F)

model_results <- predict(gbmModel,newdata=TestingSet[,1:11]) #we use predict instead of compute

cat("\n")
print(confusionMatrix(model_results, TestingSet$Class ))