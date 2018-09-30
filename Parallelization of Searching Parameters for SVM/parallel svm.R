#===Loading require libraries===#

library("parallel", lib.loc="C:/Program Files/R/R-3.4.2/library")
library("doParallel", lib.loc="~/R/win-library/3.4")
library("foreach", lib.loc="~/R/win-library/3.4")
library("iterators", lib.loc="~/R/win-library/3.4")
library("pROC", lib.loc="~/R/win-library/3.4")
library("caret", lib.loc="~/R/win-library/3.4")
library("e1071", lib.loc="~/R/win-library/3.4")


#===Initializing and assigning number of processing core/s===# 

package <- c('foreach', 'doParallel')
lapply(package, require, character.only = T)
registerDoParallel(cores = 4)


#===Dataset loading and observations===#

Credit_data <- read.csv("E:/SAMIUL RYERSON/fall2017/Algorithm2017/Project/credit_count.csv")
View(Credit_data)
summary(Credit_data)


#===Data preprocessing===#
Clean_data <- Credit_data[Credit_data$CARDHLDR == 1, ]
View(Clean_data)
summary(Clean_data)
dt.hd <- paste("AGE + ACADMOS + ADEPCNT + MAJORDRG + MINORDRG + OWNRENT + INCOME + SELFEMPL + INCPER + EXP_INC")
fn1 <- as.formula(paste("as.factor(DEFAULT) ~ ", dt.hd))


#===Data split into the number of K-Folds===#

set.seed(666)
Clean_data$fold <- caret::createFolds(1:nrow(Clean_data), k = 8, list = FALSE)


#===List of Support Vector Machine Parameters===#

newcost <- c(10, 100)
newgamma <- c(1, 2)
parameters <- expand.grid(newcost = newcost, newgamma = newgamma)


#===Passing parameter values===#

Result1 <- foreach(m = 1:nrow(parameters), .combine = rbind) %do% {
cost <- parameters[m, ]$newcost
gamma <- parameters[m, ]$newgamma

  
#===Validation testing of K-Fold for a given K ===#
  
output <- foreach(n = 1:max(Clean_data$fold), .combine = rbind, .inorder = FALSE) %dopar% {
training <- Clean_data[Clean_data$fold != n, ]
testing <- Clean_data[Clean_data$fold == n, ]
modeling <- e1071::svm(fn1, data = training, type = "C-classification", kernel = "radial", newcost = cost, newgamma = gamma, probability = TRUE)
predicting <- predict(modeling, testing, decision.values = TRUE, probability = TRUE)
data.frame(x = testing$DEFAULT, prob = attributes(predicting)$probabilities[, 2])
  }


#=== Performance evaluation ===#
time1 <- Sys.time() # To fetch the starting time
Perf_curve <- pROC::roc(as.factor(output$x), output$prob) 
data.frame(parameters[m, ], Perf_curve = Perf_curve$auc[1])
}
execution_time <- Sys.time() - time1 # execution time calculation
print(execution_time) # shows the execution time

