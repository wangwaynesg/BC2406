# ==============================================================================
# Course:       BC2406 ANALYTICS I: VISUAL AND PREDICTIVE TECHNIQUES
# Group:        SEMINAR GROUP 2 TEAM 8
# Authors:      WANG WAYNE (U1920270G),
#               CHEE WEI KIAT COLIN (U1910457D),
#               NEO SAY YIN FLORENCE (U1910346G),
#               DALINI VICKTORIA RAJANDHRAN (U1710158A)
# Data Source:  "Churn_Modelling.csv",
#               "Bank_Personal_Loan_Modelling.csv"
# Packages:     data.table, corrplot, car, caTools, 
#               ggplot2, rpart, rpart.plot, rattle
# Version:      2.3
# Updated:      31-Oct-2020
#===============================================================================

#===============================================================================
# Set the working directory and import required libraries
#===============================================================================

getwd()
setwd("C:/Users/FattyWayne/OneDrive - Nanyang Technological University/NTUBCG/Y2S1/BC2406/Team Assignment and Project")

library(data.table)
library(corrplot)
library(car)
library(caTools)
library(ggplot2)
library(rpart)
library(rpart.plot)
library(rattle)

#===============================================================================
# Churn - Preparing the dataset 
#===============================================================================

# Import the dataset
churn <- fread("Churn_Modelling.csv", stringsAsFactors = TRUE)

# Factorizing some continuous variables into categorical
churn$Exited <- factor(churn$Exited)
churn$IsActiveMember <- factor(churn$IsActiveMember)
churn$HasCrCard <- factor(churn$HasCrCard)

#===============================================================================
# Churn - Cleaning the dataset
#===============================================================================

# Check for any erroneous data
summary(churn)

# Checking for any duplicate records based on CustomerId
sum(duplicated(churn, by = "CustomerId") == TRUE)

# Checking for any NA or missing values
sum(is.na(churn))

#===============================================================================
# Churn - Data Exploration
#===============================================================================

# Check the distribution of Exited customers
ggplot(data = churn, aes(x = Exited, fill = Exited)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = ..count..), vjust = -1) +
  labs(x = "Exited",
       y = "Count",
       title = "Bar chart for Exited") +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.title = element_text(size = 14),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 14))

# Plot a correlation matrix of all variables 
corrplot(cor(data.frame(lapply(churn, function(x) as.numeric(x)))),
         method = "ellipse",
         type = "lower")

# Plot a boxplot of Age and Exited
ggplot(data = churn, aes(x = Exited, y = Age, fill = Exited)) +
  geom_boxplot() +
  labs(x = "Exited",
       y = "Age",
       title = "Boxplot for Age and Exited") +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.title = element_text(size = 14),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 14))


#===============================================================================
# Churn - Train-Test Split
#===============================================================================

set.seed(500)

churn.train <- sample.split(Y = churn$Exited, SplitRatio = 0.7)
churn.trainset <- subset(churn, churn.train == T)
churn.testset <- subset(churn, churn.train == F)

#===============================================================================
# Churn - Logistic Regression Model
#===============================================================================

set.seed(500)

churn.glm1 <- glm(Exited ~ . - CustomerId - RowNumber - Surname, 
                family = binomial, 
                data = churn.trainset)

summary(churn.glm1)

OR.churn.glm1 <- exp(coef(churn.glm1))
OR.churn.glm1

# Odds Ratio of variables like EstimatedSalary is very close to 1.
# Does that mean it is a weak predictor for Exited?

# What if we analyze balance in terms of a $57510 increase instead of a $1 increase?
# What would be the odds ratio?

# options(scipen=100)  # suppress scientific notation by using penalty
summary(churn$EstimatedSalary)
sd(churn$EstimatedSalary)

churn3 <- churn
churn3[, EstimatedSalary57510 := EstimatedSalary/57510]
# a 1 unit increase in EstimatedSalary57510 is a $57510 increase in Estimated Salary

churn3.glm2 <- glm(Exited ~ . - CustomerId - RowNumber - Surname - EstimatedSalary, 
                  family = binomial, 
                  data = churn3)
summary(churn3.glm2)
OR.churn3.glm2 <- exp(coef(churn3.glm2))
OR.churn3.glm2

OR.churn3.glm2 <- exp(confint(churn3.glm2))
OR.churn3.glm2

# EstimatedSalary is indeed a weak predictor for Exited since the OR suggests that
# a $57510 increase in Estimated Salary only increases the Odds of Exited by a
# mere factor of 1.028 

churn.glm3 <- step(churn.glm1)

summary(churn.glm3)

OR.churn.glm3 <- exp(coef(churn.glm3))
OR.churn.glm3 

# Check for multicollinearity problem in our model.
vif(churn.glm3)
# for VIF > 5 or VIF > 10, we may conclude that there is multicollinearity.
# For our case, there is no multicollinearity problem between variables.

churn.glm3$variable.imporatance
#-------------------------------------------------------------------------------
# Churn - Logistic Regression Model (Model Accuracy)
#-------------------------------------------------------------------------------

prob.trainset <- predict(churn.glm3, type = "response", newdata = churn.trainset)
predicted.churn.glm3.trainset <- ifelse(prob.trainset > 0.5, 1, 0)
table(churn.trainset$Exited, predicted.churn.glm3.trainset, deparse.level = 2)

cat("Overall Accuracy for trainset =", mean(churn.trainset$Exited == predicted.churn.glm3.trainset))

prob.testset <- predict(churn.glm3, type = "response", newdata = churn.testset)
predicted.churn.glm3.testset <- ifelse(prob.testset > 0.5, 1, 0)
churn.glm3.confusionMatrix <- table(churn.testset$Exited, predicted.churn.glm3.testset, deparse.level = 2)
churn.glm3.confusionMatrix

cat("Overall Accuracy for testset =", mean(churn.testset$Exited == predicted.churn.glm3.testset))

TP <- churn.glm3.confusionMatrix[4] # True positives
FP <- churn.glm3.confusionMatrix[3] # False positives
TN <- churn.glm3.confusionMatrix[1] # True negatives
FN <- churn.glm3.confusionMatrix[2] # False negatives

TPR <- TP / (TP + FN)
cat("True positive rate = ", round(TPR * 100, 2), "%", sep = "")
FNR <- FN / (TP + FN)
cat("False negative rate = ", round(FNR * 100, 2), "%", sep = "")

FPR <- FP / (FP + TN)
cat("False positive rate = ", round(FPR * 100, 2), "%", sep = "")
TNR <- TN / (FP + TN)
cat("True negative rate = ", round(TNR * 100, 2), "%", sep = "")





#===============================================================================
# Churn - CART Model
#===============================================================================
set.seed(500)

churn.cart1 <- rpart(Exited ~ CreditScore + Geography + Gender + Age
                     + Tenure + Balance + NumOfProducts + HasCrCard 
                     + IsActiveMember + EstimatedSalary,
                  data = churn.trainset,
                  method = "class",
                  control = rpart.control(minsplit = 2, cp = 0))

# Code for plotting maximal Tree
# rpart.plot(churn.cart1, nn= T, main = "Maximal Tree in Churn_Modelling.csv")

printcp(churn.cart1)
plotcp(churn.cart1, main = "Subtrees in Churn_Modelling.csv")

# Compute min CVerror + 1SE in maximal tree cart
CVerror.cap <- churn.cart1$cptable[which.min(churn.cart1$cptable[,"xerror"]), "xerror"] + 
  churn.cart1$cptable[which.min(churn.cart1$cptable[,"xerror"]), "xstd"]

# Find the optimal CP region whose CV error is just below CVerror.cap in maximal tree churn.cart1.
i <- 1; j<- 4
while (churn.cart1$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}

# Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
cp.opt = ifelse(i > 1, sqrt(churn.cart1$cptable[i,1] * churn.cart1$cptable[i-1,1]), 1)
cp.opt

# Prune the max tree using cp.opt = 0.002688869

churn.cart2 <- prune(churn.cart1, cp = cp.opt)
printcp(churn.cart2, digits = 3)
print(churn.cart2)

options(scipen = 100)
rpart.plot(churn.cart2, nn = T, main = "Optimal Tree in Churn_Modelling.csv")
fancyRpartPlot(churn.cart2, main="Optimal Tree in Churn_Modelling.csv")

churn.cart2$variable.importance

round(churn.cart2$variable.importance/ sum(churn.cart2$variable.importance) * 100)

summary(churn.cart2)

#-------------------------------------------------------------------------------
# Churn - CART Model (Model Accuracy)
#-------------------------------------------------------------------------------

predicted.churn.cart2.trainset <- predict(churn.cart2, type = "class", newdata = churn.trainset)
table(churn.trainset$Exited, predicted.churn.cart2.trainset, deparse.level = 2)

cat("Overall Accuracy for trainset =", mean(churn.trainset$Exited == predicted.churn.cart2.trainset))

predicted.churn.cart2.testset <- predict(churn.cart2, type = "class", newdata = churn.testset)
churn.cart2.confusionMatrix <- table(churn.testset$Exited, predicted.churn.cart2.testset, deparse.level = 2)
churn.cart2.confusionMatrix

cat("Overall Accuracy for testset =", mean(churn.testset$Exited == predicted.churn.cart2.testset))

TP <- churn.cart2.confusionMatrix[4] # True positives
FP <- churn.cart2.confusionMatrix[3] # False positives
TN <- churn.cart2.confusionMatrix[1] # True negatives
FN <- churn.cart2.confusionMatrix[2] # False negatives

TPR <- TP / (TP + FN)
cat("True positive rate = ", round(TPR * 100, 2), "%", sep = "")
FNR <- FN / (TP + FN)
cat("False negative rate = ", round(FNR * 100, 2), "%", sep = "")

FPR <- FP / (FP + TN)
cat("False positive rate = ", round(FPR * 100, 2), "%", sep = "")
TNR <- TN / (FP + TN)
cat("True negative rate = ", round(TNR * 100, 2), "%", sep = "")














#===============================================================================
# Personal Loan - Preparing the dataset 
#===============================================================================

# Import the dataset
loan <- fread("Bank_Personal_Loan_Modelling.csv", stringsAsFactors = TRUE)
str(loan)

# Multiply some variables by thousands according to data description
loan$Income <- loan$Income * 1000
loan$CCAvg <- loan$CCAvg * 1000
loan$Mortgage <- loan$Mortgage * 1000

# Factorizing some continuous variables into categorical
loan$Education <- factor(loan$Education,
                         levels = c(1, 2, 3),
                         labels = c("Undergrad", "Graduate", "Advanced/Professional"))
loan$`Personal Loan` <- factor(loan$`Personal Loan`)
loan$`Securities Account` <- factor(loan$`Securities Account`)
loan$`CD Account` <- factor(loan$`CD Account`)
loan$Online <- factor(loan$Online)
loan$CreditCard <- factor(loan$CreditCard)


#===============================================================================
# Personal Loan - Cleaning the dataset
#===============================================================================

# Checking for any NA or missing values
sum(is.na(loan))

# Checking for any duplicate records based on CustomerId
sum(duplicated(loan, by = "ID") == TRUE)

# Check for any erroneous data
summary(loan)

# Perform cleaning for rows where Experience < 0
head(loan[Experience < 0])
dim(loan[Experience < 0])

cor(loan$Experience, loan$Age)
# Since Age is highly correlated with Experience, we will use Age to estimate
# the Experience for rows with erroneous Experience.
summary(loan[Experience < 0]$Age)
for (i in 23:29) {
  x <- median(loan[Experience >= 0 & Age == i]$Experience)
  if (x < 0 | is.na(x)) {
    x <- 0
  }
  loan[Experience < 0 & Age == i, Experience:= x]
}
# Check if there are still any rows where Experience < 0
loan[Experience < 0]

# Check for any more erroneous data
summary(loan)

#===============================================================================
# Personal Loan - Data Exploration
#===============================================================================

# Check the distribution of Exited customers
ggplot(data = loan, aes(x = `Personal Loan`, fill = `Personal Loan`)) +
  geom_bar() +
  geom_text(stat = "count", aes(label = ..count..), vjust = -1) +
  labs(x = "Personal Loan",
       y = "Count",
       title = "Bar chart for Personal Loan") +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.title = element_text(size = 14),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 14))

# Plot a correlation matrix of all variables 
corrplot(cor(data.frame(lapply(loan, function(x) as.numeric(x)))),
         method = "ellipse",
         type = "lower")

# Plot a boxplot of Income and Personal Loan
ggplot(data = loan, aes(x = `Personal Loan`, y = Income, fill = `Personal Loan`)) +
  geom_boxplot() +
  labs(x = "Personal Loan",
       y = "Income",
       title = "Boxplot for Income and Personal Loan") +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.title = element_text(size = 14),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 14))

# Plot a boxplot of CCAvg and Personal Loan
ggplot(data = loan, aes(x = `Personal Loan`, y = CCAvg, fill = `Personal Loan`)) +
  geom_boxplot() +
  labs(x = "Personal Loan",
       y = "CCAvg",
       title = "Boxplot for CCAvg and Personal Loan") +
  theme(plot.title = element_text(hjust = 0.5, size = 14),
        axis.title = element_text(size = 14),
        legend.title = element_text(size = 14),
        legend.text = element_text(size = 14))


#===============================================================================
# Personal Loan - Train-Test Split
#===============================================================================

set.seed(500)

loan.train <- sample.split(Y = loan$`Personal Loan`, SplitRatio = 0.7)
loan.trainset <- subset(loan, loan.train == T)
loan.testset <- subset(loan, loan.train == F)

#===============================================================================
# Personal Loan - Logistic Regression Model
#===============================================================================

set.seed(500)

loan.glm1 <- glm(`Personal Loan` ~ . - ID - `ZIP Code`, 
                  family = binomial, 
                  data = loan.trainset)

summary(loan.glm1)

OR.loan.glm1 <- exp(coef(loan.glm1))
OR.loan.glm1

loan.glm2 <- step(loan.glm1)
summary(loan.glm2)

# Further remove insignificant variables
loan.glm3 <- glm(`Personal Loan` ~ . - ID - `ZIP Code`
                 - Experience - Age - `Securities Account` - Mortgage, 
                 family = binomial, 
                 data = loan.trainset)

summary(loan.glm3)

OR.loan.glm3 <- exp(coef(loan.glm3))
OR.loan.glm3

# Check whether Income or CCAvg is a weak predictor of Personal Loan since their
# OR is very close to 1.
1.000060741929^sd(loan$Income)
1.000148510483^sd(loan$CCAvg)
# Interpretation: An increase in income by 1 s.d. increases the odds of personal 
# loan acceptance by a factor of 16. Therefore, Income is not a weak predictor.
# Likewise for CCAvg, the increase in CCAvg by 1 s.d. increases the odds of
# personal loan acceptance by a factor of 1.29. Therefore, CCAvg is not a 
# weak predictor.


# Check for multicollinearity problem in our model.
vif(loan.glm3)
# for VIF > 5 or VIF > 10, we may conclude that there is multicollinearity.
# For our case, there is no multicollinearity problem between variables.

#-------------------------------------------------------------------------------
# Personal Loan - Logistic Regression Model (Model Accuracy)
#-------------------------------------------------------------------------------

prob.trainset <- predict(loan.glm3, type = "response", newdata = loan.trainset)
predicted.loan.glm3.trainset <- ifelse(prob.trainset > 0.5, 1, 0)
table(loan.trainset$`Personal Loan`, predicted.loan.glm3.trainset, deparse.level = 2)

cat("Overall Accuracy for trainset =", mean(loan.trainset$`Personal Loan` == predicted.loan.glm3.trainset))

prob.testset <- predict(loan.glm3, type = "response", newdata = loan.testset)
predicted.loan.glm3.testset <- ifelse(prob.testset > 0.5, 1, 0)
loan.glm3.confusionMatrix <- table(loan.testset$`Personal Loan`, predicted.loan.glm3.testset, deparse.level = 2)
loan.glm3.confusionMatrix

cat("Overall Accuracy for testset =", mean(loan.testset$`Personal Loan` == predicted.loan.glm3.testset))

TP <- loan.glm3.confusionMatrix[4] # True positives
FP <- loan.glm3.confusionMatrix[3] # False positives
TN <- loan.glm3.confusionMatrix[1] # True negatives
FN <- loan.glm3.confusionMatrix[2] # False negatives

TPR <- TP / (TP + FN)
cat("True positive rate = ", round(TPR * 100, 2), "%", sep = "")
FNR <- FN / (TP + FN)
cat("False negative rate = ", round(FNR * 100, 2), "%", sep = "")

FPR <- FP / (FP + TN)
cat("False positive rate = ", round(FPR * 100, 2), "%", sep = "")
TNR <- TN / (FP + TN)
cat("True negative rate = ", round(TNR * 100, 2), "%", sep = "")

#===============================================================================
# Personal Loan - CART Model
#===============================================================================
set.seed(500)

loan.cart1 <- rpart(`Personal Loan` ~ . - ID - `ZIP Code`,
                     data = loan.trainset,
                     method = "class",
                     control = rpart.control(minsplit = 2, cp = 0))

# Code for plotting maximal tree
# rpart.plot(loan.cart1, nn= T, main = "Maximal Tree in Churn_Modelling.csv")

printcp(loan.cart1)
plotcp(loan.cart1, main = "Subtrees in Churn_Modelling.csv")

# Compute min CVerror + 1SE in maximal tree cart
CVerror.cap <- loan.cart1$cptable[which.min(loan.cart1$cptable[,"xerror"]), "xerror"] + 
  loan.cart1$cptable[which.min(loan.cart1$cptable[,"xerror"]), "xstd"]

# Find the optimal CP region whose CV error is just below CVerror.cap in maximal tree loan.cart1.
i <- 1; j<- 4
while (loan.cart1$cptable[i,j] > CVerror.cap) {
  i <- i + 1
}

# Get geometric mean of the two identified CP values in the optimal region if optimal tree has at least one split.
cp.opt = ifelse(i > 1, sqrt(loan.cart1$cptable[i,1] * loan.cart1$cptable[i-1,1]), 1)
cp.opt

# Prune the max tree using cp.opt = 0.002688869

loan.cart2 <- prune(loan.cart1, cp = cp.opt)
printcp(loan.cart2, digits = 3)
print(loan.cart2)

options(scipen = 100)
rpart.plot(loan.cart2, nn = T, main = "Optimal Tree in Bank_Personal_Loan_Modelling.csv")
fancyRpartPlot(loan.cart2, main="Optimal Tree in Bank_Personal_Loan_Modelling.csv")

loan.cart2$variable.importance

round(loan.cart2$variable.importance/ sum(loan.cart2$variable.importance) * 100)

summary(loan.cart2)

#-------------------------------------------------------------------------------
# Personal Loan - CART Model (Model Accuracy)
#-------------------------------------------------------------------------------

predicted.loan.cart2.trainset <- predict(loan.cart2, type = "class", newdata = loan.trainset)
table(loan.trainset$`Personal Loan`, predicted.loan.cart2.trainset, deparse.level = 2)

cat("Overall Accuracy for trainset =", mean(loan.trainset$`Personal Loan` == predicted.loan.cart2.trainset))

predicted.loan.cart2.testset <- predict(loan.cart2, type = "class", newdata = loan.testset)
loan.cart2.confusionMatrix<- table(loan.testset$`Personal Loan`, predicted.loan.cart2.testset, deparse.level = 2)
loan.cart2.confusionMatrix

cat("Overall Accuracy for testset =", mean(loan.testset$`Personal Loan` == predicted.loan.cart2.testset))

TP <- loan.cart2.confusionMatrix[4] # True positives
FP <- loan.cart2.confusionMatrix[3] # False positives
TN <- loan.cart2.confusionMatrix[1] # True negatives
FN <- loan.cart2.confusionMatrix[2] # False negatives

TPR <- TP / (TP + FN)
cat("True positive rate = ", round(TPR * 100, 2), "%", sep = "")
FNR <- FN / (TP + FN)
cat("False negative rate = ", round(FNR * 100, 2), "%", sep = "")

FPR <- FP / (FP + TN)
cat("False positive rate = ", round(FPR * 100, 2), "%", sep = "")
TNR <- TN / (FP + TN)
cat("True negative rate = ", round(TNR * 100, 2), "%", sep = "")

