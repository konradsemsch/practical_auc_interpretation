
library(tidyverse)
library(ROCR)
library(titanic)

set.seed(123)

# Data preparation --------------------------------------------------------

data <- titanic_train %>% as_tibble()
data <- data %>% select(Survived, Name, Sex, Fare)

n_unique_names <- data %>% select(Name) %>% n_distinct()
stopifnot(nrow(data) == n_unique_names)

data_train <- data %>% sample_frac(0.5)
data_test <- anti_join(data, data_train, by = "Name")

# Train a classifier ------------------------------------------------------

model <- glm(Survived ~ Sex + Fare, 
             data = data_train, 
             family = binomial(link = "logit"))

# saveRDS(model, "model_formula.R")

# Prediction --------------------------------------------------------------

predicted <- predict.glm(model, newdata = data_test, type = "response")
data_test <- data_test %>% mutate(predicted = predicted)

# Calculate AUC -----------------------------------------------------------

prediction_obj <- prediction(predicted, data_test %>% select(Survived))
auc <- performance(prediction_obj, measure = "auc")@y.values[[1]]

# test AUC calculation
data_test_negative <- data_test %>% filter(Survived == 0)
data_test_positive <- data_test %>% filter(Survived == 1)

n_repetitions <- 100000

sample_negatives <- sample(1:nrow(data_test_negative), n_repetitions, replace = TRUE)
sample_positives <- sample(1:nrow(data_test_positive), n_repetitions, replace = TRUE)

scores_negatives <- data_test_negative[sample_negatives, "predicted"]
scores_positives <- data_test_positive[sample_positives, "predicted"]

simulated_auc <- sum(scores_positives > scores_negatives) / n_repetitions

# AUC interpretaion -------------------------------------------------------

# With 83% probability the model assigns assigns a higher score to a randomly 
# chosen survivor than to a randomly chosen non-survivor.