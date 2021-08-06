#########################
#########################
#### Clear Workspace ####
#########################
#########################

rm(list = ls()) 
# clear global environment to remove all loaded data sets, functions and so on.

###################
###################
#### Libraries ####
###################
###################

library(easypackages) # enables the libraries function
suppressPackageStartupMessages(
  libraries("gbm",
            "dplyr", # for data wrangling
            "h2o", # for java based implementation of GBM variants
            "xgboost", # for fitting extreme gradient boosting))
            "purrr",
            "recipes"
            ))
            
###############################
###############################
#### Set Working Directory ####
###############################
###############################

setwd("C:/R Portfolio/Gradient Boosting/Data")

bikes <- read.csv("bikes.csv")
str(bikes)
glimpse(bikes)
summary(bikes)

# Convert categorical variables into factors

bikes$season <- as.factor(bikes$season)
bikes$holiday <- as.factor(bikes$holiday)
bikes$weekday <- as.factor(bikes$weekday)
bikes$weather <- as.factor(bikes$weather)

# Convert numeric variables into integers

bikes$temperature <- as.integer(bikes$temperature)
bikes$realfeel <- as.integer(bikes$realfeel)
bikes$windspeed <- as.integer(bikes$windspeed)

levels(bikes$season) <- c("Spring", "Summer", "Autumn", "Winter")

# remove column named date
bikes <- bikes %>% select(-date)

###############################
###############################
# Training and Test Data Sets #
###############################
###############################

set.seed(1234) # changing this alters the make up of the data set, which affects predictive outputs

ind <- sample(2, nrow(bikes), replace = T, prob = c(0.8, 0.2))
train <- bikes[ind == 1, ]
test <- bikes[ind == 2, ]

#############
#############
# Basic GBM #
#############
#############       

set.seed(123)
bikes_gbml <- gbm(
  rentals ~.,
  train,
  distribution = "gaussian", # sse loss function
  n.trees = 5000,
  shrinkage = 0.1,
  interaction.depth = 3,
  n.minobsinnode = 10,
  cv.folds = 10
)

# Find index for number of trees with minimum cv error

best <- which.min(bikes_gbml$cv.error)            
best 

# Get MSE and compute RMSE

sqrt(bikes_gbml$cv.error[best])

# results indicate cross validated SS of 1241.965 which was achieved with 51 trees

# Plot error curve #

gbm.perf(bikes_gbml, method = "cv")

# General tuning strategy #

# 1 choos relatively high learning rate. generally, the default value of 0.1 works, but somewhere between 0.05 and 0.2 should work across a wide range of problems. 
# 2. Determine the optimum number of trees for this learning rate
# 3. Fix tree hyper-parameters and tune learning rate and access speed versus performance
# 4. Tune specific parameters for decided learning rate
# 5. Once tree specific parameters have been found, lower the learning rate to assess for any improvements in accuracy
# 6. Use final hyper-parameter settings and increase CV procedures to get more robust estimates. Often, the above steps are performed with a simple validation procedure of 5 fold CV due to computational constraints. If you used k-fold CV throughout steps 1 to five, then this step is not necessary.

###############
###############
# Grid Search #
###############
###############

######################
# Create grid search #
######################

hyper_grid <- expand.grid(
  learning.rate = c(0.3, 0.1, 0.05, 0.01, 0.005),
  RMSE = NA,
  trees = NA,
  time = NA)

#######################
# Execute grid search #
#######################

for(i in seq_len(nrow(hyper_grid))) {

###########
# Fit gbm #
###########
  
set.seed(123)
train_time <- system.time({
      m <- gbm(rentals ~.,
               train,
               distribution = "gaussian", # sse loss function
               n.trees = 5000,
               shrinkage = hyper_grid$learning.rate[i],
               interaction.depth = 3,
               n.minobsinnode = 10,
               cv.folds = 10
               )
    })

# add SSE, trees and training time to results
    
hyper_grid$RMSE[i] <- sqrt(min(m$cv.error))
hyper_grid$trees[i] <- which.min(m$cv.error)
hyper_grid$time[i] <- train_time[["elapsed"]]
}

# Results

arrange(hyper_grid, RMSE)

# The lowest RMSE was 107 trees and learning rate of 0.05

# Search Grid

hyper_grid <- expand.grid(
  n.trees = 107,
  shrinkage = 0.05,
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 10, 15)
)

# Create model fit function #

model_fit <- function(n.trees, shrinkage, interaction.depth, n.minobsinnode) {
  set.seed(123)
  m <- gbm(
    rentals ~.,
    train,
    distribution = "gaussian",
    n.trees = n.trees,
    shrinkage = shrinkage,
    interaction.depth = interaction.depth,
    cv.folds = 10
  )
  
# Compute RMSE

sqrt(min(m$cv.error))

}

# Perform Search grid with functional programming

hyper_grid$rmse <- purrr::pmap_dbl(
  hyper_grid,
  ~ model_fit(
    n.trees = ..1,
    shrinkage = ..2,
    interaction.depth = ..3,
    n.minobsinnode = ..4
  )
)

# Results

arrange(hyper_grid, rmse)

# The results indicate that the lowest rmse was evident with interaction depth = 3 and n.minobsinnode = 5

###################
###################
# Stochastic GBMs #
###################
###################

#############
# Modelling #
#############

# Refined hyperparameter grid
# sample_rate: row_subsampling
# col_sample_rate: col subsampling for each split
# col_sample_rate_per_tree: col_subsampling for each tree

hyper_grid <- list(
  sample_rate = c(0.5, 0.75, 1),
  col_sample_rate = c(0.5, 0.75, 1),
  col_sample_rate_per_tree = c(0.5, 0.75, 1)
)

###############################
# Random Grid Search Strategy #
###############################

search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "mse",
  stopping_tolerance = 0.001,
  stopping_rounds = 10, 
  max_runtime_secs = 60*60
)

#######################
# Perform Grid Search #
#######################

Sys.setenv(JAVA_HOME = "C:/Program Files/Java/jdk-11.0.12") # your own path of Java SE installed
library(jsonlite)
h2o.init()

train_h2o <- as.h2o(train)
response <- "rentals"
predictors <- setdiff(colnames(train), response)

grid <- h2o.grid(
  algorithm = "gbm",
  grid_id = "gbm_grid",
  x = predictors,
  y = response,
  training_frame = train_h2o,
  hyper_params = hyper_grid,
  ntrees = 5000,
  learn_rate = 0.05,
  max_depth = 5, 
  min_rows = 5, 
  nfolds = 5,
  stopping_rounds = 10, 
  search_criteria = search_criteria,
  seed = 123
 
)

###########
# Results #
###########

# collect and sort by chosen model performance metric

grid_perf <- h2o.getGrid(
  grid_id = "gbm_grid",
  sort_by = "mse",
  decreasing = F
)

grid_perf

# The best sampling values are on the lower end (0.5 - 0.75)
# A further grid search could evaluate additional values in this lower range by changing the values in the hyper_grid object above

#########################################
# Performance Metrics on the Best Model #
#########################################

# get model id for the top model chosen by cross validation error

best_model_id <- grid_perf@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)

# get performance metrics on the best model

h2o.performance(best_model, xval = T)

###########
###########
# XGBoost #
###########
###########

xgb_prep <- recipe(rentals ~., train) %>%
  step_integer(all_nominal()) %>%
  prep(training = train, retain = T) %>%
  juice()

X <- as.matrix(xgb_prep[setdiff(names(xgb_prep), "rentals")])
Y <- xgb_prep$rentals

set.seed(123)
bikes_xgb <- xgb.cv(
  X,
  label = Y,
  nrounds = 6000,
  objective = "reg:linear",
  nfold = 10, 
  params = list(
    eta = 0.01,
    max_depth = 3,
    min_child_weight = 3,
    subsample = 0.5,
    colsample_bytree = 0.5),
  verbose = 0
  )

# Minimum test CV RMSE

min(bikes_xgb$evaluation_log$test_rmse_mean)

# Hyper-parameter grid

hyper_grid <- expand.grid(
  eta = 0.01,
  max_depth = 3,
  min_child_weight = 3,
  subsample = 0.5,
  colsample_bytree = 0.5,
  gamma = c(1, 10, 100, 1000),
  lambda = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
  alpha = c(0, 1e-2, 0.1, 1, 100, 1000, 10000),
  rmse = 0, # place to send RMSE results
  trees = 0 # place to send required number of trees
)

###############
# Grid Search #
###############

for(i in seq_len(nrow(hyper_grid))) { 
  set.seed(123)
  m <- xgb.cv(
    X,
    label = Y,
    nrounds = 4000,
    objective = "reg:linear",
    early_stopping_rounds = 50,
    nfold = 10, 
    verbose = 10,
    params = list(
      eta = hyper_grid$eta[i],
      max_depth = hyper_grid$max_depth[i],
      min_child_weight = hyper_grid$min_child_weight[i],
      subsample = hyper_grid$subsample[i],
      colsample_bytree = hyper_grid$colsample_bytree[i],
      gamma = hyper_grid$gamma[i],
      lambda = hyper_grid$lambda[i],
      alpha = hyper_grid$alpha[i]
    )
    
  )
  
  hyper_grid$rmse[i] <- min(m$evaluation_log$test_rmse_mean)
  hyper_grid$trees[i] <- m$best_iteration
  
}

###########
# Results #
###########

hyper_grid %>%
  filter(rmse > 0) %>%
  arrange(rmse) %>%
  glimpse()

##########################
# Optimal Parameter List #
##########################

params <- list(
eta = 0.01,
max_depth = 3,
min_child_weight = 3,
subsample = 0.5,
colsample_bytree = 0.5
)

#####################
# Train Final Model #
#####################

xgb.fit.final <- xgboost(
  params = params,
  X,
  label = Y,
  nrounds = 3944,
  objective = "reg:squarederror",
  verbose = 0
)

##########################
# Feature Interpretation #
##########################

vip::vip(xgb.fit.final)

# Temperature an humidity are the two most important variables in predicting bike rentals