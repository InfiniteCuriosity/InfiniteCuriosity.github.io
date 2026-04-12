<b><h1>Predicting Used Car Prices With Data Science</h1></b>

Russ Conte<br>
Monday April 13, 2026<br>

<h4>Outline</h4>
1. Introduction and Statement of the Problem (What Does the Customer Actually Want?)<br>
2. Statement Regarding No Use of AI<br>
3. Literature Review (How Big is the Problem?)<br>
4. The Data Set (including exploratory data analysis)<br>
5. Model Building (18 individual models, 14 ensembles of models)<br>
6. Model summaries: Tables and Plots<br>
7. Highest Accuracy<br>
8. Strongest Predictor<br>
9. Strongest Evidence Based Recommendations<br>
10. Summary: The Story in this Data Set<br>
11. Notes<br>
12. References<br>

<h4>1. Introduction and Statement of the Problem</h4>

<h4>2. Statement regarding No Use of AI</h4>
No Artificial Intelligence systems (AI) were used in any part of the process. This analysis excludes all commercial AI systems, large language models, coding assistants, generative AI models or any other AI systems. The entire process is fully reproducible without any use of AI. Therefore this analysis does not have any of the possible errors, liabilities, or risks of AI systems.
<br><br>
Therefore any errors are entirely my responsibility alone.<br>

<h4>All the processes and results are fully transparent and reproducible</h4>

<h4>3. Literature Review</h4>
We will look at two anayses of the same data set.

https://www.kaggle.com/code/faisalamir/analysis-modelling-and-business-insight
The mean RMSE is approximately $2,338.90.

https://www.kaggle.com/code/hakimshaikhh/bmwcar-price-prediction
The mean RMSE is approximately $2,300.08

The results from this present analysis for non-ensemble models will approximately match the values in those reports. However, the RMSE using ensembles will be much lower (and lower RMSE is better). The present analysis also uses more models than both of those anayses combined, and that is part of why the present analysis has results with a lower RMSE.

<h4>4. The Data Set</h4>
The data set was originally posted at https://www.kaggle.com/datasets/nalisha/bmw-car-sales-and-price-dataset. The description of the data set states:<br><br>

> This dataset contains detailed information about BMW cars, including technical specifications, usage metrics, and pricing. It is suitable for data analysis, visualization, and machine learning regression tasks.

Columns description (as reported on kaggle.com):

|Column Name  |Description                                        |
|:------------|:--------------------------------------------------|
|model        |BMW car model name (e.g., 3 Series, X5, 5 Series)  |
|year         |Manufacturing year of the vehicle                  |
|price        |Selling price of the car                           |
|transmission |Transmission type (Manual / Automatic / Semi-Auto) |
|mileage      |Total distance driven (in miles or km)             |
|fuelType     |Fuel type (Petrol, Diesel, Hybrid, Electric)       |
|tax          |Annual road tax                                    |
|mpg          |Miles per gallon (fuel efficiency)                 |
|engineSize   |Engine size in liters                              |

The data has 10,781 rows and nine variables. All the variables are numeric.<br>

<h4>Why is this type of problem so difficult to solve?</h4>

George Akerloff, Joseph Stiglitz and Michael Spence won the 2001 Nobel Memorial Prize in Economic Sciences based on their paper, "The Market for 'Lemons': Quality Uncertainty and the Market Mechanism", published in the Quarterly Journal of Economics. The problem of finding a fair price for a used car as modeled as a game of assymetric information that the seller will win in a majority of cases. It is the assymetry of information about the used car that makes this such a difficult problem to solve.

Several laws have been passed since the publication of the paper. The Magnuson–Moss Warranty Act (P.L. 93-637) is a federal law that protects consumers by requiring if a product has a warranty, the warranty must comply with the act.

<h4>4a. Boxplots</h4><br>

![Boxplots](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_boxplots.jpg)<br>

What are boxplots, and what are they telling us about the credit card fraud data?

Boxplots provide a five number summary of each variable. The five values are:
<br>
• Minimum (or 0%) value. This is the lowest value for the specific feature.<br>
• Maximum (or 100%) value. This is the maximum value for the specific feature.<br>
• Median value. 50% of the data is above the median, and 50% is below the median.<br>
• First Quartile (Q1 or 25th percentile). The median of the lower 50% of the data set.<br>
• Third Quartile (Q3 or75th percentile). The median of the upper 50% of the data set.<br>
<br>
In addition, the Interquartile Range provides:IQR = Q3 - Q1<br>

<h4>4b: Looking at the data: Histograms of the numeric data</h4>

![Histograms](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_histograms.jpeg)<br>

<h4>4c. Price vs each feature</h4>

![Price vs each feature](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_predictor_vs_target.jpg)<br>

<h4>4d: Head of the data (the first ten rows)</h4>
Head of the data.<br>

|model    | year| price|transmission | mileage|fuelType | tax|  mpg| engineSize|
|:--------|----:|-----:|:------------|-------:|:--------|---:|----:|----------:|
|5 Series | 2014| 11200|Automatic    |   67068|Diesel   | 125| 57.6|        2.0|
|6 Series | 2018| 27000|Automatic    |   14827|Petrol   | 145| 42.8|        2.0|
|5 Series | 2016| 16000|Automatic    |   62794|Diesel   | 160| 51.4|        3.0|
|1 Series | 2017| 12750|Automatic    |   26676|Diesel   | 145| 72.4|        1.5|
|7 Series | 2014| 14500|Automatic    |   39554|Diesel   | 160| 50.4|        3.0|
|5 Series | 2016| 14900|Automatic    |   35309|Diesel   | 125| 60.1|        2.0|
|5 Series | 2017| 16000|Automatic    |   38538|Diesel   | 125| 60.1|        2.0|
|2 Series | 2018| 16250|Manual       |   10401|Petrol   | 145| 52.3|        1.5|
|4 Series | 2017| 14250|Manual       |   42668|Diesel   |  30| 62.8|        2.0|
|5 Series | 2016| 14250|Automatic    |   36099|Diesel   |  20| 68.9|        2.0|


<h4>4e: Data summary</h4>

|   |      model    |     year    |    price      |   transmission |   mileage     |    fuelType  |     tax      |     mpg      |  engineSize  |
|:--|:--------------|:------------|:--------------|:---------------|:--------------|:-------------|:-------------|:-------------|:-------------|
| <b>Min</b>|3 Series:2443  |1996 |1200 |Automatic: 3588  |1 |Diesel: 7027 |0.0 |5.5 |0.000 |
| <b>1st Qu</b>|1 Series:1969  |2016 |14950 |Manual   :2527  |5529 |Electric:   3 |135.0 |45.6 |2.000 |
| <b>Median</b>|2 Series:1229  |2017 |20462 |Semi-Auto:4666  |18347 |Hybrid  : 298 |145.0 |53.3 |2.000 |
| <b>Mean</b>|5 Series:1056  |2017 |22733 |NA              |25497 |Other   :  36 |131.7 |56.4 |2.168 |
| <b>3rd Qu</b>|4 Series: 995  |2019 |27940 |NA              |38206 |Petrol  :3417 |145.0 |62.8 |2.000 |
| <b>Max<b>|X1      : 804  |2020 |123456 |NA              |214000 |NA            |580.0 |470.8 |6.600 |
| <b>Other</b> |(Other)  :2285 |NA           |NA             |NA              |NA             |NA            |NA            |NA            |NA            |

<h4>4f: Data correlation table</h4>

|             |      model|       year| transmission|    mileage|   fuelType|        tax|        mpg| engineSize|          y|
|:------------|----------:|----------:|------------:|----------:|----------:|----------:|----------:|----------:|----------:|
|<b>model</b>        |  1.0000000|  0.0808860|    0.0045742| -0.0891965| -0.0887211|  0.2515533| -0.1328801|  0.2908422|  0.4632516|
|<b>year</b>         |  0.0808860|  1.0000000|    0.2643862| -0.7743820|  0.1396433|  0.0251998| -0.0717623| -0.0263365|  0.6237561|
|<b>transmission</b> |  0.0045742|  0.2643862|    1.0000000| -0.2918219|  0.0815140|  0.0756571| -0.1449845|  0.0357695|  0.2123343|
|<b>mileage</b>      | -0.0891965| -0.7743820|   -0.2918219|  1.0000000| -0.2425344| -0.1707466|  0.1186087| -0.0025398| -0.6053968|
|<b>fuelType</b>     | -0.0887211|  0.1396433|    0.0815140| -0.2425344|  1.0000000|  0.2917326| -0.1723221| -0.0491125|  0.1068889|
|<b>tax</b>          |  0.2515533|  0.0251998|    0.0756571| -0.1707466|  0.2917326|  1.0000000| -0.3517975|  0.4296163|  0.2634838|
|<b>mpg</b>          | -0.1328801| -0.0717623|   -0.1449845|  0.1186087| -0.1723221| -0.3517975|  1.0000000| -0.3998435| -0.2048823|
|<b>engineSize</b>   |  0.2908422| -0.0263365|    0.0357695| -0.0025398| -0.0491125|  0.4296163| -0.3998435|  1.0000000|  0.4601962|
|<b>y (price)</b>            |  0.4632516|  0.6237561|    0.2123343| -0.6053968|  0.1068889|  0.2634838| -0.2048823|  0.4601962|  1.0000000|

Outliers list<br>


<h4>Summary of the Exploratory Data Analysis</h4>



<h4>What the Exploratory Data Analysis suggests</h4>



<h2>5. Model Building (18 individual models, 14 ensembles of models)</h2>

NumericEnsembles will do all of the following:<br>
• Automatically split the data into train/test/validation sets<br>
• Automatically fit each of 18 individual models and 14 ensembles of numeric models to the training data<br>
• Automatically resample as many times as requested (25 times for this example)<br>
• Automatically make predictions and measure accuracy on the holdout data (test and validation)<br>
• Automatically make summary graphics for each of the measures<br>
• Automatically make a summary table for the importance of each variable in the data set<br>
• There are no API calls, no use of any coding assistants, no use of any AI systems.<br>
• Neither the user's data nor activity is shared, stored, or tracked in any way. They do not help produce more accurate models.<br>

<h2>How NumericEnsembles builds a team of rival models requiring only one line of code from the user</h2>
First NumericEnsembles will automatically build a set of numeric models. Some of the models are individual, others are ensembles, some are regular learning, others are deep learning, some are tuned models, others are not tuned models.<br>

<h2>6. Model summaries: Tables and Plots</h2>

The 32 numeric model summaries (18 individual models and 14 ensembles of models):<br>
| Model name  | Individual or Ensemble | Type of Learning  | Type of Tuning |
|:-----:|:----------:|:----------:|:---------:|
| 1. Bagging | Individual | Regular | Not tuned |
| 2. BayesGLM | Individual | Regular | Not tuned |
| 3. BayesRNN | Individual | Regular | Not tuned |
| 4. Cubist | Individual | Regular | Not tuned |
| 5. Earth | Individual | Regular | Not tuned |
| 6. Elastic | Individual | Regular | Cross-Validation |
| 7. Generalized Additive Models | Individual | Regular | Smoothing Splines |
| 8. Gradient Boosted | Individual | Deep Learning | Trees, Shrinkage, Interaction Depth |
| 9. Lasso | Individual | Regular | Cross-Validation |
| 10. Linear | Indvidual | Regular | Tuned (e1071) |
| 11. Neuralnet | Individual | Deep Learning | Size and linout |
| 12. Partial Least Squares | Individual | Regular | Not tuned |
| 13. Principal Components Analysis | Individual | Regular | Not tuned |
| 14. Ridge | Individual | Regular | Cross-Validation |
| 15. RPart | Individual | Regular | Not tuned |
| 16. Support Vector Machines | Individual | Regular | Tuned (e1071)|
| 17. Trees| Individual | Regular | Cross-Validation |
| 18. XGBoost | Individual | Deep Learning | xgb.params, nrounds = 70 |
| 19. Ensemble Bagging | Ensemble | Regular | Not tuned |
| 20. Ensemble BayesGLM | Ensemble | Regular | Not tuned |
| 21. Ensemble BayesRNN | Ensemble | Regular | Not tuned |
| 22. Ensemble Cubinst | Ensemble | Regular | Not tuned |
| 23. Ensemble Earth | Ensemble | Regular | Not tuned |
| 24. Ensemble Elastic | Ensemble | Regular | Cross-Validation |
| 25. Ensemble Gradient Boosted | Ensemble | Deep Learning | Trees, Shrinkage, Interaction Depth |
| 26. Ensemble Lasso | Ensemble | Regular | Cross-Validation |
| 27. Ensemble Linear | Ensemble | Regular | Tuned (e1071) |
| 28. Neuralnet | Ensemble | Deep Learning | Size and linout |
| 29. Ensemble Ridge | Ensemble | Regular Learning | Cross-Validation |
| 30. Ensemble RPart | Ensemble | Regular Learning | Not tuned |
| 31. Ensemble Support Vector Machines | Ensemble | Regular Learning | Tuned (e1071) |
| 32. Ensemble Trees | Ensemble | Regular Learning | Cross-Validation |


<h4>How the NumericEnsembles package makes this much faster and easier to solve, using only one line of code, while maintaining a very high level of accuracy on the holdout data. Here is the one line of code (plus a couple of lines to time the analysis and check for errors):</h4>

```
start_time <- Sys.time()

library(NumericEnsembles)
Numeric(data = read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/EnsemblesData/refs/heads/main/bmw.csv', stringsAsFactors = TRUE),
  colnum = 3,
  numresamples = 25,
  remove_VIF_above = 5.00,
  remove_data_correlations_greater_than = 0.99,
  remove_ensemble_correlations_greater_than = 1.00,
  scale_all_predictors_in_data = "N",
  data_reduction_method = 0,
  ensemble_reduction_method = 0,
  how_to_handle_strings = 1,
  predict_on_new_data = "N",
  save_all_trained_models = "N",
  stratified_random_column = 0,
  set_seed = "N",
  save_all_plots = "N",
  use_parallel = "Y",
  train_amount = 0.60,
  test_amount = 0.20,
  validation_amount = 0.20)
  
  end_time <- Sys.time()
  duration <- end_time - start_time
  duration
  warnings()
```

Comments on NumericEnsembles applied to the BMW Used Car Price data set:

<h4>Everything ran in 3.016171 minutes without any errors, warnings, or issues. The majority of the time was spent saving the image files and trained models. If the models and images are not saved, everything completed in 1.776362 mins, a substantially shorter run time.</h4>

<h3>The models:</h3>

### Individual models first, then ensembles

|Model name | Model |
|:----------|:-----------|
|Bagging | ```bagging_train_fit <- ipred::bagging(formula = y ~ ., data = train)```|
|BayesGLM | ```arm::bayesglm(y ~ ., data = train, family = gaussian(link = "identity"))```|
|BayesRNN | ```bayesrnn_train_fit <- brnn::brnn(x = as.matrix(train), y = train$y)```|
|Cubist | ```cubist_train_fit <- Cubist::cubist(x = train[, 1:ncol(train) - 1], y = train$y)```|
|Earth |```earth_train_fit <- earth::earth(x = train[, 1:ncol(train) - 1], y = train$y)```|
|Elastic |```y <- train$y```<br>```x <- data.matrix(train %>% dplyr::select(-y))```</br>```elastic_model <- glmnet::glmnet(x, y, alpha = 0.5)```<br>```elastic_cv <- glmnet::cv.glmnet(x, y, alpha = 0.5)```<br>```best_elastic_lambda <- elastic_cv$lambda.min```<br>```best_elastic_model <- glmnet::glmnet(x, y, alpha = 0.5, lambda = best_elastic_lambda)```|
|GAM (Generalized Additive Models) with Smoothing Splines | ```n_unique_vals <- purrr::map_dbl(df, dplyr::n_distinct)```<br># Names of columns with >= 4 unique vals<br>```keep <- names(n_unique_vals)[n_unique_vals >= 4]```<br>```gam_data <- df %>%dplyr::select(dplyr::all_of(keep))``` ```# Model data``` ```train1 <- train %>%dplyr::select(dplyr::all_of(keep))``` ```test1 <- test %>%dplyr::select(dplyr::all_of(keep))<br>```validation1 <- validation %>%dplyr::select(dplyr::all_of(keep))```names_df <- names(gam_data[, 1:ncol(gam_data) - 1])```<br>```f2 <- stats::as.formula(paste0("y ~", paste0("gam::s(", names_df, ")", collapse = "+")))```<br>``` gam_train_fit <- gam(f2, data = train1)```|
|Gradient Boosted |```gb_train_fit <- gbm::gbm(train$y ~ ., data = train, distribution = "gaussian", n.trees = 100, shrinkage = 0.1, interaction.depth = 10)```|
|Lasso |```y <- train$y```<br>```x <- data.matrix(train %>% dplyr::select(-y))```</br>```lasso_model <- glmnet::glmnet(x, y, alpha = 1)```<br>```lasso_cv <- glmnet::cv.glmnet(x, y, alpha = 1)```<br>```best_lasso_lambda <- lasso_cv$lambda.min```<br>```best_lasso_model <- glmnet::glmnet(x, y, alpha = 1, lambda = best_lasso_lambda)```|
|Linear |```linear_train_fit <- e1071::tune.rpart(formula = y ~ ., data = train)```|
|Neuralnet |```neuralnet_train_fit <- nnet::nnet(train$y ~ ., data = train, size = 0, linout = TRUE, skip = TRUE)```|
|Partial Least Squares|```pls_train_fit <- pls::plsr(train$y ~ ., data = train)```|
|Principal Components Analysis|```pcr_train_fit <- pls::pcr(train$y ~ ., data = train)```|
|Ridge |```y <- train$y```<br>```x <- data.matrix(train %>% dplyr::select(-y))```</br>```ridge_model <- glmnet::glmnet(x, y, alpha = 0)```<br>```ridge_cv <- glmnet::cv.glmnet(x, y, alpha = 0)```<br>```best_ridge_lambda <- ridge_cv$lambda.min```<br>```best_ridge_model <- glmnet::glmnet(x, y, alpha = 0, lambda = best_ridge_lambda)```|
|RPart|```rpart_train_fit <- rpart::rpart(train$y ~ ., data = train)```|
|Support Vector Machines|```svm_train_fit <- e1071::tune.svm(x = train, y = train$y, data = train)```|
|Trees|``` tree_train_fit <- tree::tree(train$y ~ ., data = train)```|
|XGBoost|```train_x <- data.matrix(train[, -ncol(train)])```<br>```train_y <- train[, ncol(train)]```<br># define predictor and response variables in test set<br>```test_x <- data.matrix(test[, -ncol(test)])```<br>```test_y <- test[, ncol(test)]```# <br>define predictor and response variables in validation set<br>```validation_x <- data.matrix(validation[, -ncol(validation)])```<br>```validation_y <- validation[, ncol(validation)]```<br># define final train, test and validation sets<br>```xgb_train <- xgboost::xgb.DMatrix(data = train_x, label = train_y)```<br>```xgb_test <- xgboost::xgb.DMatrix(data = test_x, label = test_y)```<br>```xgb_validation <- xgboost::xgb.DMatrix(data = validation_x, label = validation_y)<br>```# define watchlist<br>```watchlist <- list(train = xgb_train, validation = xgb_validation)```<br>```watchlist_test <- list(train = xgb_train, test = xgb_test)```<br>```watchlist_validation <- list(train = xgb_train, validation = xgb_validation)```<br># fit XGBoost model and display training and validation data at each round<br>```nxgb_model <- xgboost::xgb.train(data = xgb_train, params = xgboost::xgb.params(max_depth = 3), nrounds = 70)```<br>```xgb_model_validation <- xgboost::xgb.train(data = xgb_train, params = xgboost::xgb.params(max_depth = 3), nrounds = 70)```|
|Ensemble (how the ensemble is made)|```ensemble <- data.frame("Bagging" = y_hat_bagging * 1 / bagging_holdout_RMSE_mean```,<br>```"BayesGLM" = y_hat_bayesglm * 1 / bayesglm_holdout_RMSE_mean,```<br>```"BayesRNN" = y_hat_bayesrnn * 1 / bayesrnn_holdout_RMSE_mean,```<br>```"Cubist" = y_hat_cubist * 1 / cubist_holdout_RMSE_mean,```<br>```"Earth" = y_hat_earth * 1 / earth_holdout_RMSE_mean,```<br>```"Elastic" = y_hat_elastic *1 / elastic_holdout_RMSE,```<br>```"GAM" = y_hat_gam * 1 / gam_holdout_RMSE_mean,```<br>```"GBM" = y_hat_gb * 1 / gb_holdout_RMSE_mean,```<br>```"Lasso" = y_hat_lasso *1 / lasso_holdout_RMSE_mean,```<br>```"Linear" = y_hat_linear * 1 / linear_holdout_RMSE_mean,```<br>```"Neuralnet" = y_hat_neuralnet *1 / neuralnet_holdout_RMSE_mean,```<br>```"PCR" = y_hat_pcr * 1 / pcr_holdout_RMSE_mean,```<br>```"PLS" = y_hat_pls * 1 / pls_holdout_RMSE_mean,```<br>```"Ridge" = y_hat_ridge *1 / ridge_holdout_RMSE_mean,```<br>```"Rpart" = y_hat_rpart * 1 / rpart_holdout_RMSE_mean,```<br>```"SVM" = y_hat_svm * 1 / svm_holdout_RMSE_mean,```<br>```"Tree" = y_hat_tree * 1 / tree_holdout_RMSE_mean,```<br>```"XGBoost" = y_hat_xgb * 1 / xgb_holdout_RMSE_mean```)|
|Ensemble Bagging|``` ensemble_bagging_train_fit <- ipred::bagging(formula = y_ensemble ~ ., data = ensemble_train)```|
|Ensemble BayesGLM|```ensemble_bayesglm_train_fit <- arm::bayesglm(y_ensemble ~ ., data = ensemble_train, family = gaussian(link = "identity"))```|
|Ensemble BayesRNN|```ensemble_bayesrnn_train_fit <- brnn::brnn(x = as.matrix(ensemble_train), y = ensemble_train$y_ensemble)```|
|Ensemble Cubist| ```ensemble_cubist_train_fit <- Cubist::cubist(x = ensemble_train[, 1:ncol(ensemble_train) - 1], y = ensemble_train$y_ensemble)```|
|Ensemble Earth|```ensemble_earth_train_fit <- earth::earth(x = ensemble_train[, 1:ncol(ensemble_train) - 1], y = ensemble_train$y_ensemble)```|
|Ensemble Elastic|```ensemble_y <- ensemble_train$y_ensemble```<br>```ensemble_x <- data.matrix(ensemble_train %>% dplyr::select(-y_ensemble))```<br>```ensemble_elastic_model <- glmnet(ensemble_x, ensemble_y, alpha = 0.5)```<br>```ensemble_elastic_cv <- glmnet::cv.glmnet(ensemble_x, ensemble_y, alpha = 0.5)```<br>```ensemble_best_elastic_lambda <- ensemble_elastic_cv$lambda.min```<br>```ensemble_best_elastic_model <- glmnet(ensemble_x, ensemble_y, alpha = 0, lambda = ensemble_best_elastic_lambda)```|
|Ensemble Gradient Boosted|```ensemble_gb_train_fit <- gbm::gbm(ensemble_train$y_ensemble ~ .,```<br>```data = ensemble_train, distribution = "gaussian", n.trees = 100, shrinkage = 0.1, interaction.depth = 10```|
|Ensemble Lasso|```ensemble_y <- ensemble_train$y_ensemble```<br>```ensemble_x <- data.matrix(ensemble_train %>% dplyr::select(-y_ensemble))```<br>```ensemble_lasso_model <- glmnet(ensemble_x, ensemble_y, alpha = 1)```<br>```ensemble_lasso_cv <- glmnet::cv.glmnet(ensemble_x, ensemble_y, alpha = 1)```<br>```ensemble_best_lasso_lambda <- ensemble_lasso_cv$lambda.min```<br>```ensemble_best_lasso_model <- glmnet(ensemble_x, ensemble_y, alpha = 0, lambda = ensemble_best_lasso_lambda)```|
|Ensemble Linear|```ensemble_linear_train_fit <- e1071::tune.rpart(formula = y_ensemble ~ ., data = ensemble_train)```|
|Ensemble Neuralnet|```ensemble_neuralnet_train_fit <- nnet::nnet(ensemble_train$y_ensemble ~ ., data = ensemble_train, size = 0, linout = TRUE, skip = TRUE)```|
|Ensemble Ridge|```ensemble_y <- ensemble_train$y_ensemble```<br>```ensemble_x <- data.matrix(ensemble_train %>% dplyr::select(-y_ensemble))```<br>```ensemble_ridge_model <- glmnet(ensemble_x, ensemble_y, alpha = 1)```<br>```ensemble_ridge_cv <- glmnet::cv.glmnet(ensemble_x, ensemble_y, alpha = 0)```<br>```ensemble_best_ridge_lambda <- ensemble_ridge_cv$lambda.min```<br>```ensemble_best_ridge_model <- glmnet(ensemble_x, ensemble_y, alpha = 0, lambda = ensemble_best_ridge_lambda)```|
|Ensemble RPart|```ensemble_rpart_train_fit <- rpart::rpart(ensemble_train$y_ensemble ~ ., data = ensemble_train)```|
|Ensemble Support Vector Machines|```ensemble_svm_train_fit <- e1071::tune.svm(x = ensemble_train, y = ensemble_train$y_ensemble, data = ensemble_train)```|
|Trees|```ensemble_tree_train_fit <- tree::tree(ensemble_train$y_ensemble ~ ., data = ensemble_train)```|





<h2>7. Highest Accuracy (Summary report sorted by root mean squared error)</h2>

|Model                     | Mean holdout RMSE| RMSE Lower 95 Conf Int| RMSE Upper 95 Conf Int| RMSE Std Dev|
|:-------------------------|-----------------:|----------------------:|----------------------:|------------:|
|Actual data               |            0.0000|                 0.0000|                 0.0000|       0.0000|
|Ensemble Earth            |           63.0734|                61.1806|                64.9662|       0.7095|
|Ensemble BayesGLM         |           65.8068|                63.8320|                67.7816|       0.6322|
|Ensemble Neuralnet        |           65.8165|                63.8413|                67.7916|       0.6334|
|BayesRNN                  |           76.2650|                73.9763|                78.5536|       0.6820|
|Ensemble BayesRNN         |          111.0657|               107.7326|               114.3987|       0.8284|
|Ensemble Cubist           |          156.4354|               151.7408|               161.1299|      12.0436|
|Ensemble Lasso            |          338.5986|               328.4374|               348.7598|      36.8136|
|Ensemble Elastic          |          499.0891|               484.1117|               514.0665|     270.0379|
|Ensemble Gradient Boosted |          577.7069|               560.3701|               595.0436|      19.8136|
|Ensemble SVM              |         1254.1619|              1216.5251|              1291.7987|      27.9148|
|Ensemble Ridge            |         1260.7839|              1222.9484|              1298.6194|    1179.9504|
|SVM                       |         1319.7860|              1280.1798|              1359.3921|     579.1075|
|Ensemble Bagging          |         1645.9035|              1567.4459|              1724.3610|       9.5654|
|Ensemble Trees            |         2138.4129|              2074.2401|              2202.5856|       5.0887|
|Ensemble Linear           |         2139.1727|              2074.9772|              2203.3683|       5.3388|
|Ensemble Rpart            |         2139.1727|              2074.9772|              2203.3683|       5.3388|
|XGBoost                   |         2801.3520|              2717.2848|              2885.4192|       0.8397|
|Gradient Boosted          |         2815.4712|              2730.9803|              2899.9622|       1.1076|
|Cubist                    |         2869.1998|              2783.0965|              2955.3032|       2.8047|
|Earth                     |         3918.8412|              3801.2387|              4036.4438|       8.9406|
|Bagging                   |         4955.4255|              4806.7155|              5104.1354|       6.4483|
|Linear                    |         5252.0066|              5094.3964|              5409.6169|       6.0310|
|Rpart                     |         5252.0066|              5094.3964|              5409.6169|       6.0310|
|Tree                      |         5252.3087|              5094.6894|              5409.9280|       5.8572|
|Neuralnet                 |         5892.7542|              5715.9155|              6069.5930|      13.5280|
|Elastic                   |         5899.6480|              5722.6024|              6076.6936|      15.3841|
|BayesGLM                  |         5899.6800|              5722.6334|              6076.7266|      13.5280|
|Lasso                     |         5900.0956|              5723.0365|              6077.1547|      15.4604|
|GAM                       |         5903.7299|              5726.5618|              6080.8980|      13.6749|
|Ridge                     |         5918.1990|              5740.5967|              6095.8014|      15.6303|
|PLS                       |         7470.3503|              7246.1687|              7694.5320|      14.9193|
|PCR                       |         7867.0987|              7631.0108|              8103.1866|      15.3738|

<br>

<h2>Summary charts and reports</h2>

Summary results table

<h4>Accuracy barchart, including one standard deviation error bars</h4>

![Accuracy_barchart](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_accuracy_barchart.jpg)<br>

Accuracy plot by each of the 32 models and each resample:

![Accuracy_plot](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_accuracy_plot_free_scales.jpg)<br>

Bias of each of the trained models

![Bias_barchart](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_bias_barchart.jpg)<br>

Kolomogrov-Smirnov test (measures how likely the predicted distribution is to be similar to the training set distribution).

![K_S_test](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_k_s_test_barchart.jpg)<br>

Overfitting barcharts (three charts)

![K_S_test](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_overfitting_barchart.jpg)<br>

![Overfitting_free_scales](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_overfitting_plot_free_scales.jpg)<br>

![Total_plot_free_scales](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_total_plot_free_scales.jpg)<br>

Variance Inflation Factor

Ensemble Correlation



Neither the original data nor the ensemble had strongly correlated predictors.

<br>The correlation table for the ensemble:

We can also see how much time each model took to run (mean time, measured in seconds):

![Duration](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Big_credit_card_fraud_duration_barchart.jpg)<br>

<br>

<h2>9. Strongest Evidence Based Recommendations</h2>

First principles to solve the problem:<br>
• The Exploratory Data Analysis suggested the target is logistic<br>
• The LogisticEnsembles package was used to do the entire analysis<br>
• The Exploratory Data Analysis showed the data have very narrow interquartile ranges<br>
• The Correlation tables show that neither the original data nor the ensemble are strongly correlated with the target<br>
• Three models had 100% predictive accuracy on the holdout data: Ensemble C50, Ensemble Elastic and Ensemble XGBoost<br>
• The same three models had excellent scores on Sensitivity, Specificity, Type I Error, Type II Error, Precision, Negative Predictive Value and F1 Score<br>
• An analysis of the strongest predictors showed that eight of the ten strongest predictors are negative.<br>
• Given these results, it is recommended that LogisticEnsembles be used with similar data sets about credit card fraud.<br>

<h2>10. Summary: The Story in this Data Set</h2><br>


<h2>11: Conclusions</h2>

The LogisticEnsembles package was able to complete the entire analysis in less than five minutes, providing results on the holdout data which meet the customer's requirements for predicting fraud in this data set with very high accuracy.

<h2>12. References</h2>

#Rstats #DataScience #XGBoost #Fraud #Finance #FinancialFraud #CreditCard #Crime #FinancialCrime #FightingCrime #CrimeFighter #Dataviz #ggplot2 #tidyverse
