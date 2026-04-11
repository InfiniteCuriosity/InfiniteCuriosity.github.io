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
10. Comparison to Other Results for this Data Set<br>
11. Notes<br>
12. References<br>

<h4>1. Introduction and Statement of the Problem</h4>

<h4>2. Statement regarding No Use of AI</h4>
No Artificial Intelligence systems (AI) were used in any part of the process. This analysis excludes all commercial AI systems, large language models, coding assistants, generative AI models or any other AI systems. The entire process is fully reproducible without any use of AI. Therefore this analysis does not have any of the possible errors, liabilities, or risks of AI systems.
<br><br>
Therefore any errors are entirely my responsibility alone.<br>

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
|Elastic |  y <- train$y<br>x <- data.matrix(train %>% dplyr::select(-y))<br>elastic_model <- glmnet::glmnet(x, y, alpha = 0.5)<br>elastic_cv <- glmnet::cv.glmnet(x, y, alpha = 0.5)<br>best_elastic_lambda <- elastic_cv$lambda.min<br>best_elastic_model <- glmnet::glmnet(x, y, alpha = 0.5, family = "binomial")<br>|
|Flexible Discriminant Analysis | fda_train_fit <- MachineShop::fit(as.factor(y) ~ ., data = train01, model = "FDAModel", family = binomial(link = "logit")) |<br>
|Generalized Additive Models | gam::gam(y ~ ., data = train, family = binomial(link = "logit"))|<br>
|Gradient Boosted |gbm::gbm(train$y ~ ., data = train, distribution = "bernoulli")|<br>
|Generalized Linear Models|stats::glm(y ~ ., data = train, family = "binomial")|<br>
|Neuralnet| nnet::nnet(train$y ~ ., data = train, size = 0, linout = TRUE, skip = TRUE, family = binomial(link = "logit"))|<br>
|Ridge| y <- train$y<br>x <- data.matrix(train %>% dplyr::select(-y))<br>ridge_model <- glmnet::glmnet(x, y, alpha = 0)<br>ridge_cv <- glmnet::cv.glmnet(x, y, alpha = 0)<br>best_ridge_lambda <- ridge_cv$lambda.min<br>best_ridge_model <- glmnet::glmnet(x, y, alpha = 0, family = "binomial")|<br>
|ensemble (how the ensemble is made)|data.frame(<br>"Elastic" = c(elastic_test_predictions, elastic_validation_predictions),<br>"Flexible_Discriminant_Analysis" = c(fda_test_pred, fda_validation_pred),<br>"Generalized_Additive_Models" = c(gam_test_predictions,<br>gam_validation_predictions)<br>"Generalized_Linear_Models" = as.numeric(c(glm_test_predictions, glm_validation_predictions))<br>"Gradient_Boosted" = as.numeric(c(gb_test_predictions, gb_validation_predictions))<br>"Neuralnet" = c(neuralnet_test_predictions, neuralnet_validation_predictions)<br>"Ridge" = c(ridge_test_predictions, ridge_validation_predictions)<br>)<br>|
|Ensemble C50|C50::C5.0(as.factor(ensemble_y_train) ~ ., data = ensemble_train, family = binomial(link = "logit"))<br>
|Ensemble Elastic| y <- ensemble_train$y<br>x <- data.matrix(ensemble_train[, 1:ncol(ensemble_train)])<br>ensemble_elastic_model <- glmnet::glmnet(x, y, alpha = 0.5)<br>ensemble_elastic_cv <- glmnet::cv.glmnet(x, y, alpha = 0.5)<br>best_ensemble_elastic_lambda <- ensemble_elastic_cv$lambda.min<br>best_ensemble_elastic_model <- glmnet::glmnet(x, y, alpha = 0.5, family = "binomial"))|<br>
|Ensemble Generalized Linear Models |y <- ensemble_train$y<br>x <- data.matrix(ensemble_train %>% dplyr::select(-y))<br>ensemble_glmnet_model <- glmnet::glmnet(x, y)<br>ensemble_glmnet_cv <- glmnet::cv.glmnet(x = x,y = y)<br>best_ensemble_glmnet_lambda <- ensemble_glmnet_cv$lambda.min<br>best_ensemble_glmnet_model <- glmnet::glmnet(x, y, alpha = best_ensemble_glmnet_lambda, family = "binomial")|<br>
|Ensemble Neuralnet | nnet::nnet(ensemble_train$y ~ ., data = ensemble_train, size = 0, linout = TRUE, skip = TRUE, family = binomial(link = "logit"))|<br>
|Ensemble XGBoost |ensemble_train_x = data.matrix(ensemble_train[, 1 : ncol(ensemble_train)])<br>ensemble_train_y = ensemble_train[,ncol(ensemble_train) : ncol(ensemble_train)]<br><br>#define predictor and response variables in ensemble_test set<br>ensemble_test_x = data.matrix(ensemble_test[, 1 : ncol(ensemble_test)])<br>ensemble_test_y = ensemble_test[, ncol(ensemble_test) : ncol(ensemble_test)]<br><br>#define predictor and response variables in ensemble_validation set<br>ensemble_validation_x = data.matrix(ensemble_validation[, 1 : ncol(ensemble_validation)])<br>ensemble_validation_y = ensemble_validation[, ncol(ensemble_validation): ncol(ensemble_validation)]<br><br>#define final ensemble_train, ensemble_test and ensemble_validation sets<br> ensemble_xgb_train <-  xgboost::xgb.DMatrix(data = ensemble_train_x, label = as.matrix(ensemble_train_y))<br>ensemble_xgb_test <- xgboost::xgb.DMatrix(data = ensemble_test_x, label = as.matrix(ensemble_test_y))<br>ensemble_xgb_validation <-  xgboost::xgb.DMatrix(data = ensemble_validation_x, label = as.matrix(ensemble_validation_y))<br><br>#define watchlist<br>watchlist = list(ensemble_train = ensemble_xgb_train, ensemble_validation=ensemble_xgb_validation)<br>watchlist_ensemble_test <- list(ensemble_train = ensemble_xgb_train, ensemble_test = ensemble_xgb_test)<br>watchlist_ensemble_validation <- list(ensemble_train = ensemble_xgb_train, ensemble_validation = ensemble_xgb_validation)<br><br>#Definte XGB Model<br>ensemble_xgb_model <- xgboost::xgb.train(data = ensemble_xgb_train, evals = watchlist_ensemble_test, nrounds = 70)<br>|


<h2>7. Highest Accuracy (Summary report sorted by root mean squared error, smallest error on the top of the report)</h2>

<br>

<h2>Summary charts and reports</h2>

<h4>Accuracy barchart, including one standard deviation error bars</h4>

![Accuracy_barchart](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_accuracy_barchart.jpg)<br>

![Accuracy_plot](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_accuracy_plot_free_scales.jpg)<br>

![Bias_barchart](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_bias_barchart.jpg)<br>

![K_S_test](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_k_s_test_barchart.jpg)<br>

![K_S_test](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_overfitting_barchart.jpg)<br>

![Overfitting_free_scales](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_overfitting_plot_free_scales.jpg)<br>

![Total_plot_free_scales](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_total_plot_free_scales.jpg)<br>

Variance Inflation Factor

Ensemble Correlation



Neither the original data nor the ensemble had strongly correlated predictors.

<br>The correlation table for the ensemble:

We can also see how much time each model took to run (measured in seconds):

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

10. Comparison to Other Results for this Data Set<br>


<h2>11: Conclusions</h2>

The LogisticEnsembles package was able to complete the entire analysis in less than five minutes, providing results on the holdout data which meet the customer's requirements for predicting fraud in this data set with very high accuracy.

<h2>12. References</h2>

#Rstats #DataScience #XGBoost #Fraud #Finance #FinancialFraud #CreditCard #Crime #FinancialCrime #FightingCrime #CrimeFighter #Dataviz #ggplot2 #tidyverse
