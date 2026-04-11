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
Therefore any errors are entirely my responsibility.<br>

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

|Column.name  |Description                                        |
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

<h4>Boxplots</h4><br>

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



<h4>Step 1b: Looking at the data: Histograms of the data</h4>

![Histograms](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Predicting_car_prices_histograms.jpg)<br>

The histograms confirm what the boxplots showed: The majority of the values for each feature are narrowly spread out. Only the time feature has data which is widely spread out.

<h4>Step 1c: Head of the data</h4>
Head of the data. 


<h4>Step 1d: Data summary</h4>



<h4>Summary of the Exploratory Data Analysis</h4>

The Exploratory Data Analysis of the Credit Card data provides evidence that the predictors, V1 to V30, are not strongly correlated to Class, they have a narrow distribution, and all the features except Time contain values outside the Interquartile range. The values of V1 through V30 vary across a wide range: For example, V1 varies from -56.40751 to 2.45493, but V5 varies from -113.74331 to 34.80167. 

<h4>What the Exploratory Data Analysis suggests</h4>


It may be very instructive to add charts, tables and graphs from our models to our result to help us in our decision making.

The NumericEnsembles package hosted on CRAN can accomplish all of these requirements, and will be used for this data set.

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

The 32 numeric model summaries:<br>
| Model name  | Individual or Ensemble | Type of Learning  | Type of Tuning |
|:-----:|:----------:|:----------:|:---------:|
| Elastic| Individual | Regular Learning | Cross-Validation |
| Flexible Discriminant Analysis | Individual | Regular Learning | Not Tuned |
| Generalized Additve Models | Individual | Regular Learning | Not Tuned |
| Generalized Linear Models | Individual | Regular Learning | Cross-Validation |
| Gradient Boosted | Individual | Deep Learning | Optimize trees = 100, depth = 1 |
| Neuralnet | Individual | Deep Learning | Linout = True, Skip = True |
| Ridge  | Individual | Regular earning | Cross-Validation |
| Ensemble C50 | Ensemble | Regular Learning | Not Tuned |
| Ensemble Elastic | Ensemble | Regular Learning | Cross-Validation |
| Ensemble GLMNET | Ensemble | Deep Learning | Cross-Validation |
| Ensemble Neuralnet | Ensemble | Deep Learning | Linout = True, Skip = True |
| Ensemble XGBoost | Ensemble | Deep Learning | Validation |

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
  save_all_trained_models = "Y",
  stratified_random_column = 0,
  set_seed = "N",
  save_all_plots = "Y",
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


<h2>7. Highest Accuracy</h2>

|Model                          | Area Under the Curve| True Positive Rate (aka Sensitivity)| True Negative Rate (aka Specificity)| False Positive Rate aka (Type I Error)| False Negative Rate (aka Type II Error)| Positive Predictive Value (aka Precision)| Negative Predictive Value| F1 Score| Duration| Duration standard deviation|
|:------------------------------|----------------:|----------------------------------:|----------------------------------:|------------------------------------:|-------------------------------------:|---------------------------------------:|-------------------------:|--------:|--------:|-----------:|
|Ensemble C50                   |           1.0000|                             1.0000|                             1.0000|                               0.0000|                                0.0000|                                  1.0000|                    1.0000|   1.0000|   1.0742|      0.0214|
|Ensemble Elastic               |           1.0000|                             1.0000|                             1.0000|                               0.0000|                                0.0000|                                  1.0000|                    1.0000|   1.0000|   1.1288|      0.3737|
|Ensemble XGBoost               |           1.0000|                             1.0000|                             1.0000|                               0.0000|                                0.0000|                                  1.0000|                    1.0000|   1.0000|   0.2054|      0.0291|
|Ensemble Neuralnet             |           0.9346|                             0.8986|                             0.9606|                               0.0394|                                0.1014|                                  0.0385|                    0.9998|   0.0738|   0.1442|      0.0226|
|Generalized Additive Models    |           0.9268|                             0.9002|                             0.9605|                               0.0395|                                0.0998|                                  0.0390|                    0.9998|   0.0747|   0.9536|      0.0887|
|Generalized Linear Models      |           0.9268|                             0.9002|                             0.9605|                               0.0395|                                0.0998|                                  0.0390|                    0.9998|   0.0747|   1.0113|      0.0622|
|Flexible Discriminant Analysis |           0.8877|                             0.6950|                             0.9997|                               0.0003|                                0.3050|                                  0.7960|                    0.9995|   0.7417|   0.6158|      0.0816|
|Neuralnet                      |           0.8601|                             0.9672|                             0.7624|                               0.2376|                                0.0328|                                  0.0071|                    0.9999|   0.0141|   2.2355|      0.1190|
|Gradient Boosted               |           0.8386|                             0.4438|                             0.9997|                               0.0003|                                0.5562|                                  0.6668|                    0.9990|   0.4909|   5.2172|      0.0778|
|Elastic                        |           0.8224|                             0.6099|                             0.6099|                               0.0000|                                0.3901|                                  0.8733|                    0.9993|   0.7174|   3.0656|      0.2167|
|Ridge                          |           0.7979|                             0.5703|                             0.9999|                               0.0001|                                0.4297|                                  0.8838|                    0.9992|   0.6924|   3.4303|      0.1135|
|Ensemble GLM                   |           0.7975|                             0.7101|                             0.9998|                               0.0002|                                0.2899|                                  0.8585|    
<br>
Comments on the summary report:

The LogisticEnsembles package automatically calculated all of the results, sorted by Area Under the Curve, and put them in a summary table:<br>
• Area Under the Curve<br>
• True Positive Rate (Sensitivity)<br>
• True Negative Rate (Specificity)<br>
• False Positive Rate (Type I Error)<br>
• False Negative Rate (Type II Error)<br>
• Positive Predictive Value (Precision)<br>
• Negative Predictive Value<br>
• F1 Score<br>
• Duration (in seconds)<br>
• Standard Deviation of the mean duration<br>

Three ensembles (Ensemble C50, Ensemble Elastic and Ensemble XGBoost) had 100% accuracy as measured by the AUC score on the holdout data. This can be viewed by looking at the ROC (Receiver Operating Curves) for the data, with the Area Under the Curve (AUC) noted for each graph:<br><br>

![ROC Curves](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Big_credit_card_data_ROC_curves.jpg)<br>

<h3>Summary graphs on each of the measures for all of the models and resamples</h3>

True positive rate:

![True positive rate](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/big_credit_card_fraud_true_positive_rate_free_scales.jpg)<br>

True negative rate:

![True negative rate](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/big_credit_card_fraud_true_negative_rate_free_scales.jpg)<br>

False positive rate:

![False positive rate](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/big_credit_card_fraud_false_positive_rate_free_scales.jpg)<br>

False negative rate:

![False negative rate](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/big_credit_card_fraud_false_negative_rate_free_scales.jpg)<br>

Positive predictive value:

![Positive predictive value](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/big_credit_card_fraud_positive_predictive_value_free_scales.jpg)<br>

Negative predictive value:

![Negative predictive value](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/big_credit_card_fraud_negative_predictive_value_free_scales.jpg)<br>

F1 score:

![F1 score](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/big_credit_card_fraud_F1_score_free_scales.jpg)<br>

<h2>8. Strongest Predictor</h2><br>


![Variable importance table](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Big_Credit_Card_Fraud_Variable_Importance_Report.jpg)<br>

![Variable Importance Barchart](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Big_credit_variable_importance_barchart.jpg)<br>

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
