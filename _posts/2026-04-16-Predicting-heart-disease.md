<h1>Predicting Heart Disease Using the Cleveland Heart Data Set and Classification Ensembles</h1>

Russ Conte<br>
April 16, 2026

<h3>Abstract</h3>

<h4>Reproducibility of the results</h4>
You can get the same results with this data set. The methods, results and graphics are all fully reproducible, as described in this blog post

<h4>Outline</h4>
In this blog post we will look at:<br>
• How big is the problem?<br>
• Why is this type of problem so difficult to solve?<br>
• Look at the data: box plots, histograms, head of the data, data summaries<br>
• What does the customer actually want?<br>
• Highest accuracy classification model using the ClassificationEnsembles package<br>
• Strongest predictor using reports from the ClassificationEnsembles package<br>
• Strongest evidence based recommendations<br>
• Conclusion/summary<br>

<h4>How big is the problem?</h4>
The issue of heart disease in the United States is huge. According to the Centers for Disease Control and Prevention:

> Heart disease is the leading cause of death for men, women, and people of most racial and ethnic groups.[^1]<br>
One person dies every 34 seconds from cardiovascular disease.[^1]<br>
In 2023, 919,032 people died from cardiovascular disease. That's the equivalent of 1 in every 3 deaths.[^1]<br>
Heart disease is costly. The cost of health care services and medications from heart disease amounted to more than $168 billion between 2021 and 20222.<br>



<h4>The data set</h4>

302 observations, 12 columns

Source: https://www.kaggle.com/datasets/aavigan/cleveland-clinic-heart-disease-dataset<br>

Head of the data set:

| Row | Age|Sex  |Chest_pain_type | Resting_blood_pressure| Cholesteral|Fasting_blood_sugar | Max_heart_rate|Exercise_induced_angina | Old_peak|Slope |Sick_or_buff |y  |
|:---|---:|:----|:---------------|----------------------:|-----------:|:-------------------|--------------:|:-----------------------|--------:|:-----|:------------|:--|
|287 |  44|fem  |notang          |                    118|         242|fal                 |            149|fal                     |      0.3|flat  |buff         |H  |
|241 |  39|fem  |notang          |                    138|         220|fal                 |            152|fal                     |      0.0|flat  |buff         |H  |
|47  |  44|male |asympt          |                    112|         290|fal                 |            153|fal                     |      0.0|up    |sick         |S2 |
|128 |  45|male |abnang          |                    128|         308|fal                 |            170|fal                     |      0.0|up    |buff         |H  |
|261 |  57|fem  |abnang          |                    130|         236|fal                 |            174|fal                     |      0.0|flat  |sick         |S1 |
|191 |  64|fem  |asympt          |                    130|         303|fal                 |            122|fal                     |      2.0|flat  |buff         |H  |


<h4>Why is this type of problem so difficult to solve?</h4>


<h2>Step 1: Look at the data: box plots, histograms, head of the data, data summaries</h2>


<h4>Step 1b: Looking at the data: Histograms of the data</h4>

<h4>Step 1c: Head of the data</h4>


<h4>Step 1d: Data summary</h4>


<h4>Correlation of the target (Class) to each variable (closer to 1.00 is better)</h4>

<h4>Summary of the Exploratory Data Analysis</h4>

<h4>What the Exploratory Data Analysis suggests</h4>

<h2>What does the customer actually want?</h2>


<h2>Step 2: Building the models using the classificationEnsembles package, as hosted on CRAN</h2>

ClassificationEnsembles will do all of the following:<br>
• Automatically split the data into train/test/validation sets<br>
• Automatically fit each of 7 classification models and 5 ensembles of classification models to the training data<br>
• Automatically resample as many times as requested (25 times for this example)<br>
• Automatically make predictions and check accuracy on the holdout data (test and validation)<br>
• Automatically make summary graphics for each of the measures<br>
• Automatically make a summary table for the importance of each variable in the data set<br>
• There are no API calls, no use of any coding assistants, no use of any AI systems.<br>
• Neither the user's data nor activity is shared, stored, or tracked in any way. They do not help produce more accurate models.<br>

<h2>How ClassificationEnsembles builds a team of rival models requiring only one line of code from the user</h2>

Classification model summaries:<br>
| Model name  | Individual or Ensemble | Type of Learning  | Type of Tuning |


<h4>How the ClassificationEnsembles package makes this much faster and easier to solve, using only one line of code, while maintaining a very high level of accuracy on the holdout data. Here is the one line of code (plus a couple of lines to time the analysis and check for errors):</h4>

```
library(ClassificationEnsembles)
df <- Cleveland_heart
df$Class <- as.factor(df$Class)
start_time <- Sys.time()
Classification(data = df,
               colnum = 12,
               numresamples = 25,
               predict_on_new_data = "N",
               save_all_plots = "Y",
               set_seed = "N",
               how_to_handle_strings = 1,
               remove_VIF_above <- 5.00,
               save_all_trained_models = "Y",
               stratified_random_column = 0,
               scale_all_numeric_predictors_in_data = "N",
               use_parallel = "Y",
               train_amount = 0.50,
               test_amount = 0.25,
               validation_amount = 0.25)

end_time <- Sys.time()
duration <- end_time - start_time
duration
warnings()

```

Comments on ClassificationEnsembles applied to the Cleveland Heart data set:


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




<h2>Step 3: Highest Accuracy results on the holdout data, resampled 25 times, sorted (decreasing) by accuracy on the holdout data</h2>

<br>

Comments on the summary report:

The ClassificationEnsembles package automatically calculated all of the results, sorted by Accuracy, and put them in a summary table:<br>

<h2>Step 4: Strongest Predictors: Which variables are the strongest predictors, and how strong are they?</h2>


We can also see how much time each model took to run (measured in seconds):

![Duration](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Big_credit_card_fraud_duration_barchart.jpg)<br>

<h2>Step 5: Strongest evidence based recommendations to fight credit card fraud based on the data and analysis</h2>

First principles to solve the problem:<br>
• The Exploratory Data Analysis suggested the target is classification<br>
• The ClassificationEnsembles package was used to do the entire analysis<br>

<h2>Step 6: Conclusions</h2>

The ClassificationEnsembles package was able to complete the entire analysis in less than five minutes, providing results on the holdout data which meet the customer's requirements for predicting fraud in this data set with very high accuracy.

#Rstats #DataScience #RHealth #HealthStats #Health #HeartDisease #Wellness #HeartHealth #Diabetes #Cardology #Doctor #Dataviz #ggplot2 #tidyverse

Footnotes:

[^1]:National Center for Health Statistics. Multiple Cause of Death 2018–2023 on CDC WONDER Database. Accessed February 1, 2025. [https://wonder.cdc.gov/mcd.html](https://www.cdc.gov/heart-disease/data-research/facts-stats/index.html)
