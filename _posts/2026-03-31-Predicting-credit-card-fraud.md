<b><h1>How data science can help fight credit card fraud</h1></b>

<h3>Introduction</h3>
Credit card fraud is a huge problem in the retail sector, with total losses in the billions of dollars each year. This blog post will highlight how you can understand the process of fighting credit card fraud with data science.

<h4>The data set (it's enormous!)</h4>
One of the largest credit card fraud data sets was posted at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.

The data:<br>
• Reports transactions "made by credit cards in September 2013 by European cardholders."<br>
• Contains 492 frauds out of 284,807 transactions (~ 0.172% of the total)<br>
• Has been anonymized, but the label (fraud or genuine) and amounts are accurate.<br>
• The data set contains 284,807 rows, and 31 variables. It is 66.3 MB in size.<br>
• The data is transformed by Principal Components Analysis, due to confidentiality of the original data.<br>

<h4>Why is this so difficult to solve?</h4>
Most examples of data have large results that are easy to see. However, fraud data typically only shows up in a small fraction of 1% of the transactions. In our case it's less than 2/10 of 1% of the transactions. We are looking for the proverbial needle in the haystack.

<h4>How big is the problem?</h4>
According to a paper from the European Central Bank on payment fraud dated 15 December, 2025, "In 2024 payment fraud rate in European Economic Area stable at around 0.002% of total value of transactions in a calendar year." (https://www.ecb.europa.eu/press/pr/date/2025/html/ecb.pr251215~e133d9d683.en.html) This amounted to a total of €4.2 billion in 2024, which is an increase from € 3.5 billion in 2023 and € 3.4 billion in 2022.<br>
<br>
The report from the European Central Bank breaks down the numbers as follows:

>"For 2024, the overall losses for credit transfers were €2.200 billion (a year-on-year increase of 16%), and for card payments with cards issued in the EU/EEA they were €1.329 billion (a year-on-year increase of 29%). For credit transfers, payment service users bore approximately 85% of total fraud losses in 2024, mainly as a result of scams that tricked users into initiating fraudulent transactions."

<h2>Step 1: Looking at the data</h2>
We will begin by looking at boxplots of the data:<br>
![boxplots](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Big_credit_card_boxplots.jpg)<br>
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

The boxplots for the Credit Card Fraud data set clearly show the values for V1 through v28 have a very small interquartile range, and number of values above and below that range for virtually all features.The values for each feature vary across the data set. For example, the Boxplots show that V10 has values between approximately -24 and 24, but V24 has values between approximately -3 and 5.The boxplot for y (the target) only has values of 0 and 1, without any interquartile range.

<h4>Step 1b: Looking at the data: Histograms of the data</h4>
![histograms](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Big_credit_card_histograms.jpg)<br>

The histograms confirm what the boxplots showed: The majority of the values for each feature are narrowly spread out. Only the time feature has data which is widely spread out.

<h4>Step 1c: Head of the data for selected columns: Time, V1–V5, Class</h4>
Head of the data. Note that only V1 through V5 are shown, but the original data has columns V1 through V28, columns for Time, Amount and Class, as noted below:


| Time  | V1         | V2         | V3        | V4         | V5         | Amount | Class |
|:-----:|:----------:|:----------:|:---------:|:----------:|:----------:|:------:|:-----:|
| 0     | -1.3598071 | -0.0727812 | 2.5363467 | 1.3781552  | -0.3383208 | 149.62 | 0     |
| 0     | 1.1918571  | 0.2661507  | 0.1664801 | 0.4481541  | 0.0600176  | 2.69   | 0     |
| 1     | -1.3583541 | -1.3401631 | 1.7732093 | 0.3797796  | -0.5031981 | 378.66 | 0     |
| 1     | -0.9662717 | -0.1852260 | 1.7929933 | -0.8632913 | -0.0103089 | 123.50 | 0     |
| 2     | -1.1582331 | 0.8777368  | 1.5487178 | 0.4030339  | -0.4071934 | 69.99  | 0     |
| 2     | -0.4259659 | 0.9605230  | 1.1411093 | -0.1682521 | 0.4209869  | 3.67   | 0     |


<h4>Step 1d: Data summary for selected columns: Time, V1–V5, Class</h4>

|Measure | Time           | V1               | V2                | V3               | V4                | V5                | Class             |
|:------:|:--------------:|:----------------:|:-----------------:|:----------------:|:-----------------:|:-----------------:|:-----------------:|
|Min     | 0       | -56.40751   | -72.71573    | -48.3256    | -5.68317     | -113.74331   | 0.000000    |
|1st Qtr| 54202 | 0.92037 | -0.59855 | -0.8904 | -0.84864 | -0.69160 | 0.000000 |
|Median| 84692 | 0.01811 | 0.06549  | 0.1798  | -0.01985  | -0.05434 | 0.000000  |
|Mean| 94814   | 0.00000   | 0.00000    | 0.0000    | 0.00000    | 0.00000    | 0.001727    |
|3rd Qtr| 139320 | 1.31564 | 0.80372  | 1.0272  | 0.74334  | 0.61193  | 0.000000 |
|Max| 172792   | 2.45493   | 22.05773   | 9.3826    | 16.87534    | 34.80167   | 1.000000    |


<h4>Correlation of the target (Class) to each variable (closer to 1.00 is better</h4>

| Feature | Correlation to Class |
|:--------|---------------------:|
| Time    |           -0.0123226 |
| V1      |           -0.1013473 |
| V2      |            0.0912887 |
| V3      |           -0.1929608 |
| V4      |            0.1334475 |
| V5      |           -0.0949743 |
| V6      |           -0.0436432 |
| V7      |           -0.1872566 |
| V8      |            0.0198751 |
| V9      |           -0.0977327 |
| V10     |           -0.2168829 |
| V11     |            0.1548756 |
| V12     |           -0.2605929 |
| V13     |           -0.0045698 |
| V14     |           -0.3025437 |
| V15     |           -0.0042234 |
| V16     |           -0.1965389 |
| V17     |           -0.3264811 |
| V18     |           -0.1114853 |
| V19     |            0.0347830 |
| V20     |            0.0200903 |
| V21     |            0.0404134 |
| V22     |            0.0008053 |
| V23     |           -0.0026852 |
| V24     |           -0.0072209 |
| V25     |            0.0033077 |
| V26     |            0.0044554 |
| V27     |            0.0175797 |
| V28     |            0.0095360 |
| Amount  |            0.0056318 |
| Class   |            1.0000000 |

<h4>Summary of the Exploratory Data Analysis</h4>

The Exploratory Data Analysis of the Credit Card data provides evidence that the predictors, V1 to V30, are not strongly correlated to Class, they have a narrow distribution, and all the features except Time contain values outside the Interquartile range. The values of V1 through V30 vary across a wide range: For example, V1 varies from -56.40751 to 2.45493, but V5 varies from -113.74331 to 34.80167. 

<h4>What the Exploratory Data Analysis suggests</h4>

The value of an Exploratory Data Analysis is that it suggests methods to accurately model the data. Given the nature of the data:<br>
• The target, Class, is logistic (either fraud or not fraud)<br>
• Fraud is indicated in 492 out of the 284,807 rows, so AUC is a much more reliable measure than Accuracy.<br>
• A range of models is suggested, as this might provide a better result than a single modeling system, such as Generalized Linear Models.<br>
• We will use seven individual models and five ensembles of models in our analysis.<br>
• We will use a combination of regular learning and deep learning models<br>
• We will use a combination of tuned models (in all possible situations) and untuned models (if it is not possible to tune the models)<br>

It may be very instructive to add charts, tables and graphs from our models to our result to help us in our decision making.

The LogisticEnsembles package hosted on CRAN can accomplish all of these requirements, and will be used for this data set.

<h2>Step 2: Building the models using the LogisticEnsembles package</h2>

Next we will build a set of logistic models. The models used by the LogisticEnsembles package are a team of rivals. Some of the models are individual, others are ensembles, some are regular learning, others are deep learning, some are tuned models, others are not tuned models.<br>

Logistic model summaries:<br>
| Model name  | Individual or Ensemble | Type of Learning  | Type of Tuning |
|:-----:|:----------:|:----------:|:---------:|
| Elastic| Individual | Regular Learning | Cross-Validation |
| Flexible Discriminant Analysis | Individual | Regular Learning | Not Tuned |
| Generalized Additve Models | Individual | Regular Learning | Not Tuned |
| Generalized Linear Models | Individual | Regular Learning | Cross-Validation |
| Gradient Boosted | Individual | Deep Learning | Optimize trees = 100, depth = 1 |
| Neuralnet | Individual | Deep Learning | Linout = True, Skip = True |
| XGBoost  | Individual | Deep Learning | Validation |
| Ensemble C50 | Ensemble | Regular Learning | Not Tuned |
| Ensemble Elastic | Ensemble | Regular Learning | Cross-Validation |
| Ensemble GLMNET | Ensemble | Deep Learning | Cross-Validation |
| Ensemble Neuralnet | Ensemble | Deep Learning | Linout = True, Skip = True |
| Ensemble XGBoost | Ensemble | Deep Learning | Validation |

<h4>Running all the models automatically:</h4>

```
library(LogisticEnsembles)

start_time <- Sys.time()
Logistic(data = read.csv('/Users/russellconte/creditcard.csv'),
         colnum = 31,
         numresamples = 2,
         positive_rate = 0.001727486,
         remove_VIF_greater_than <- 5.00,
         save_all_trained_models = "Y",
         save_all_plots = "Y",
         set_seed = "Y",
         how_to_handle_strings = 0,
         do_you_have_new_data = "N",
         stratified_column_number = 0,
         remove_data_correlations_greater_than = 0.99,
         remove_ensemble_correlations_greater_than = 0.99,
         use_parallel = "Y",
         train_amount = 0.50,
         test_amount = 0.25,
         validation_amount = 0.25)
end_time <- Sys.time()
duration <- end_time - start_time
duration
warnings()
```

Comments:

The data is stored on my local computer, since it is 150 MB in size.<br>
The seed was set at 12345.<br>
The models are only resampled twice, because the seed is set, so the results will be identical for all runs.<br>
All the plots and models are saved to be used in writing reports and summaries.<br>
Both data and ensemble correlations > 0.99 are removed, thus removing the possibility of perfectly correlated features.<br>
The process is timed.<br>

<h4>Everything ran in 3.016171 minutes without any errors, warnings, or issues. The majority of the time was spent saving the image files and trained models. If the models and images are not saved, everything completed in 1.776362 mins, a substnatially shorter run time.</h4>

<h2>Step 3: Summary results on the holdout data</h2>

|Model                          | Area Under The Curve| True Positive Rate (Sensitivity)| True Negative Rate (Specificity)| False Positive Rate (Type I Error)| False Negative Rate (Type II Error)| Positive Predictive Value (Precision)| Negative Predictive Value| F1 Score| Duration| Duration sd|
|:------------------------------|----------------:|----------------------------------:|----------------------------------:|------------------------------------:|-------------------------------------:|---------------------------------------:|-------------------------:|--------:|--------:|-----------:|
|XGBoost                        |           1.0000|                             1.0000|                             1.0000|                               0.0000|                                0.0000|                                  1.0000|                    1.0000|   1.0000|   0.7763|      0.0264|
|Ensemble C50                   |           1.0000|                             1.0000|                             1.0000|                               0.0000|                                0.0000|                                  1.0000|                    1.0000|   1.0000|   1.0697|      0.0044|
|Ensemble Elastic               |           1.0000|                             1.0000|                             1.0000|                               0.0000|                                0.0000|                                  1.0000|                    1.0000|   1.0000|   1.0019|      0.0601|
|Ensemble XGBoost               |           1.0000|                             1.0000|                             1.0000|                               0.0000|                                0.0000|                                  1.0000|                    1.0000|   1.0000|   0.1817|      0.0087|
|Generalized Additive Models    |           0.9104|                             0.8611|                             0.9597|                               0.0403|                                0.1389|                                  0.0325|                    0.9998|   0.0626|   1.1270|      0.1512|
|Ensemble Neuralnet             |           0.9102|                             0.8501|                             0.9587|                               0.0413|                                0.1499|                                  0.0267|                    0.9998|   0.0517|   0.2246|      0.1515|
|Flexible Discriminant Analysis |           0.8340|                             0.6526|                             0.9999|                               0.0001|                                0.3474|                                  0.9058|                    0.9995|   0.7515|   0.6751|      0.1948|
|Ensemble GLM                   |           0.7766|                             0.5404|                             1.0000|                               0.0000|                                0.4596|                                  0.9286|                    0.9994|   0.6829|   0.6606|      0.0096|
|Elastic                        |           0.7734|                             0.5393|                             0.5393|                               0.0000|                                0.4607|                                  0.7425|                    0.9993|   0.6248|   4.9798|      0.3573|
|Generalized Linear Models      |           0.7712|                             0.5325|                             0.9997|                               0.0003|                                0.4675|                                  0.7386|                    0.9993|   0.6188|   5.0943|      0.4767|
|Neuralnet                      |           0.7543|                             0.9831|                             0.5265|                               0.4735|                                0.0169|                                  0.0033|                    0.9999|   0.0065|   2.1872|      0.0810|
|Gradient Boosted               |           0.5067|                             0.0145|                             1.0000|                               0.0000|                                0.9855|                                  0.3000|                    0.9985|   0.0277|   5.6235|      0.6494|

Comments:
One regular model (XGBoost) and three ensembles (Ensemble C50, Ensemble Elastic and Ensemble XGBoost) had 100% accuracy as measured by the AUC score on the holdout data. This can be viewed by looking at the ROC (Receiver Operating Curves) for the data, with the Area Under the Curve (AUC) noted for each graph:

![ROC Curves](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Big_credit_card_ROC_curves.jpg)<br>

<h2>Step 5: Strongest evidence based recommendations to fight credit card fraud based on the data set</h2>

<h2>Step 6: Conclusions</h2>

<h2>Step 7: References</h2>

something to delete.
