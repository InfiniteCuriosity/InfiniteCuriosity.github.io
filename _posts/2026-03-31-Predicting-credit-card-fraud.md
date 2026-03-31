<b><h1>How data science can help fight credit card fraud</h1></b>

<h3>Introduction</h3>
Credit card fraud is a huge problem in the retail sector, with total losses in the billions of dollars each year. This blog post will highlight how you can understand the process of fighting credit card fraud with data science.

<h4>The data set (it's enormous!)</h4>
One of the largest credit card fraud data sets was posted at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.

The data:<br>
• Reports transactions "made by credit cards in September 2013 by European cardholders."<br>
• Contains 492 frauds out of 284,807 transactions (~ 0.172% of the total)<br>
• Has been anonymized, but the label (fraud or genuine) and dollar amounts are accurate.<br>
• The data set contains 284,807 rows, and 31 variables. It is 66.3 MB in size.<br>
• The data is transformed by Principal Components Analysis, due to confidentiality of the original data.<br>

<h4>Why is this so difficult to solve?</h4>
Most examples of data have large results that are easy to see. However, fraud data typically only shows up in a small fraction of 1% of the transactions. In our case it's less than 2/10 of 1% of the transactions. We are looking for the proverbial needle in the haystack.

<h4>How big is the problem?</h4>
According to a paper from the European Central Bank on payment fraud dated 15 December, 2025, "In 2024 payment fraud rate in European Economic Area stable at around 0.002% of total value of transactions in a calendar year." (https://www.ecb.europa.eu/press/pr/date/2025/html/ecb.pr251215~e133d9d683.en.html) This amounted to a total of €4.2 billion in 2024, which is an increase from € 3.5 billion in 2023 and € 3.4 billion in 2022.<br>
<br>
The report from the European Central Bank breaks down the numbers as follows:

>"For 2024, the overall losses for credit transfers were €2.200 billion (a year-on-year increase of 16%), and for card payments with cards issued in the EU/EEA they were €1.329 billion (a year-on-year increase of 29%). For credit transfers, payment service users bore approximately 85% of total fraud losses in 2024, mainly as a result of scams that tricked users into initiating fraudulent transactions."

<h2>Step 1: Looking at the data (Boxplots, histograms, head of the data, summary of the data, correlation of each feature to the target) </h2>
We will begin by looking at boxplots of the data:<br>

![Boxplots](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Big_credit_card_boxplots.jpg)
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

![Histograms](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Big_credit_card_histograms.jpg)<br>

The histograms confirm what the boxplots showed: The majority of the values for each feature are narrowly spread out. Only the time feature has data which is widely spread out.

<h4>Step 1c: Head of the data</h4>
Head of the data. 

| Time|         V1|         V2|        V3|         V4|         V5|         V6|         V7|         V8|         V9|        V10|        V11|        V12|        V13|        V14|        V15|        V16|        V17|        V18|        V19|        V20|        V21|        V22|        V23|        V24|        V25|        V26|        V27|        V28|  y|
|----:|----------:|----------:|---------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|--:|
|    0| -1.3598071| -0.0727812| 2.5363467|  1.3781552| -0.3383208|  0.4623878|  0.2395986|  0.0986979|  0.3637870|  0.0907942| -0.5515995| -0.6178009| -0.9913898| -0.3111694|  1.4681770| -0.4704005|  0.2079712|  0.0257906|  0.4039930|  0.2514121| -0.0183068|  0.2778376| -0.1104739|  0.0669281|  0.1285394| -0.1891148|  0.1335584| -0.0210531|  0|
|    0|  1.1918571|  0.2661507| 0.1664801|  0.4481541|  0.0600176| -0.0823608| -0.0788030|  0.0851017| -0.2554251| -0.1669744|  1.6127267|  1.0652353|  0.4890950| -0.1437723|  0.6355581|  0.4639170| -0.1148047| -0.1833613| -0.1457830| -0.0690831| -0.2257752| -0.6386720|  0.1012880| -0.3398465|  0.1671704|  0.1258945| -0.0089831|  0.0147242|  0|
|    1| -1.3583541| -1.3401631| 1.7732093|  0.3797796| -0.5031981|  1.8004994|  0.7914610|  0.2476758| -1.5146543|  0.2076429|  0.6245015|  0.0660837|  0.7172927| -0.1659459|  2.3458649| -2.8900832|  1.1099694| -0.1213593| -2.2618571|  0.5249797|  0.2479982|  0.7716794|  0.9094123| -0.6892810| -0.3276418| -0.1390966| -0.0553528| -0.0597518|  0|
|    1| -0.9662717| -0.1852260| 1.7929933| -0.8632913| -0.0103089|  1.2472032|  0.2376089|  0.3774359| -1.3870241| -0.0549519| -0.2264873|  0.1782282|  0.5077569| -0.2879237| -0.6314181| -1.0596472| -0.6840928|  1.9657750| -1.2326220| -0.2080378| -0.1083005|  0.0052736| -0.1903205| -1.1755753|  0.6473760| -0.2219288|  0.0627228|  0.0614576|  0|
|    2| -1.1582331|  0.8777368| 1.5487178|  0.4030339| -0.4071934|  0.0959215|  0.5929407| -0.2705327|  0.8177393|  0.7530744| -0.8228429|  0.5381956|  1.3458516| -1.1196698|  0.1751211| -0.4514492| -0.2370332| -0.0381948|  0.8034869|  0.4085424| -0.0094307|  0.7982785| -0.1374581|  0.1412670| -0.2060096|  0.5022922|  0.2194222|  0.2151531|  0|
|    2| -0.4259659|  0.9605230| 1.1411093| -0.1682521|  0.4209869| -0.0297276|  0.4762009|  0.2603143| -0.5686714| -0.3714072|  1.3412620|  0.3598938| -0.3580907| -0.1371337|  0.5176168|  0.4017259| -0.0581328|  0.0686531| -0.0331938|  0.0849677| -0.2082535| -0.5598248| -0.0263977| -0.3714266| -0.2327938|  0.1059148|  0.2538442|  0.0810803|  0|


<h4>Step 1d: Data summary</h4>

|  Measure |     Time      |      V1          |      V2          |      V3         |      V4         |      V5           |      V6         |      V7         |      V8          |      V9          |     V10          |     V11         |     V12         |     V13         |     V14         |     V15         |     V16          |     V17          |     V18          |     V19          |     V20          |     V21          |     V22           |     V23          |     V24         |     V25          |     V26         |     V27           |     V28          |      y          |
|:--|:--------------|:-----------------|:-----------------|:----------------|:----------------|:------------------|:----------------|:----------------|:-----------------|:-----------------|:-----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:------------------|:-----------------|:----------------|:-----------------|:----------------|:------------------|:-----------------|:----------------|
| Min  | 0 | -56.40751 | 72.71573 | -48.3256 | -5.68317 | -113.74331 | -26.1605 | -43.5572 | -73.21672 | -13.43407 | -24.58826 | -4.79747 | -18.6837 | -5.79188 | -19.2143 | -4.49894 | -14.12985 | -25.16280 | -9.498746 | -7.213527 | -54.49772 | -34.83038 | -10.933144 | -44.80774 | -2.83663 | -10.29540 | -2.60455 | -22.565679 | -15.43008 | 0.000000 |
|1st Qu   | 54202 | -0.92037 | -0.59855 | -0.8904 | -0.84864 | -0.69160 | -0.7683 | -0.5541 | -0.20863 | -0.64310 | -0.53543 | -0.76249 | -0.4056 | -0.64854 | -0.4256 | -0.58288 | -0.46804 | -0.48375 | -0.498850 | -0.456299 | -0.21172 | -0.22839 | -0.542350 | -0.16185 | -0.35459 | -0.31715 | -0.32698 | -0.070839 | -0.05296 |0.000000 |
|Median   | 84692 | 0.01811 | 0.06549 | 0.1798 | -0.01985 | -0.05434 | -0.2742 | 0.0401 | 0.02236 | -0.05143 | -0.09292 | -0.03276 | 0.1400 | -0.01357 | 0.0506 | 0.04807 | 0.06641 | -0.06568 | -0.003636 | 0.003735 | -0.06248 | -0.02945 | 0.006782 | -0.01119 | 0.04098 | 0.01659 | -0.05214 |  0.001342 | 0.01124 | 0.000000 |
|Mean   | 94814 | 0.00000 | 0.00000 | 0.0000 | 0.00000 | 0.00000 | 0.0000 | 0.0000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.0000 | 0.00000 | 0.0000 | 0.00000 | 0.00000 | 0.00000 | 0.000000 | 0.000000 | 0.00000 | 0.00000 | 0.000000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.000000 | 0.00000 | 0.001727 |
|3rd Qu   | 139320 | 1.31564 | 0.80372 | 1.0272 | 0.74334 | 0.61193 | 0.3986 | 0.5704 | 0.32735 | 0.59714 | 0.45392 | 0.73959 | 0.6182 | 0.66251 | 0.4931 | 0.64882 | 0.52330 | 0.39968 | 0.500807 | 0.458949 | 0.13304 | 0.18638 | 0.528554 | 0.14764 | 0.43953 | 0.35072 | 0.24095 | 0.091045 | 0.07828 | 0.000000 |
|Max   | 172792 | 2.45493 | 22.05773 | 9.3826 | 16.87534 | 34.80167 | 73.3016 | 120.5895 | 20.00721 | 15.59499 | 23.74514 | 12.01891 | 7.8484 | 7.12688 | 10.5268 | 8.87774 | 17.31511 | 9.25353 | 5.041069 | 5.591971 | 39.42090 | 27.20284 | 10.503090 | 22.52841 | 4.58455 | 7.51959 | 3.51735 | 31.612198 | 33.84781 | 1.000000 |

<h4>Correlation of the target (Class) to each variable (closer to 1.00 is better)</h4>

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

<h4>How the LogisticEnsembles package makes this much faster and easier to solve, while maintaining a very high level of accuracy on the holdout data:</h4>

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

<h4>Which variable are the strongest predictors?</h4>

We begin by looking at a table of the importance of the predictors. Several points stand out:

• Ten of the 12 largest predictors have a negative influence on the result
• The ten most important variable account for 77.32% of the total, and more than 66.81% of the value is negative.
• It is strongly recommended to look at variables V17, V14, V12, V10, V16, V7, and V3, as they are all negative, and constitute more than 63% of the total.

![Variable importance table](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Big_Credit_Card_Fraud_Variable_Importance_Report.jpg)<br>

![Variable Importance Barchart](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Big_credit_variable_importance_barchart.jpg)<br>

Neither the original data nor the ensemble had strongly correlated predictors:

|     |    Time|      V1|      V2|      V3|      V4|      V5|      V6|      V7|      V8|      V9|     V10|     V11|     V12|     V13|     V14|     V15|     V16|     V17|     V18|    V19|     V20|    V21|    V22|     V23|     V24|     V25|     V26|     V27|     V28|       y|
|:----|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|------:|-------:|------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
|Time |  1.0000|  0.1174| -0.0106| -0.4196| -0.1053|  0.1731| -0.0630|  0.0847| -0.0369| -0.0087|  0.0306| -0.2477|  0.1243| -0.0659| -0.0988| -0.1835|  0.0119| -0.0733|  0.0904| 0.0290| -0.0509| 0.0447| 0.1441|  0.0511| -0.0162| -0.2331| -0.0414| -0.0051| -0.0094| -0.0123|
|V1   |  0.1174|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.1013|
|V2   | -0.0106|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0913|
|V3   | -0.4196|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.1930|
|V4   | -0.1053|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.1334|
|V5   |  0.1731|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.0950|
|V6   | -0.0630|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.0436|
|V7   |  0.0847|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.1873|
|V8   | -0.0369|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0199|
|V9   | -0.0087|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.0977|
|V10  |  0.0306|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.2169|
|V11  | -0.2477|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.1549|
|V12  |  0.1243|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.2606|
|V13  | -0.0659|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.0046|
|V14  | -0.0988|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.3025|
|V15  | -0.1835|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.0042|
|V16  |  0.0119|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.1965|
|V17  | -0.0733|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.3265|
|V18  |  0.0904|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.1115|
|V19  |  0.0290|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 1.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0348|
|V20  | -0.0509|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  1.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0201|
|V21  |  0.0447|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 1.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0404|
|V22  |  0.1441|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0008|
|V23  |  0.0511|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.0027|
|V24  | -0.0162|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.0072|
|V25  | -0.2331|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0033|
|V26  | -0.0414|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0045|
|V27  | -0.0051|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0176|
|V28  | -0.0094|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0095|
|y    | -0.0123| -0.1013|  0.0913| -0.1930|  0.1334| -0.0950| -0.0436| -0.1873|  0.0199| -0.0977| -0.2169|  0.1549| -0.2606| -0.0046| -0.3025| -0.0042| -0.1965| -0.3265| -0.1115| 0.0348|  0.0201| 0.0404| 0.0008| -0.0027| -0.0072|  0.0033|  0.0045|  0.0176|  0.0095|  1.0000|

The correlation table for the ensemble:


R version 4.5.2 (2025-10-31) -- "[Not] Part in a Rumble"
Copyright (C) 2025 The R Foundation for Statistical Computing
Platform: aarch64-apple-darwin20

R is free software and comes with ABSOLUTELY NO WARRANTY.
You are welcome to redistribute it under certain conditions.
Type 'license()' or 'licence()' for distribution details.

  Natural language support but running in an English locale

R is a collaborative project with many contributors.
Type 'contributors()' for more information and
'citation()' on how to cite R or R packages in publications.

Type 'demo()' for some demos, 'help()' for on-line help, or
'help.start()' for an HTML browser interface to help.
Type 'q()' to quit R.

> # https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
> options(rgl.useNULL = TRUE, rgl.printRglwidget = TRUE) #https://github.com/dmurdoch/rgl/issues/488
> library(rgl)
> library(LogisticEnsembles)
Loading required package: adabag
Loading required package: rpart
Loading required package: caret
Loading required package: ggplot2
Loading required package: lattice
Loading required package: foreach
Loading required package: doParallel
Loading required package: iterators
Loading required package: parallel
Loading required package: arm
Loading required package: MASS
Loading required package: Matrix
Loading required package: lme4

arm (Version 1.15-2, built: 2026-3-22)

Working directory is /Users/russellconte/Library/Mobile Documents/com~apple~CloudDocs/Documents/Machine Learning templates in R

Loading required package: brnn
Loading required package: Formula
Loading required package: truncnorm
Package 'brnn', 0.9.3 (2023-11-05)
Type 'help(brnn)' for summary information
Loading required package: C50
Loading required package: car
Loading required package: carData
Registered S3 method overwritten by 'car':
  method           from
  na.action.merMod lme4

Attaching package: ‘car’

The following object is masked from ‘package:arm’:

    logit

Loading required package: corrplot
corrplot 0.95 loaded

Attaching package: ‘corrplot’

The following object is masked from ‘package:arm’:

    corrplot

Loading required package: dplyr

Attaching package: ‘dplyr’

The following object is masked from ‘package:car’:

    recode

The following object is masked from ‘package:MASS’:

    select

The following objects are masked from ‘package:stats’:

    filter, lag

The following objects are masked from ‘package:base’:

    intersect, setdiff, setequal, union

Loading required package: e1071

Attaching package: ‘e1071’

The following object is masked from ‘package:ggplot2’:

    element

Loading required package: gam
Loading required package: splines
Loaded gam 1.22-7

Loading required package: gbm
Loaded gbm 2.2.3
This version of gbm is no longer under development. Consider transitioning to gbm3, https://github.com/gbm-developers/gbm3
Loading required package: ggplotify
Loading required package: glmnet
Loaded glmnet 4.1-10
Loading required package: gridExtra

Attaching package: ‘gridExtra’

The following object is masked from ‘package:dplyr’:

    combine

Loading required package: gt
Loading required package: htmltools
Loading required package: htmlwidgets
Loading required package: ipred

Attaching package: ‘ipred’

The following object is masked from ‘package:adabag’:

    bagging

Loading required package: klaR
Loading required package: MachineShop

Attaching package: ‘MachineShop’

The following objects are masked from ‘package:caret’:

    calibration, lift, precision, recall, rfe, sensitivity, specificity

The following object is masked from ‘package:stats’:

    ppr

Loading required package: magrittr
Loading required package: mda
Loading required package: class
Loaded mda 0.5-5


Attaching package: ‘mda’

The following object is masked from ‘package:MachineShop’:

    confusion

Loading required package: nnet
Loading required package: olsrr

Attaching package: ‘olsrr’

The following object is masked from ‘package:MASS’:

    cement

The following object is masked from ‘package:datasets’:

    rivers

Loading required package: pls

Attaching package: ‘pls’

The following object is masked from ‘package:corrplot’:

    corrplot

The following objects are masked from ‘package:arm’:

    coefplot, corrplot

The following object is masked from ‘package:caret’:

    R2

The following object is masked from ‘package:stats’:

    loadings

Loading required package: pROC
Type 'citation("pROC")' for a citation.

Attaching package: ‘pROC’

The following object is masked from ‘package:MachineShop’:

    auc

The following objects are masked from ‘package:stats’:

    cov, smooth, var

Loading required package: purrr

Attaching package: ‘purrr’

The following object is masked from ‘package:magrittr’:

    set_names

The following object is masked from ‘package:MachineShop’:

    lift

The following object is masked from ‘package:car’:

    some

The following objects are masked from ‘package:foreach’:

    accumulate, when

The following object is masked from ‘package:caret’:

    lift

Loading required package: randomForest
randomForest 4.7-1.2
Type rfNews() to see new features/changes/bug fixes.

Attaching package: ‘randomForest’

The following object is masked from ‘package:gridExtra’:

    combine

The following object is masked from ‘package:dplyr’:

    combine

The following object is masked from ‘package:ggplot2’:

    margin

Loading required package: ranger
ranger 0.18.0 using 2 threads (default). Change with num.threads in ranger() and predict(), options(Ncpus = N), options(ranger.num.threads = N) or environment variable R_RANGER_NUM_THREADS.

Attaching package: ‘ranger’

The following object is masked from ‘package:randomForest’:

    importance

Loading required package: reactable
Loading required package: readr
Loading required package: scales

Attaching package: ‘scales’

The following object is masked from ‘package:readr’:

    col_factor

The following object is masked from ‘package:purrr’:

    discard

The following object is masked from ‘package:arm’:

    rescale

Loading required package: tidyr

Attaching package: ‘tidyr’

The following object is masked from ‘package:magrittr’:

    extract

The following objects are masked from ‘package:Matrix’:

    expand, pack, unpack

Loading required package: vip

Attaching package: ‘vip’

The following object is masked from ‘package:utils’:

    vi

Loading required package: xgboost
> 
> start_time <- Sys.time()
> Logistic(data = read.csv('/Users/russellconte/creditcard.csv'),
+          colnum = 31,
+          numresamples = 2,
+          positive_rate = 0.001727486,
+          remove_VIF_greater_than <- 5.00,
+          save_all_trained_models = "Y",
+          save_all_plots = "Y",
+          set_seed = "Y",
+          how_to_handle_strings = 0,
+          do_you_have_new_data = "N",
+          stratified_column_number = 0,
+          remove_data_correlations_greater_than = 0.99,
+          remove_ensemble_correlations_greater_than = 0.99,
+          use_parallel = "Y",
+          train_amount = 0.50,
+          test_amount = 0.25,
+          validation_amount = 0.25)
Which integer would you like to use for the seed? 12345
Width of the graphics: end_time <- Sys.time()
Height of the graphics: duration <- end_time - start_time
Which units? You may use in, cm, mm or px. duration
What multiplicative scaling factor? warnings()
Which device to use? You may enter eps, jpeg, pdf, png, svg or tiff: 
Warning messages:
1: In Logistic(data = read.csv("/Users/russellconte/creditcard.csv"),  :
  NAs introduced by coercion
2: In Logistic(data = read.csv("/Users/russellconte/creditcard.csv"),  :
  NAs introduced by coercion
3: In Logistic(data = read.csv("/Users/russellconte/creditcard.csv"),  :
  NAs introduced by coercion

> # https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
> options(rgl.useNULL = TRUE, rgl.printRglwidget = TRUE) #https://github.com/dmurdoch/rgl/issues/488
> library(rgl)
> library(LogisticEnsembles)
> 
> start_time <- Sys.time()
> Logistic(data = read.csv('/Users/russellconte/creditcard.csv'),
+          colnum = 31,
+          numresamples = 2,
+          positive_rate = 0.001727486,
+          remove_VIF_greater_than <- 5.00,
+          save_all_trained_models = "Y",
+          save_all_plots = "Y",
+          set_seed = "Y",
+          how_to_handle_strings = 0,
+          do_you_have_new_data = "N",
+          stratified_column_number = 0,
+          remove_data_correlations_greater_than = 0.99,
+          remove_ensemble_correlations_greater_than = 0.99,
+          use_parallel = "Y",
+          train_amount = 0.50,
+          test_amount = 0.25,
+          validation_amount = 0.25)
Which integer would you like to use for the seed? 12345
Width of the graphics: 16
Height of the graphics: 10
Which units? You may use in, cm, mm or px. in
What multiplicative scaling factor? 1
Which device to use? You may enter eps, jpeg, pdf, png, svg or tiff: pdf
Plot resolution. Applies only to raster output types (jpeg, png, tiff): 150

Resampling number 1 of 2,

Working on Elastic
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on Flexible Discriminant Analysis (FDA)
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on Generalized Additive Models (GAM)
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on Gradient Boosted
Using 100 trees...

Using 100 trees...

Using 100 trees...

Setting levels: control = 0, case = 1
Setting direction: controls > cases
Working on glmnet
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on Neuralnet
# weights:  30
initial  value 125784611286082.687500 
iter  10 value 25999732952678.914062
iter  20 value 25163180247258.300781
iter  30 value 8230296571524.081055
final  value 132.032527 
converged
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on XGBoost
Setting levels: control = 0, case = 1
Setting direction: controls < cases

Working on the Ensembles section

Working on Ensemble C50
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on ensemble_elastic
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on ensemble_glmnet
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on Ensemble Neuralnet
# weights:  7
initial  value 4940.186067 
iter  10 value 51.190365
final  value 41.105578 
converged
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on Ensembles using XGBoost (XGB)
[1]	ensemble_train-rmse:0.029866	ensemble_test-rmse:0.027833 
[2]	ensemble_train-rmse:0.020975	ensemble_test-rmse:0.019547 
[3]	ensemble_train-rmse:0.014731	ensemble_test-rmse:0.013728 
[4]	ensemble_train-rmse:0.010346	ensemble_test-rmse:0.009641 
[5]	ensemble_train-rmse:0.007266	ensemble_test-rmse:0.006771 
[6]	ensemble_train-rmse:0.005103	ensemble_test-rmse:0.004755 
[7]	ensemble_train-rmse:0.003584	ensemble_test-rmse:0.003340 
[8]	ensemble_train-rmse:0.002517	ensemble_test-rmse:0.002345 
[9]	ensemble_train-rmse:0.001768	ensemble_test-rmse:0.001647 
[10]	ensemble_train-rmse:0.001241	ensemble_test-rmse:0.001157 
[11]	ensemble_train-rmse:0.000872	ensemble_test-rmse:0.000812 
[12]	ensemble_train-rmse:0.000612	ensemble_test-rmse:0.000571 
[13]	ensemble_train-rmse:0.000430	ensemble_test-rmse:0.000401 
[14]	ensemble_train-rmse:0.000302	ensemble_test-rmse:0.000281 
[15]	ensemble_train-rmse:0.000212	ensemble_test-rmse:0.000198 
[16]	ensemble_train-rmse:0.000149	ensemble_test-rmse:0.000139 
[17]	ensemble_train-rmse:0.000105	ensemble_test-rmse:0.000097 
[18]	ensemble_train-rmse:0.000073	ensemble_test-rmse:0.000068 
[19]	ensemble_train-rmse:0.000052	ensemble_test-rmse:0.000048 
[20]	ensemble_train-rmse:0.000036	ensemble_test-rmse:0.000034 
[21]	ensemble_train-rmse:0.000025	ensemble_test-rmse:0.000024 
[22]	ensemble_train-rmse:0.000018	ensemble_test-rmse:0.000017 
[23]	ensemble_train-rmse:0.000013	ensemble_test-rmse:0.000012 
[24]	ensemble_train-rmse:0.000009	ensemble_test-rmse:0.000008 
[25]	ensemble_train-rmse:0.000006	ensemble_test-rmse:0.000006 
[26]	ensemble_train-rmse:0.000004	ensemble_test-rmse:0.000004 
[27]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[28]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[29]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[30]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[31]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[32]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[33]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[34]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[35]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[36]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[37]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[38]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[39]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[40]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[41]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[42]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[43]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[44]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[45]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[46]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[47]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[48]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[49]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[50]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[51]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[52]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[53]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[54]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[55]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[56]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[57]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[58]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[59]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[60]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[61]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[62]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[63]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[64]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[65]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[66]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[67]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[68]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[69]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[70]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
Setting levels: control = 0, case = 1
Setting direction: controls < cases

Resampling number 2 of 2,

Working on Elastic
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on Flexible Discriminant Analysis (FDA)
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on Generalized Additive Models (GAM)
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on Gradient Boosted
Using 100 trees...

Using 100 trees...

Using 100 trees...

Setting levels: control = 0, case = 1
Setting direction: controls > cases
Working on glmnet
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on Neuralnet
# weights:  30
initial  value 125784611286082.687500 
iter  10 value 25999732952678.914062
iter  20 value 25163180247258.300781
iter  30 value 8230296571524.081055
final  value 132.032527 
converged
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on XGBoost
Setting levels: control = 0, case = 1
Setting direction: controls < cases

Working on the Ensembles section

Working on Ensemble C50
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on ensemble_elastic
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on ensemble_glmnet
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on Ensemble Neuralnet
# weights:  7
initial  value 4940.186067 
iter  10 value 51.190365
final  value 41.105578 
converged
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Working on Ensembles using XGBoost (XGB)
[1]	ensemble_train-rmse:0.029866	ensemble_test-rmse:0.027833 
[2]	ensemble_train-rmse:0.020975	ensemble_test-rmse:0.019547 
[3]	ensemble_train-rmse:0.014731	ensemble_test-rmse:0.013728 
[4]	ensemble_train-rmse:0.010346	ensemble_test-rmse:0.009641 
[5]	ensemble_train-rmse:0.007266	ensemble_test-rmse:0.006771 
[6]	ensemble_train-rmse:0.005103	ensemble_test-rmse:0.004755 
[7]	ensemble_train-rmse:0.003584	ensemble_test-rmse:0.003340 
[8]	ensemble_train-rmse:0.002517	ensemble_test-rmse:0.002345 
[9]	ensemble_train-rmse:0.001768	ensemble_test-rmse:0.001647 
[10]	ensemble_train-rmse:0.001241	ensemble_test-rmse:0.001157 
[11]	ensemble_train-rmse:0.000872	ensemble_test-rmse:0.000812 
[12]	ensemble_train-rmse:0.000612	ensemble_test-rmse:0.000571 
[13]	ensemble_train-rmse:0.000430	ensemble_test-rmse:0.000401 
[14]	ensemble_train-rmse:0.000302	ensemble_test-rmse:0.000281 
[15]	ensemble_train-rmse:0.000212	ensemble_test-rmse:0.000198 
[16]	ensemble_train-rmse:0.000149	ensemble_test-rmse:0.000139 
[17]	ensemble_train-rmse:0.000105	ensemble_test-rmse:0.000097 
[18]	ensemble_train-rmse:0.000073	ensemble_test-rmse:0.000068 
[19]	ensemble_train-rmse:0.000052	ensemble_test-rmse:0.000048 
[20]	ensemble_train-rmse:0.000036	ensemble_test-rmse:0.000034 
[21]	ensemble_train-rmse:0.000025	ensemble_test-rmse:0.000024 
[22]	ensemble_train-rmse:0.000018	ensemble_test-rmse:0.000017 
[23]	ensemble_train-rmse:0.000013	ensemble_test-rmse:0.000012 
[24]	ensemble_train-rmse:0.000009	ensemble_test-rmse:0.000008 
[25]	ensemble_train-rmse:0.000006	ensemble_test-rmse:0.000006 
[26]	ensemble_train-rmse:0.000004	ensemble_test-rmse:0.000004 
[27]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[28]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[29]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[30]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[31]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[32]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[33]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[34]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[35]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[36]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[37]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[38]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[39]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[40]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[41]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[42]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[43]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[44]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[45]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[46]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[47]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[48]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[49]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[50]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[51]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[52]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[53]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[54]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[55]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[56]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[57]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[58]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[59]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[60]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[61]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[62]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[63]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[64]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[65]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[66]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[67]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[68]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[69]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
[70]	ensemble_train-rmse:0.000003	ensemble_test-rmse:0.000003 
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls > cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
Setting levels: control = 0, case = 1
Setting direction: controls < cases
$`Head of data`

$`Summary tables`
$`Summary tables`$Elastic
                        
elastic_test_predictions      0      1
                       0 284280    202
                       1     80    244

$`Summary tables`$`Fixture Discrmininant Analysis`
                    
fda_test_predictions      0      1
                   0 284324    148
                   1     36    298

$`Summary tables`$`Generalized Additive Methods`
                    
gam_test_predictions      0      1
                   0 272906     62
                   1  11454    384

$`Summary tables`$`Generalized Linear Models`
                       
glmnet_test_predictions      0      1
                      0 284280    204
                      1     80    242

$`Summary tables`$`Gradient Boosted`
                   
gb_test_predictions      0      1
                  0 284346    440
                  1     14      6

$`Summary tables`$Neuralnet
                          
neuralnet_test_predictions      0      1
                         0 149704      8
                         1 134656    438

$`Summary tables`$XGBoost
                    
xgb_test_predictions      0      1
                   0 284360      0
                   1      0    446

$`Summary tables`$`Ensemble C50`
                      ensemble_y_test
ensemble_C50_test_pred      0      1
                     0 142214      0
                     1      0    188

$`Summary tables`$`Ensemble Elastic`
                          ensemble_y_test
ensemble_elastic_test_pred      0      1
                         0 142214      0
                         1      0    188

$`Summary tables`$`Ensemble GLM`
                         ensemble_y_test
ensemble_glmnet_test_pred      0      1
                        0 142208     84
                        1      6    104

$`Summary tables`$`Ensemble Neuralnet`
                            ensemble_y_test
ensemble_neuralnet_test_pred      0      1
                           0 136344     26
                           1   5870    162

$`Summary tables`$`Ensemble XGBoost`
                      ensemble_y_test
ensemble_xgb_test_pred      0      1
                     0 142214      0
                     1      0    188


$`AUC fixed scales`

$`AUC free scales`

$`Duration barchart`

$`ROC curves`

$Boxplots

$Barchart

$`Barchart percentage`

$`Correlation table`

$`VIF results`

$`True positive rate fixed scales`

$`True positive rate free scales`

$`True negative rate fixed scales`

$`True negative rate free scales`

$`False positive rate fixed scales`

$`False positive rate free scales`

$`False negative rate fixed scales`

$`False negative rate free scales`

$`F1 score fixed scales`

$`F1 score free scales`

$`Stratified sampling report`
[1] 0

$`Positive predictive value fixed scales`

$`Positive predictive value free scales`

$`Negative predictive value fixed scales`

$`Negative predictive value free scales`

$`Ensemble Correlation`

$`Ensemble head`

$`Data Summary`

$`Holdout results`

$`Outlier list`

$`Variable importance`

$`Variable importance barchart`

$`How to handle strings`
[1] 0

$`Train amount`
[1] 0.5

$`Test amount`
[1] 0.25

$`Validation amount`
[1] 0.25

> end_time <- Sys.time()
> duration <- end_time - start_time
> duration
Time difference of 7.947203 mins
> warnings()
Warning messages:
1: In Logistic(data = read.csv("/Users/russellconte/creditcard.csv"),  ... :
  NAs introduced by coercion
2: In Logistic(data = read.csv("/Users/russellconte/creditcard.csv"),  ... :
  NAs introduced by coercion
3: In Logistic(data = read.csv("/Users/russellconte/creditcard.csv"),  ... :
  NAs introduced by coercion

> correlationtablef45a68ee68 <- readRDS("/private/var/folders/2c/hvxqjjcn3sq5lcshxhv8wnsw0000gp/T/RtmpsaJlZC/correlationtablef45a68ee68.RDS")
> correlationtablef45a68ee68
        Time      V1      V2      V3      V4      V5      V6      V7      V8      V9     V10     V11     V12     V13     V14     V15     V16
Time  1.0000  0.1174 -0.0106 -0.4196 -0.1053  0.1731 -0.0630  0.0847 -0.0369 -0.0087  0.0306 -0.2477  0.1243 -0.0659 -0.0988 -0.1835  0.0119
V1    0.1174  1.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V2   -0.0106  0.0000  1.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V3   -0.4196  0.0000  0.0000  1.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V4   -0.1053  0.0000  0.0000  0.0000  1.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V5    0.1731  0.0000  0.0000  0.0000  0.0000  1.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V6   -0.0630  0.0000  0.0000  0.0000  0.0000  0.0000  1.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V7    0.0847  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  1.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V8   -0.0369  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  1.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V9   -0.0087  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  1.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V10   0.0306  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  1.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V11  -0.2477  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  1.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V12   0.1243  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  1.0000  0.0000  0.0000  0.0000  0.0000
V13  -0.0659  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  1.0000  0.0000  0.0000  0.0000
V14  -0.0988  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  1.0000  0.0000  0.0000
V15  -0.1835  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  1.0000  0.0000
V16   0.0119  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  1.0000
V17  -0.0733  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V18   0.0904  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V19   0.0290  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V20  -0.0509  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V21   0.0447  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V22   0.1441  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V23   0.0511  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V24  -0.0162  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V25  -0.2331  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V26  -0.0414  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V27  -0.0051  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
V28  -0.0094  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000
y    -0.0123 -0.1013  0.0913 -0.1930  0.1334 -0.0950 -0.0436 -0.1873  0.0199 -0.0977 -0.2169  0.1549 -0.2606 -0.0046 -0.3025 -0.0042 -0.1965
         V17     V18    V19     V20    V21    V22     V23     V24     V25     V26     V27     V28       y
Time -0.0733  0.0904 0.0290 -0.0509 0.0447 0.1441  0.0511 -0.0162 -0.2331 -0.0414 -0.0051 -0.0094 -0.0123
V1    0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 -0.1013
V2    0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0913
V3    0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 -0.1930
V4    0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.1334
V5    0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 -0.0950
V6    0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 -0.0436
V7    0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 -0.1873
V8    0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0199
V9    0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 -0.0977
V10   0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 -0.2169
V11   0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.1549
V12   0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 -0.2606
V13   0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 -0.0046
V14   0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 -0.3025
V15   0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 -0.0042
V16   0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 -0.1965
V17   1.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 -0.3265
V18   0.0000  1.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000 -0.1115
V19   0.0000  0.0000 1.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0348
V20   0.0000  0.0000 0.0000  1.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0201
V21   0.0000  0.0000 0.0000  0.0000 1.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0404
V22   0.0000  0.0000 0.0000  0.0000 0.0000 1.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  0.0008
V23   0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  1.0000  0.0000  0.0000  0.0000  0.0000  0.0000 -0.0027
V24   0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  1.0000  0.0000  0.0000  0.0000  0.0000 -0.0072
V25   0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  1.0000  0.0000  0.0000  0.0000  0.0033
V26   0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  1.0000  0.0000  0.0000  0.0045
V27   0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  1.0000  0.0000  0.0176
V28   0.0000  0.0000 0.0000  0.0000 0.0000 0.0000  0.0000  0.0000  0.0000  0.0000  0.0000  1.0000  0.0095
y    -0.3265 -0.1115 0.0348  0.0201 0.0404 0.0008 -0.0027 -0.0072  0.0033  0.0045  0.0176  0.0095  1.0000
> knitr::kable(correlationtablef45a68ee68[, 30])


|     |       x|
|:----|-------:|
|Time | -0.0123|
|V1   | -0.1013|
|V2   |  0.0913|
|V3   | -0.1930|
|V4   |  0.1334|
|V5   | -0.0950|
|V6   | -0.0436|
|V7   | -0.1873|
|V8   |  0.0199|
|V9   | -0.0977|
|V10  | -0.2169|
|V11  |  0.1549|
|V12  | -0.2606|
|V13  | -0.0046|
|V14  | -0.3025|
|V15  | -0.0042|
|V16  | -0.1965|
|V17  | -0.3265|
|V18  | -0.1115|
|V19  |  0.0348|
|V20  |  0.0201|
|V21  |  0.0404|
|V22  |  0.0008|
|V23  | -0.0027|
|V24  | -0.0072|
|V25  |  0.0033|
|V26  |  0.0045|
|V27  |  0.0176|
|V28  |  0.0095|
|y    |  1.0000|
> datasummaryf45c0a538b <- readRDS("/private/var/folders/2c/hvxqjjcn3sq5lcshxhv8wnsw0000gp/T/RtmpsaJlZC/datasummaryf45c0a538b.RDS")
> knitr::kable(datasummaryf45c0a538b)


|   |     Time      |      V1          |      V2          |      V3         |      V4         |      V5           |      V6         |      V7         |      V8          |      V9          |     V10          |     V11         |     V12         |     V13         |     V14         |     V15         |     V16          |     V17          |     V18          |     V19          |     V20          |     V21          |     V22           |     V23          |     V24         |     V25          |     V26         |     V27           |     V28          |      y          |
|:--|:--------------|:-----------------|:-----------------|:----------------|:----------------|:------------------|:----------------|:----------------|:-----------------|:-----------------|:-----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:------------------|:-----------------|:----------------|:-----------------|:----------------|:------------------|:-----------------|:----------------|
|   |Min.   :     0 |Min.   :-56.40751 |Min.   :-72.71573 |Min.   :-48.3256 |Min.   :-5.68317 |Min.   :-113.74331 |Min.   :-26.1605 |Min.   :-43.5572 |Min.   :-73.21672 |Min.   :-13.43407 |Min.   :-24.58826 |Min.   :-4.79747 |Min.   :-18.6837 |Min.   :-5.79188 |Min.   :-19.2143 |Min.   :-4.49894 |Min.   :-14.12985 |Min.   :-25.16280 |Min.   :-9.498746 |Min.   :-7.213527 |Min.   :-54.49772 |Min.   :-34.83038 |Min.   :-10.933144 |Min.   :-44.80774 |Min.   :-2.83663 |Min.   :-10.29540 |Min.   :-2.60455 |Min.   :-22.565679 |Min.   :-15.43008 |Min.   :0.000000 |
|   |1st Qu.: 54202 |1st Qu.: -0.92037 |1st Qu.: -0.59855 |1st Qu.: -0.8904 |1st Qu.:-0.84864 |1st Qu.:  -0.69160 |1st Qu.: -0.7683 |1st Qu.: -0.5541 |1st Qu.: -0.20863 |1st Qu.: -0.64310 |1st Qu.: -0.53543 |1st Qu.:-0.76249 |1st Qu.: -0.4056 |1st Qu.:-0.64854 |1st Qu.: -0.4256 |1st Qu.:-0.58288 |1st Qu.: -0.46804 |1st Qu.: -0.48375 |1st Qu.:-0.498850 |1st Qu.:-0.456299 |1st Qu.: -0.21172 |1st Qu.: -0.22839 |1st Qu.: -0.542350 |1st Qu.: -0.16185 |1st Qu.:-0.35459 |1st Qu.: -0.31715 |1st Qu.:-0.32698 |1st Qu.: -0.070839 |1st Qu.: -0.05296 |1st Qu.:0.000000 |
|   |Median : 84692 |Median :  0.01811 |Median :  0.06549 |Median :  0.1798 |Median :-0.01985 |Median :  -0.05434 |Median : -0.2742 |Median :  0.0401 |Median :  0.02236 |Median : -0.05143 |Median : -0.09292 |Median :-0.03276 |Median :  0.1400 |Median :-0.01357 |Median :  0.0506 |Median : 0.04807 |Median :  0.06641 |Median : -0.06568 |Median :-0.003636 |Median : 0.003735 |Median : -0.06248 |Median : -0.02945 |Median :  0.006782 |Median : -0.01119 |Median : 0.04098 |Median :  0.01659 |Median :-0.05214 |Median :  0.001342 |Median :  0.01124 |Median :0.000000 |
|   |Mean   : 94814 |Mean   :  0.00000 |Mean   :  0.00000 |Mean   :  0.0000 |Mean   : 0.00000 |Mean   :   0.00000 |Mean   :  0.0000 |Mean   :  0.0000 |Mean   :  0.00000 |Mean   :  0.00000 |Mean   :  0.00000 |Mean   : 0.00000 |Mean   :  0.0000 |Mean   : 0.00000 |Mean   :  0.0000 |Mean   : 0.00000 |Mean   :  0.00000 |Mean   :  0.00000 |Mean   : 0.000000 |Mean   : 0.000000 |Mean   :  0.00000 |Mean   :  0.00000 |Mean   :  0.000000 |Mean   :  0.00000 |Mean   : 0.00000 |Mean   :  0.00000 |Mean   : 0.00000 |Mean   :  0.000000 |Mean   :  0.00000 |Mean   :0.001727 |
|   |3rd Qu.:139320 |3rd Qu.:  1.31564 |3rd Qu.:  0.80372 |3rd Qu.:  1.0272 |3rd Qu.: 0.74334 |3rd Qu.:   0.61193 |3rd Qu.:  0.3986 |3rd Qu.:  0.5704 |3rd Qu.:  0.32735 |3rd Qu.:  0.59714 |3rd Qu.:  0.45392 |3rd Qu.: 0.73959 |3rd Qu.:  0.6182 |3rd Qu.: 0.66251 |3rd Qu.:  0.4931 |3rd Qu.: 0.64882 |3rd Qu.:  0.52330 |3rd Qu.:  0.39968 |3rd Qu.: 0.500807 |3rd Qu.: 0.458949 |3rd Qu.:  0.13304 |3rd Qu.:  0.18638 |3rd Qu.:  0.528554 |3rd Qu.:  0.14764 |3rd Qu.: 0.43953 |3rd Qu.:  0.35072 |3rd Qu.: 0.24095 |3rd Qu.:  0.091045 |3rd Qu.:  0.07828 |3rd Qu.:0.000000 |
|   |Max.   :172792 |Max.   :  2.45493 |Max.   : 22.05773 |Max.   :  9.3826 |Max.   :16.87534 |Max.   :  34.80167 |Max.   : 73.3016 |Max.   :120.5895 |Max.   : 20.00721 |Max.   : 15.59499 |Max.   : 23.74514 |Max.   :12.01891 |Max.   :  7.8484 |Max.   : 7.12688 |Max.   : 10.5268 |Max.   : 8.87774 |Max.   : 17.31511 |Max.   :  9.25353 |Max.   : 5.041069 |Max.   : 5.591971 |Max.   : 39.42090 |Max.   : 27.20284 |Max.   : 10.503090 |Max.   : 22.52841 |Max.   : 4.58455 |Max.   :  7.51959 |Max.   : 3.51735 |Max.   : 31.612198 |Max.   : 33.84781 |Max.   :1.000000 |
> knitr::kable(datasummaryf45c0a538b[, c(1:8, 31)])
Error in `[.default`(datasummaryf45c0a538b, , c(1:8, 31)) : 
  subscript out of bounds

> knitr::kable(datasummaryf45c0a538b[, c(1:8, 30)])


|   |     Time      |      V1          |      V2          |      V3         |      V4         |      V5           |      V6         |      V7         |      y          |
|:--|:--------------|:-----------------|:-----------------|:----------------|:----------------|:------------------|:----------------|:----------------|:----------------|
|   |Min.   :     0 |Min.   :-56.40751 |Min.   :-72.71573 |Min.   :-48.3256 |Min.   :-5.68317 |Min.   :-113.74331 |Min.   :-26.1605 |Min.   :-43.5572 |Min.   :0.000000 |
|   |1st Qu.: 54202 |1st Qu.: -0.92037 |1st Qu.: -0.59855 |1st Qu.: -0.8904 |1st Qu.:-0.84864 |1st Qu.:  -0.69160 |1st Qu.: -0.7683 |1st Qu.: -0.5541 |1st Qu.:0.000000 |
|   |Median : 84692 |Median :  0.01811 |Median :  0.06549 |Median :  0.1798 |Median :-0.01985 |Median :  -0.05434 |Median : -0.2742 |Median :  0.0401 |Median :0.000000 |
|   |Mean   : 94814 |Mean   :  0.00000 |Mean   :  0.00000 |Mean   :  0.0000 |Mean   : 0.00000 |Mean   :   0.00000 |Mean   :  0.0000 |Mean   :  0.0000 |Mean   :0.001727 |
|   |3rd Qu.:139320 |3rd Qu.:  1.31564 |3rd Qu.:  0.80372 |3rd Qu.:  1.0272 |3rd Qu.: 0.74334 |3rd Qu.:   0.61193 |3rd Qu.:  0.3986 |3rd Qu.:  0.5704 |3rd Qu.:0.000000 |
|   |Max.   :172792 |Max.   :  2.45493 |Max.   : 22.05773 |Max.   :  9.3826 |Max.   :16.87534 |Max.   :  34.80167 |Max.   : 73.3016 |Max.   :120.5895 |Max.   :1.000000 |
> df_headf45776cdca6 <- readRDS("/private/var/folders/2c/hvxqjjcn3sq5lcshxhv8wnsw0000gp/T/RtmpsaJlZC/df_headf45776cdca6.RDS")
> knitr::kable(df_headf45776cdca6)


| Time|         V1|         V2|        V3|         V4|         V5|         V6|         V7|         V8|         V9|        V10|        V11|        V12|        V13|        V14|        V15|        V16|        V17|        V18|        V19|        V20|        V21|        V22|        V23|        V24|        V25|        V26|        V27|        V28|  y|
|----:|----------:|----------:|---------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|--:|
|    0| -1.3598071| -0.0727812| 2.5363467|  1.3781552| -0.3383208|  0.4623878|  0.2395986|  0.0986979|  0.3637870|  0.0907942| -0.5515995| -0.6178009| -0.9913898| -0.3111694|  1.4681770| -0.4704005|  0.2079712|  0.0257906|  0.4039930|  0.2514121| -0.0183068|  0.2778376| -0.1104739|  0.0669281|  0.1285394| -0.1891148|  0.1335584| -0.0210531|  0|
|    0|  1.1918571|  0.2661507| 0.1664801|  0.4481541|  0.0600176| -0.0823608| -0.0788030|  0.0851017| -0.2554251| -0.1669744|  1.6127267|  1.0652353|  0.4890950| -0.1437723|  0.6355581|  0.4639170| -0.1148047| -0.1833613| -0.1457830| -0.0690831| -0.2257752| -0.6386720|  0.1012880| -0.3398465|  0.1671704|  0.1258945| -0.0089831|  0.0147242|  0|
|    1| -1.3583541| -1.3401631| 1.7732093|  0.3797796| -0.5031981|  1.8004994|  0.7914610|  0.2476758| -1.5146543|  0.2076429|  0.6245015|  0.0660837|  0.7172927| -0.1659459|  2.3458649| -2.8900832|  1.1099694| -0.1213593| -2.2618571|  0.5249797|  0.2479982|  0.7716794|  0.9094123| -0.6892810| -0.3276418| -0.1390966| -0.0553528| -0.0597518|  0|
|    1| -0.9662717| -0.1852260| 1.7929933| -0.8632913| -0.0103089|  1.2472032|  0.2376089|  0.3774359| -1.3870241| -0.0549519| -0.2264873|  0.1782282|  0.5077569| -0.2879237| -0.6314181| -1.0596472| -0.6840928|  1.9657750| -1.2326220| -0.2080378| -0.1083005|  0.0052736| -0.1903205| -1.1755753|  0.6473760| -0.2219288|  0.0627228|  0.0614576|  0|
|    2| -1.1582331|  0.8777368| 1.5487178|  0.4030339| -0.4071934|  0.0959215|  0.5929407| -0.2705327|  0.8177393|  0.7530744| -0.8228429|  0.5381956|  1.3458516| -1.1196698|  0.1751211| -0.4514492| -0.2370332| -0.0381948|  0.8034869|  0.4085424| -0.0094307|  0.7982785| -0.1374581|  0.1412670| -0.2060096|  0.5022922|  0.2194222|  0.2151531|  0|
|    2| -0.4259659|  0.9605230| 1.1411093| -0.1682521|  0.4209869| -0.0297276|  0.4762009|  0.2603143| -0.5686714| -0.3714072|  1.3412620|  0.3598938| -0.3580907| -0.1371337|  0.5176168|  0.4017259| -0.0581328|  0.0686531| -0.0331938|  0.0849677| -0.2082535| -0.5598248| -0.0263977| -0.3714266| -0.2327938|  0.1059148|  0.2538442|  0.0810803|  0|
> knitr::kable(df_headf45776cdca6[, c(1:8, 30)])


| Time|         V1|         V2|        V3|         V4|         V5|         V6|         V7|  y|
|----:|----------:|----------:|---------:|----------:|----------:|----------:|----------:|--:|
|    0| -1.3598071| -0.0727812| 2.5363467|  1.3781552| -0.3383208|  0.4623878|  0.2395986|  0|
|    0|  1.1918571|  0.2661507| 0.1664801|  0.4481541|  0.0600176| -0.0823608| -0.0788030|  0|
|    1| -1.3583541| -1.3401631| 1.7732093|  0.3797796| -0.5031981|  1.8004994|  0.7914610|  0|
|    1| -0.9662717| -0.1852260| 1.7929933| -0.8632913| -0.0103089|  1.2472032|  0.2376089|  0|
|    2| -1.1582331|  0.8777368| 1.5487178|  0.4030339| -0.4071934|  0.0959215|  0.5929407|  0|
|    2| -0.4259659|  0.9605230| 1.1411093| -0.1682521|  0.4209869| -0.0297276|  0.4762009|  0|
> holdout_resultsf4579f2d0d9 <- readRDS("/private/var/folders/2c/hvxqjjcn3sq5lcshxhv8wnsw0000gp/T/RtmpsaJlZC/holdout_resultsf4579f2d0d9.RDS")
> knitr::kable(holdout_resultsf4579f2d0d9)


|Model                          | Area_Under_Curve| True_Positive_Rate_aka_Sensitivity| True_Negative_Rate_aka_Specificity| False_Positive_Rate_aka_Type_I_Error| False_Negative_Rate_aka_Type_II_Error| Positive_Predictive_Value_aka_Precision| Negative_Predictive_Value| F1_Score| Duration| Duration_sd|
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
> VIF_tablef45346d951b <- readRDS("/private/var/folders/2c/hvxqjjcn3sq5lcshxhv8wnsw0000gp/T/RtmpsaJlZC/VIF_tablef45346d951b.RDS")
> knitr::kable(VIF_tablef45346d951b)


|     |        x|
|:----|--------:|
|Time | 1.879717|
|V1   | 1.025906|
|V2   | 1.000211|
|V3   | 1.330979|
|V4   | 1.020827|
|V5   | 1.056305|
|V6   | 1.007464|
|V7   | 1.013490|
|V8   | 1.002566|
|V9   | 1.000141|
|V10  | 1.001762|
|V11  | 1.115321|
|V12  | 1.029065|
|V13  | 1.008164|
|V14  | 1.018333|
|V15  | 1.063262|
|V16  | 1.000266|
|V17  | 1.010099|
|V18  | 1.015374|
|V19  | 1.001578|
|V20  | 1.004863|
|V21  | 1.003762|
|V22  | 1.039010|
|V23  | 1.004916|
|V24  | 1.000492|
|V25  | 1.102121|
|V26  | 1.003223|
|V27  | 1.000050|
|V28  | 1.000166|
> 0.7732 - 0.0499 - 0.0552
[1] 0.6681
> knitr::kable(df_headf45776cdca6)


| Time|         V1|         V2|        V3|         V4|         V5|         V6|         V7|         V8|         V9|        V10|        V11|        V12|        V13|        V14|        V15|        V16|        V17|        V18|        V19|        V20|        V21|        V22|        V23|        V24|        V25|        V26|        V27|        V28|  y|
|----:|----------:|----------:|---------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|----------:|--:|
|    0| -1.3598071| -0.0727812| 2.5363467|  1.3781552| -0.3383208|  0.4623878|  0.2395986|  0.0986979|  0.3637870|  0.0907942| -0.5515995| -0.6178009| -0.9913898| -0.3111694|  1.4681770| -0.4704005|  0.2079712|  0.0257906|  0.4039930|  0.2514121| -0.0183068|  0.2778376| -0.1104739|  0.0669281|  0.1285394| -0.1891148|  0.1335584| -0.0210531|  0|
|    0|  1.1918571|  0.2661507| 0.1664801|  0.4481541|  0.0600176| -0.0823608| -0.0788030|  0.0851017| -0.2554251| -0.1669744|  1.6127267|  1.0652353|  0.4890950| -0.1437723|  0.6355581|  0.4639170| -0.1148047| -0.1833613| -0.1457830| -0.0690831| -0.2257752| -0.6386720|  0.1012880| -0.3398465|  0.1671704|  0.1258945| -0.0089831|  0.0147242|  0|
|    1| -1.3583541| -1.3401631| 1.7732093|  0.3797796| -0.5031981|  1.8004994|  0.7914610|  0.2476758| -1.5146543|  0.2076429|  0.6245015|  0.0660837|  0.7172927| -0.1659459|  2.3458649| -2.8900832|  1.1099694| -0.1213593| -2.2618571|  0.5249797|  0.2479982|  0.7716794|  0.9094123| -0.6892810| -0.3276418| -0.1390966| -0.0553528| -0.0597518|  0|
|    1| -0.9662717| -0.1852260| 1.7929933| -0.8632913| -0.0103089|  1.2472032|  0.2376089|  0.3774359| -1.3870241| -0.0549519| -0.2264873|  0.1782282|  0.5077569| -0.2879237| -0.6314181| -1.0596472| -0.6840928|  1.9657750| -1.2326220| -0.2080378| -0.1083005|  0.0052736| -0.1903205| -1.1755753|  0.6473760| -0.2219288|  0.0627228|  0.0614576|  0|
|    2| -1.1582331|  0.8777368| 1.5487178|  0.4030339| -0.4071934|  0.0959215|  0.5929407| -0.2705327|  0.8177393|  0.7530744| -0.8228429|  0.5381956|  1.3458516| -1.1196698|  0.1751211| -0.4514492| -0.2370332| -0.0381948|  0.8034869|  0.4085424| -0.0094307|  0.7982785| -0.1374581|  0.1412670| -0.2060096|  0.5022922|  0.2194222|  0.2151531|  0|
|    2| -0.4259659|  0.9605230| 1.1411093| -0.1682521|  0.4209869| -0.0297276|  0.4762009|  0.2603143| -0.5686714| -0.3714072|  1.3412620|  0.3598938| -0.3580907| -0.1371337|  0.5176168|  0.4017259| -0.0581328|  0.0686531| -0.0331938|  0.0849677| -0.2082535| -0.5598248| -0.0263977| -0.3714266| -0.2327938|  0.1059148|  0.2538442|  0.0810803|  0|
> knitr::kable(datasummaryf45c0a538b)


|   |     Time      |      V1          |      V2          |      V3         |      V4         |      V5           |      V6         |      V7         |      V8          |      V9          |     V10          |     V11         |     V12         |     V13         |     V14         |     V15         |     V16          |     V17          |     V18          |     V19          |     V20          |     V21          |     V22           |     V23          |     V24         |     V25          |     V26         |     V27           |     V28          |      y          |
|:--|:--------------|:-----------------|:-----------------|:----------------|:----------------|:------------------|:----------------|:----------------|:-----------------|:-----------------|:-----------------|:----------------|:----------------|:----------------|:----------------|:----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:-----------------|:------------------|:-----------------|:----------------|:-----------------|:----------------|:------------------|:-----------------|:----------------|
|   |Min.   :     0 |Min.   :-56.40751 |Min.   :-72.71573 |Min.   :-48.3256 |Min.   :-5.68317 |Min.   :-113.74331 |Min.   :-26.1605 |Min.   :-43.5572 |Min.   :-73.21672 |Min.   :-13.43407 |Min.   :-24.58826 |Min.   :-4.79747 |Min.   :-18.6837 |Min.   :-5.79188 |Min.   :-19.2143 |Min.   :-4.49894 |Min.   :-14.12985 |Min.   :-25.16280 |Min.   :-9.498746 |Min.   :-7.213527 |Min.   :-54.49772 |Min.   :-34.83038 |Min.   :-10.933144 |Min.   :-44.80774 |Min.   :-2.83663 |Min.   :-10.29540 |Min.   :-2.60455 |Min.   :-22.565679 |Min.   :-15.43008 |Min.   :0.000000 |
|   |1st Qu.: 54202 |1st Qu.: -0.92037 |1st Qu.: -0.59855 |1st Qu.: -0.8904 |1st Qu.:-0.84864 |1st Qu.:  -0.69160 |1st Qu.: -0.7683 |1st Qu.: -0.5541 |1st Qu.: -0.20863 |1st Qu.: -0.64310 |1st Qu.: -0.53543 |1st Qu.:-0.76249 |1st Qu.: -0.4056 |1st Qu.:-0.64854 |1st Qu.: -0.4256 |1st Qu.:-0.58288 |1st Qu.: -0.46804 |1st Qu.: -0.48375 |1st Qu.:-0.498850 |1st Qu.:-0.456299 |1st Qu.: -0.21172 |1st Qu.: -0.22839 |1st Qu.: -0.542350 |1st Qu.: -0.16185 |1st Qu.:-0.35459 |1st Qu.: -0.31715 |1st Qu.:-0.32698 |1st Qu.: -0.070839 |1st Qu.: -0.05296 |1st Qu.:0.000000 |
|   |Median : 84692 |Median :  0.01811 |Median :  0.06549 |Median :  0.1798 |Median :-0.01985 |Median :  -0.05434 |Median : -0.2742 |Median :  0.0401 |Median :  0.02236 |Median : -0.05143 |Median : -0.09292 |Median :-0.03276 |Median :  0.1400 |Median :-0.01357 |Median :  0.0506 |Median : 0.04807 |Median :  0.06641 |Median : -0.06568 |Median :-0.003636 |Median : 0.003735 |Median : -0.06248 |Median : -0.02945 |Median :  0.006782 |Median : -0.01119 |Median : 0.04098 |Median :  0.01659 |Median :-0.05214 |Median :  0.001342 |Median :  0.01124 |Median :0.000000 |
|   |Mean   : 94814 |Mean   :  0.00000 |Mean   :  0.00000 |Mean   :  0.0000 |Mean   : 0.00000 |Mean   :   0.00000 |Mean   :  0.0000 |Mean   :  0.0000 |Mean   :  0.00000 |Mean   :  0.00000 |Mean   :  0.00000 |Mean   : 0.00000 |Mean   :  0.0000 |Mean   : 0.00000 |Mean   :  0.0000 |Mean   : 0.00000 |Mean   :  0.00000 |Mean   :  0.00000 |Mean   : 0.000000 |Mean   : 0.000000 |Mean   :  0.00000 |Mean   :  0.00000 |Mean   :  0.000000 |Mean   :  0.00000 |Mean   : 0.00000 |Mean   :  0.00000 |Mean   : 0.00000 |Mean   :  0.000000 |Mean   :  0.00000 |Mean   :0.001727 |
|   |3rd Qu.:139320 |3rd Qu.:  1.31564 |3rd Qu.:  0.80372 |3rd Qu.:  1.0272 |3rd Qu.: 0.74334 |3rd Qu.:   0.61193 |3rd Qu.:  0.3986 |3rd Qu.:  0.5704 |3rd Qu.:  0.32735 |3rd Qu.:  0.59714 |3rd Qu.:  0.45392 |3rd Qu.: 0.73959 |3rd Qu.:  0.6182 |3rd Qu.: 0.66251 |3rd Qu.:  0.4931 |3rd Qu.: 0.64882 |3rd Qu.:  0.52330 |3rd Qu.:  0.39968 |3rd Qu.: 0.500807 |3rd Qu.: 0.458949 |3rd Qu.:  0.13304 |3rd Qu.:  0.18638 |3rd Qu.:  0.528554 |3rd Qu.:  0.14764 |3rd Qu.: 0.43953 |3rd Qu.:  0.35072 |3rd Qu.: 0.24095 |3rd Qu.:  0.091045 |3rd Qu.:  0.07828 |3rd Qu.:0.000000 |
|   |Max.   :172792 |Max.   :  2.45493 |Max.   : 22.05773 |Max.   :  9.3826 |Max.   :16.87534 |Max.   :  34.80167 |Max.   : 73.3016 |Max.   :120.5895 |Max.   : 20.00721 |Max.   : 15.59499 |Max.   : 23.74514 |Max.   :12.01891 |Max.   :  7.8484 |Max.   : 7.12688 |Max.   : 10.5268 |Max.   : 8.87774 |Max.   : 17.31511 |Max.   :  9.25353 |Max.   : 5.041069 |Max.   : 5.591971 |Max.   : 39.42090 |Max.   : 27.20284 |Max.   : 10.503090 |Max.   : 22.52841 |Max.   : 4.58455 |Max.   :  7.51959 |Max.   : 3.51735 |Max.   : 31.612198 |Max.   : 33.84781 |Max.   :1.000000 |
> knitr::kable(correlationtablef45a68ee68)


|     |    Time|      V1|      V2|      V3|      V4|      V5|      V6|      V7|      V8|      V9|     V10|     V11|     V12|     V13|     V14|     V15|     V16|     V17|     V18|    V19|     V20|    V21|    V22|     V23|     V24|     V25|     V26|     V27|     V28|       y|
|:----|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|------:|-------:|------:|------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
|Time |  1.0000|  0.1174| -0.0106| -0.4196| -0.1053|  0.1731| -0.0630|  0.0847| -0.0369| -0.0087|  0.0306| -0.2477|  0.1243| -0.0659| -0.0988| -0.1835|  0.0119| -0.0733|  0.0904| 0.0290| -0.0509| 0.0447| 0.1441|  0.0511| -0.0162| -0.2331| -0.0414| -0.0051| -0.0094| -0.0123|
|V1   |  0.1174|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.1013|
|V2   | -0.0106|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0913|
|V3   | -0.4196|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.1930|
|V4   | -0.1053|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.1334|
|V5   |  0.1731|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.0950|
|V6   | -0.0630|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.0436|
|V7   |  0.0847|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.1873|
|V8   | -0.0369|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0199|
|V9   | -0.0087|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.0977|
|V10  |  0.0306|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.2169|
|V11  | -0.2477|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.1549|
|V12  |  0.1243|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.2606|
|V13  | -0.0659|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.0046|
|V14  | -0.0988|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.3025|
|V15  | -0.1835|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.0042|
|V16  |  0.0119|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.1965|
|V17  | -0.0733|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.3265|
|V18  |  0.0904|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.1115|
|V19  |  0.0290|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 1.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0348|
|V20  | -0.0509|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  1.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0201|
|V21  |  0.0447|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 1.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0404|
|V22  |  0.1441|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0008|
|V23  |  0.0511|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.0027|
|V24  | -0.0162|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0000| -0.0072|
|V25  | -0.2331|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0000|  0.0033|
|V26  | -0.0414|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0000|  0.0045|
|V27  | -0.0051|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0000|  0.0176|
|V28  | -0.0094|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000| 0.0000|  0.0000| 0.0000| 0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  0.0000|  1.0000|  0.0095|
|y    | -0.0123| -0.1013|  0.0913| -0.1930|  0.1334| -0.0950| -0.0436| -0.1873|  0.0199| -0.0977| -0.2169|  0.1549| -0.2606| -0.0046| -0.3025| -0.0042| -0.1965| -0.3265| -0.1115| 0.0348|  0.0201| 0.0404| 0.0008| -0.0027| -0.0072|  0.0033|  0.0045|  0.0176|  0.0095|  1.0000|


The correlation table for the ensemble shows that none of the ensemble features were strongly correlated to the target, y:

|                               | Elastic| Flexible_Discriminant_Analysis| Generalized_Additive_Models| Generalized_Linear_Models| Gradient_Boosted| Neuralnet|      y|
|:------------------------------|-------:|------------------------------:|---------------------------:|-------------------------:|----------------:|---------:|------:|
|Elastic                        |  1.0000|                         0.7049|                      0.1621|                    0.9845|           0.0246|    0.0355| 0.6414|
|Flexible_Discriminant_Analysis |  0.7049|                         1.0000|                      0.1584|                    0.7010|           0.0242|    0.0361| 0.7718|
|Generalized_Additive_Models    |  0.1621|                         0.1584|                      1.0000|                    0.1616|           0.0276|    0.2028| 0.1626|
|Generalized_Linear_Models      |  0.9845|                         0.7010|                      0.1616|                    1.0000|           0.0247|    0.0354| 0.6381|
|Gradient_Boosted               |  0.0246|                         0.0242|                      0.0276|                    0.0247|           1.0000|    0.0088| 0.0632|
|Neuralnet                      |  0.0355|                         0.0361|                      0.2028|                    0.0354|           0.0088|    1.0000| 0.0403|
|y                              |  0.6414|                         0.7718|                      0.1626|                    0.6381|           0.0632|    0.0403| 1.0000|

We can also see how much time each model took to run, and all of them ran in very little time:

![Duration](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/big_credit_card_duration_barchart.jpg)<br>

<h2>Step 5: Strongest evidence based recommendations to fight credit card fraud based on the data set</h2>

<h2>Step 6: Conclusions</h2>

<h2>Step 7: References</h2>

something to delete.
