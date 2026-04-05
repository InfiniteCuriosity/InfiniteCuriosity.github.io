<b><h1>How data science can help fight credit card fraud with read fraud data</h1></b>

Russ Conte<br>
Tuesday March 31, 2026<br>

<h3>Introduction</h3>
Credit card fraud is a huge problem in the retail sector, with total losses in the billions of dollars each year. This blog post will highlight how you can understand the process of fighting credit card fraud with data science.

<h4>You can get the same results with this data set. The methods, results and graphics are all fully reproducible, as described in this blog post</h4>

In this blog post we will look at:<br>
• How big is the problem of credit card fraud?<br>
• Why is this type of problem so difficult to solve?<br>
• Look at the data: box plots, histograms, head of the data, data summaries<br>
• What does the customer actually want?<br>
• Highest accuracy logistic model using the LogisticEnsembles package<br>
• Strongest predictor using reports from the LogisticEnsembles package<br>
• Strongest evidence based recommendations<br>
• Conclusion/summary<br>

<h4>How big is the problem of credit card fraud?</h4>
According to a paper from the European Central Bank on payment fraud dated 15 December, 2025, "In 2024 payment fraud rate in European Economic Area stable at around 0.002% of total value of transactions in a calendar year." (https://www.ecb.europa.eu/press/pr/date/2025/html/ecb.pr251215~e133d9d683.en.html) This amounted to a total of €4.2 billion in 2024, which is an increase from € 3.5 billion in 2023 and € 3.4 billion in 2022.<br>
<br>
The report from the European Central Bank breaks down the numbers as follows:

>"For 2024, the overall losses for credit transfers were €2.200 billion (a year-on-year increase of 16%), and for card payments with cards issued in the EU/EEA they were €1.329 billion (a year-on-year increase of 29%). For credit transfers, payment service users bore approximately 85% of total fraud losses in 2024, mainly as a result of scams that tricked users into initiating fraudulent transactions."

<h4>The data set</h4>
One of the largest credit card fraud data sets on kaggle.com was posted at https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud.

The data:<br>
• Reports transactions "made by credit cards in September 2013 by European cardholders."<br>
• Contains 492 frauds out of 284,807 transactions (~ 0.172% of the total)<br>
• Has been anonymized, but the label (fraud or genuine) and dollar amounts are accurate.<br>
• The data set contains 284,807 rows, and 31 variables. It is 66.3 MB in size.<br>
• The data is transformed by Principal Components Analysis, due to confidentiality of the original data.<br>

<h4>Why is this type of problem so difficult to solve?</h4>
Most examples of data have large results that are easy to see. However, fraud data typically only shows up in a small fraction of 1% of the transactions. In our case it's less than 2/10 of 1% of the transactions. We are looking for the proverbial needle in the haystack, that's part of why it is so difficult to solve.<br>

In addition, Dal Pozzolo and colleagues cited additional causes that make this problem difficult to solve:<br>

>Detecting frauds in credit card transactions is perhaps one of the best testbeds for computational intelligence algorithms. In fact, this problem involves a number of relevant challenges, namely: concept drift (customers' habits evolve and fraudsters change their strategies over time), class imbalance (genuine transactions far outnumber frauds), and verification latency (only a small set of transactions are timely checked by investigators). However, the vast majority of learning algorithms that have been proposed for fraud detection rely on assumptions that hardly hold in a real-world fraud-detection system (FDS).<br>

Source: https://www.researchgate.net/publication/319867396_Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy

<h2>Step 1: Look at the data: box plots, histograms, head of the data, data summaries</h2>
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

<h2>What does the customer actually want?</h2>
Given our data, our customer wants to predict credit card fraud. They also want results that are fully reproducible.

<h2>Step 2: Building the models using the LogisticEnsembles package, as hosted on CRAN</h2>

LogisticEnsembles will do the following:<br>
• Automatically split the data into train/test/validation sets<br>
• Automatically fit each of 7 logistic models and 5 ensembles of logistic models to the training data<br>
• Automatically resample as many times as requested (two times for this example)<br>
• Automatically make predictions and check accuracy on the holdout data (test and validation)<br>
• Automatically make summary graphics for each of the measures<br>
• Automatically make a summary table for the importance of each variable in the data set<br>
• There are no API calls, no use of any coding assistants, no use of any AI systems.<br>
• Neither the user's data nor activity is shared, stored, or tracked in any way. They do not help produce more accurate models.<br>

<h2>How LogisticEnsembles builds a team of rival models requiring only one line of code from the user</h2>
First LogisticEnsembles will automatically build a set of logistic models. The models used by the LogisticEnsembles package are a team of rivals. Some of the models are individual, others are ensembles, some are regular learning, others are deep learning, some are tuned models, others are not tuned models.<br>

Logistic model summaries:<br>
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

<h4>How the LogisticEnsembles package makes this much faster and easier to solve, using only one line of code, while maintaining a very high level of accuracy on the holdout data. Here is the one line of code (plus a couple of lines to time the analysis and check for errors):</h4>

```
library(LogisticEnsembles)

start_time <- Sys.time()
Logistic(data = read.csv('/Users/russellconte/creditcard.csv'),
         colnum = 31,
         numresamples = 25,
         positive_rate = 0.001727486,
         remove_VIF_greater_than <- 5.00,
         save_all_trained_models = "Y",
         save_all_plots = "Y",
         set_seed = "N",
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

Comments on LogisticEnsembles applied to the Credit Card Fraud data set:

The data is stored on my local computer, since it is 150 MB in size.<br>
The models are resampled 25 times, and the mean of those results will be provided.<br>
All the plots and models are saved to be used in writing reports and summaries.<br>
Both data and ensemble correlations > 0.99 are removed, thus removing the possibility of perfectly correlated features in the original data and the ensembles.<br>
The process is timed.<br>

<h4>Everything ran in 3.016171 minutes without any errors, warnings, or issues. The majority of the time was spent saving the image files and trained models. If the models and images are not saved, everything completed in 1.776362 mins, a substantially shorter run time.</h4>

<h2>Step 3: Highest Accuracy results on the holdout data, resampled 25 times, sorted (decreasing) by Area Under The Curve per model</h2>

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

<h4>Step 4: Strongest Predictors: Which variables are the strongest predictors, and how strong are they?</h4>

We begin by looking at a table of the importance of the predictors. Several points stand out:<br>

• Ten of the 12 largest predictors have a negative influence on the result<br>
• The ten most important variables account for 77.32% of the total, and more than 66.81% of the value is negative.<br>
• It is strongly recommended to look at variables V17, V14, V12, V10, V16, V7, and V3, as they are all negative, and constitute more than 63% of the total.<br>

![Variable importance table](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Big_Credit_Card_Fraud_Variable_Importance_Report.jpg)<br>

![Variable Importance Barchart](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Big_credit_variable_importance_barchart.jpg)<br>

Neither the original data nor the ensemble had strongly correlated predictors. First the correlation table of the original data:

<br>The correlation table for the ensemble:

|                               | Elastic| Flexible Discriminant Analysis| Generalized Linear Models| Gradient Boosted| Neuralnet|  Ridge|      y|
|:------------------------------|-------:|------------------------------:|-------------------------:|----------------:|---------:|------:|------:|
|Elastic                        |  1.0000|                         0.8668|                    0.1543|           0.7578|    0.0615| 0.9451| 0.7563|
|Flexible_Discriminant_Analysis |  0.8668|                         1.0000|                    0.1700|           0.8682|    0.0678| 0.8579| 0.8257|
|Generalized_Linear_Models      |  0.1543|                         0.1700|                    1.0000|           0.1597|    0.3480| 0.1477| 0.1624|
|Gradient_Boosted               |  0.7578|                         0.8682|                    0.1597|           1.0000|    0.0679| 0.7654| 0.7195|
|Neuralnet                      |  0.0615|                         0.0678|                    0.3480|           0.0679|    1.0000| 0.0589| 0.0690|
|Ridge                          |  0.9451|                         0.8579|                    0.1477|           0.7654|    0.0589| 1.0000| 0.7302|
|y                              |  0.7563|                         0.8257|                    0.1624|           0.7195|    0.0690| 0.7302| 1.0000|

We can also see how much time each model took to run (measured in seconds). All of the models ran in less than five seconds, half of them ran in less than one second:

![Duration](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/Big_credit_card_fraud_duration_barchart.jpg)<br>

<h2>Step 5: Strongest evidence based recommendations to fight credit card fraud based on the credit card fraud data set</h2>

First principles to solve the problem:<br>
• The Exploratory Data Analysis suggested the target is logistic<br>
• The LogisticEnsembles package was used to do the entire analysis<br>
• The Exploratory Data Analysis showed the data have very narrow interquartile ranges<br>
• The Correlation tables show that neithe the original data nor the ensemble are strongly correlated with the target<br>
• Four models had 100% predictive accuracy on the holdout data: XGBoost, Ensemble C50, Ensemble Elastic and Ensemble XGBoost<br>
• The same four models had excellent scores on Sensitivity, Specificity, Type I Error, Type II Error, Precision, Negative Predictive Value and F1 Score<br>
• An analysis of the strongest predictors showed that eight of the ten strongest predictors are negative.<br>
• Given these results, it is recommended that LogisticEnsembles be used with similar data sets about fraud.<br>

<h2>Step 6: Conclusions</h2>

The LogisticEnsembles package was able to complete the entire analysis in less than five minutes, providing results on the holdout data which meet the customer's requirements for predicting fraud in this data set with very high accuracy.

#Rstats #DataScience #XGBoost #Fraud #Finance #FinancialFraud #CreditCard #Crime #FinancialCrime #FightingCrime #CrimeFighter #Dataviz #ggplot2 #tidyverse
