<b><h1>How data science can help fight credit card fraud</h1></b>

<h3>Introduction</h3>
Credit card fraud is a huge problem in the retail sector, with total losses in the billions of dollars each year. This blog post will highlight how you can understand the process of fighting credit card fraud with data science.

<h4>The data set (it's enormous!)</h4>
One of the largest credit card fraud data sets was posted at [https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

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

<h4>Step 1a: Looking at the data: Boxplots</h4>
![boxplots](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/boxplots.jpg)<br>
What are boxplots, and what are they telling us about the credit card fraud data?

Boxplots provide a five number summary of each variable. The five values are:
<br>
• Minimum (or 0%) value. This is the lowest value for the specific feature.<br>
• Maximum (or 100%) value. This is the maximum value for the specific feature.<br>
• Median value. 50% of the data is above the median, and 50% is below the median.<br>
• First Quartile (Q1 or 25th percentile). The median of the lower 50% of the data set.<br>
• Third Quartile (Q3 or75th percentile). The median of the upper 50% of the data set.<br>
<br>
In addition, the Interquartile Range provides:IQR = Q3 - Q1<br><br>

The boxplots for the Credit Card Fraud data set clearly show the values for V1 through v28 have a very small interquartile range, and number of values above and below that range for virtually all features.The values for each feature vary across the data set. For example, the Boxplots show that V10 has values between approximately -24 and 24, but V24 has values between approximately -3 and 5.The boxplot for y (the target) only has values of 0 and 1, without any interquartile range.

<h4>Step 1b: Looking at the data: Histograms of the data</h4>
![histograms](https://raw.githubusercontent.com/InfiniteCuriosity/InfiniteCuriosity.github.io/refs/heads/main/_posts/images/histograms.jpg)<br>

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

The Exploratory Data Analysis of the Credit Card data provides evidence that the predictors, V1 to V30, are not strongly correlated to Class, they have a narrow distribution, and all the features except Time containt values outside the Interquartile range.

<h4>What the Exploratory Data Analysis suggests</h4><br>

The value of an Exploratory Data Analysis is that it suggests methods to accurately model the data. Given the nature of the data:<br>
• The target, Class, is logistic (either fraud or not fraud)<br>
• Fraud is indicated in 492 out of the 284,807 rows, so AUC is a much more reliable measure than Accuracy.<br>
• A range of models is suggested, as this might provide a better result than a single modeling system, such as Generalized Linear Models.<br>
• We will use seven individual models and five ensembles of models in our analysis.<br>
• We will use a combination of regular learning and deep learning models<br>
• We will use a combination of tuned models (in all possible situations) and untuned models (if it is not possible to tune the models)<br>

It may be very instructive to add charts, tables and graphs from our models to our result to help us in our decision making.

The LogisticEnsembles package hosted on CRAN can accomplish all of these requirements, and will be used for this data set.

<h4>Step 2: Building the models using the LogisticEnsembles package</h4>
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

<h4>Step 3: Model results</h4>

<h4>Step 5: Strongest evidence based recommendations to fight credit card fraud based on the data set</h4>

<h4>Step 6: Conclusions</h4>

<h4>Step 7: References</h4>

something to delete.
