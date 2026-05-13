<h1>Athlete Recovery and Biometric Training Analysis with 32 models, individual and ensembles</h1>

<img width="2816" height="1536" alt="Athletes training" src="https://github.com/user-attachments/assets/388e214a-db21-49e8-bc82-3b7f56d10360" />

Athlete Recovery and Biometric Training Analysis with 32 models, individual and ensembles


Athletes training against a Chicago skyline. Image created with assistance from Google Gemini Pro.
Russ Conte

May 13, 2026

Introduction.
The Athlete Recovery and Biometric Performance Dataset was posted on kaggle.com. The original data set is:

https://www.kaggle.com/datasets/sarveshchhetri/athlete-recovery-and-biometric-performance-dataset

Many thanks to Sarvesh Chhetri who posted the dataset.

The data set author describes the data set as: “The Athlete Recovery & Biometric Performance Dataset is a comprehensive, longitudinal synthetic dataset that tracks the daily training habits, biometrics, and recovery patterns of ~300 athletes over 28 days.”

The data set includes a number of metrics that athletes can use to improve their performance and results. The data set author describes these as:

⚡ Physical Exertion
Training Type 

Duration 

Intensity 

🛌 Lifestyle Factors
Sleep Duration 

Caffeine Intake 

Stress Level 

❤️ Physiological Responses
Resting Heart Rate 

Heart Rate Variability (HRV) 

Muscle Soreness 

Energy Level 

🧠 Subjective Metrics
Mood Score
🎯 Target Variable
recovery_score (0-100)
Github repository for reproducible results
https://github.com/InfiniteCuriosity/EnsemblesData/blob/main/athlete_recovery_synthetic.csv

Load the data:
df <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/EnsemblesData/refs/heads/main/athlete_recovery_synthetic.csv', stringsAsFactors = TRUE)

This will load the data into R and make it easy for us to run the analysis.

Exploratory Data Analysis
Head of the data

head(df)


| Athlete_ID| Day| Day_of_Week| Age| Gender| Sport_Type| Training_Type| Training_Duration_Min| Training_Intensity| Sleep_Duration_Hours| Caffeine_Intake_mg| Stress_Level| Resting_Heart_Rate| HRV_ms| Mood_Score| Muscle_Soreness| Energy_Level| Recovery_Score|
|----------:|---:|-----------:|---:|------:|----------:|-------------:|---------------------:|------------------:|--------------------:|------------------:|------------:|------------------:|------:|----------:|---------------:|------------:|--------------:|
|       1000|   1|           1|  28|      2|          5|             2|                    46|                7.9|                  7.7|                270|            1|                 59|     75|        4.3|             4.8|          5.9|           51.1|
|       1000|   2|           2|  28|      2|          5|             1|                    71|                7.0|                  7.3|                258|            2|                 55|     79|        5.9|             4.0|          5.8|           63.7|
|       1000|   3|           3|  28|      2|          5|             2|                    45|                7.3|                  7.7|                214|            3|                 57|     77|        5.1|             4.0|          5.6|           71.0|
|       1000|   4|           4|  28|      2|          5|             1|                    86|                7.5|                  7.9|                228|            1|                 61|     71|        3.0|             6.2|          2.7|           37.2|
|       1000|   5|           5|  28|      2|          5|             2|                    28|                9.2|                  6.7|                  0|            1|                 64|     62|        3.3|             7.4|          1.1|           12.3|
|       1000|   6|           6|  28|      2|          5|             1|                    70|                5.5|                  8.4|                149|            2|                 58|     79|        6.2|             5.1|          3.1|           82.3|

The data has 18 columns and 7774 rows. The data has been cleaned, there are no NAs or other issues with the data. It is a tidy data set.

Types of data
<br>
'data.frame':	7774 obs. of  18 variables:<br>
 $ Athlete_ID           : int  1000 1000 1000 1000 1000 1000 1000 1000 1000 1000 ...<br>
 $ Day                  : int  1 2 3 4 5 6 7 8 9 10 ...<br>
 $ Day_of_Week          : int  1 2 3 4 5 6 7 1 2 3 ...<br>
 $ Age                  : int  28 28 28 28 28 28 28 28 28 28 ...<br>
 $ Gender               : num  2 2 2 2 2 2 2 2 2 2 ...<br>
 $ Sport_Type           : num  5 5 5 5 5 5 5 5 5 5 ...<br>
 $ Training_Type        : num  2 1 2 1 2 1 5 4 2 4 ...<br>
 $ Training_Duration_Min: int  46 71 45 86 28 70 42 63 54 78 ...<br>
 $ Training_Intensity   : num  7.9 7 7.3 7.5 9.2 5.5 2.9 7.7 9.4 8.5 ...<br>
 $ Sleep_Duration_Hours : num  7.7 7.3 7.7 7.9 6.7 8.4 7.8 7.8 8 5.9 ...<br>
 $ Caffeine_Intake_mg   : int  270 258 214 228 0 149 217 343 216 299 ...<br>
 $ Stress_Level         : num  1 2 3 1 1 2 3 1 1 1 ...<br>
 $ Resting_Heart_Rate   : int  59 55 57 61 64 58 59 60 61 67 ...<br>
 $ HRV_ms               : int  75 79 77 71 62 79 73 65 68 59 ...<br>
 $ Mood_Score           : num  4.3 5.9 5.1 3 3.3 6.2 5.2 4.3 4.9 2.7 ...<br>
 $ Muscle_Soreness      : num  4.8 4 4 6.2 7.4 5.1 3.5 7.6 7.2 8 ...<br>
 $ Energy_Level         : num  5.9 5.8 5.6 2.7 1.1 3.1 4.1 1 1.1 1 ...<br>
 $ Recovery_Score       : num  51.1 63.7 71 37.2 12.3 82.3 44.5 25.1 27.4 0 ...<br>
<br>
The str function reports the type of data for each feature. 14 of the columns are integer or numeric, and four are factors. We can see the number of values for each factor:

Gender:

|Female | Male | Non-binary|
|:---:|:---:|:---:|
|3885 | 3661 | 228 |

Sport Type:

|Combat | Endurance | Mixed | Strength | Team | Sport|
|:---:|:---:|:---:|:---:|:---:|:---:|
|852 | 1781 | 1164 | 1380| 2597| 

Training Type:

|Cardio | HIIT | Rest | Strength | Yoga |
|:---:|:---:|:---:|:---:|:---:|
|2094| 2060| 881| 2069| 670 |

Stress Level:

|High | Low | Medium |
|:---:|:---:|:---:|
|1572 | 1590 | 4612 | 

The numbers seem to be well represented. None of the values are close to 0, so we should not have any issues with rare data.

Summary of the data

|   |  Athlete_ID |     Day      | Day_of_Week  |     Age      |    Gender   |  Sport_Type  |Training_Type |Training_Duration_Min |Training_Intensity |Sleep_Duration_Hours |Caffeine_Intake_mg | Stress_Level |Resting_Heart_Rate |    HRV_ms     |  Mood_Score  |Muscle_Soreness | Energy_Level  |Recovery_Score |
|:--|:------------|:-------------|:-------------|:-------------|:------------|:-------------|:-------------|:---------------------|:------------------|:--------------------|:------------------|:-------------|:------------------|:--------------|:-------------|:---------------|:--------------|:--------------|
|   |Min.   :1000 |Min.   : 1.00 |Min.   :1.000 |Min.   :18.00 |Min.   :1.00 |Min.   :1.000 |Min.   :1.000 |Min.   : 10.00        |Min.   : 1.00      |Min.   :5.000        |Min.   :  0.0      |Min.   :1.000 |Min.   :38.00      |Min.   : 28.00 |Min.   :1.000 |Min.   : 1.000  |Min.   : 1.000 |Min.   :  0.00 |
|   |1st Qu.:1074 |1st Qu.: 7.00 |1st Qu.:2.000 |1st Qu.:23.00 |1st Qu.:1.00 |1st Qu.:2.000 |1st Qu.:1.000 |1st Qu.: 36.00        |1st Qu.: 4.90      |1st Qu.:7.000        |1st Qu.:125.0      |1st Qu.:2.000 |1st Qu.:53.00      |1st Qu.: 65.00 |1st Qu.:4.500 |1st Qu.: 3.600  |1st Qu.: 2.000 |1st Qu.: 30.02 |
|   |Median :1149 |Median :14.00 |Median :4.000 |Median :26.00 |Median :2.00 |Median :4.000 |Median :2.000 |Median : 56.00        |Median : 6.60      |Median :7.500        |Median :210.0      |Median :3.000 |Median :57.00      |Median : 75.00 |Median :5.300 |Median : 5.200  |Median : 3.200 |Median : 50.60 |
|   |Mean   :1149 |Mean   :14.48 |Mean   :4.005 |Mean   :25.84 |Mean   :1.53 |Mean   :3.397 |Mean   :2.635 |Mean   : 52.61        |Mean   : 6.18      |Mean   :7.499        |Mean   :183.4      |Mean   :2.391 |Mean   :57.06      |Mean   : 74.67 |Mean   :5.242 |Mean   : 5.116  |Mean   : 3.343 |Mean   : 50.64 |
|   |3rd Qu.:1224 |3rd Qu.:22.00 |3rd Qu.:6.000 |3rd Qu.:28.00 |3rd Qu.:2.00 |3rd Qu.:5.000 |3rd Qu.:4.000 |3rd Qu.: 69.00        |3rd Qu.: 7.90      |3rd Qu.:8.000        |3rd Qu.:265.0      |3rd Qu.:3.000 |3rd Qu.:61.00      |3rd Qu.: 84.00 |3rd Qu.:6.000 |3rd Qu.: 6.700  |3rd Qu.: 4.400 |3rd Qu.: 71.08 |
|   |Max.   :1299 |Max.   :28.00 |Max.   :7.000 |Max.   :41.00 |Max.   :3.00 |Max.   :5.000 |Max.   :5.000 |Max.   :115.00        |Max.   :10.00      |Max.   :9.500        |Max.   :400.0      |Max.   :3.000 |Max.   :79.00      |Max.   :115.00 |Max.   :9.500 |Max.   :10.000  |Max.   :10.000 |Max.   :100.00 |


The summary function gives the min, 1st Qu, Median, Mean, 3rd Qu and Max for each feature. Several are notable. The athletes range from 18 to 41 years old, include five different sport types and training types. The data also reports some athletes having 0.0 caffeine intake, a resting heart rate of 38.00 beats per minute, and recovery scores ranging from 0.00 to 100.00

Boxplots


The box plots report a number of unusual findings. The majority of box plots do not have any outliers, but six of them do, as noted by the red dots, which indicate values outside the middle 66% of the values: Age and energy level (with all the values above the inter-quartile range), sleep duration has a number of values below the inter-quartile range, and HRV_ms, mood_score and Resting_heart_rate have values above and below the inter-quartile range.

Outliers as measured by Cook’s Distance:



Outliers as measured by Cook’s distance
This data set clearly has a lot of values that are reported as outliers. This report will keep the outliers, as they are not due to any systemic or methodological errors.

Histograms of each numeric column



Histograms of each numeric column
We can see many of the values are approximately normally distributed, such as energy level, MRV_ms, mood_score, Muscle_Soreness, Recovery_Score, Resting_Heart_Rate, and Sleep_Duration_Hours. The plots for Training_Duration_Min and Training_Intensity show a slightly non-symmetrical result.

Recovery_Score (y) vs each predictor



Recovery score vs each of the predictors
The plot of Recovery_Score (on the y-axis) vs each of the predictors is very revealing. The extreme majority of the plots show the data within a clearly defined range.

Correlation of the data (report)

library(tidyverse)

── Attaching core tidyverse packages ──────────────────────── tidyverse 2.0.0 ──
✔ dplyr     1.2.1     ✔ readr     2.2.0
✔ forcats   1.0.1     ✔ stringr   1.6.0
✔ ggplot2   4.0.3     ✔ tibble    3.3.1
✔ lubridate 1.9.5     ✔ tidyr     1.3.2
✔ purrr     1.2.2     
── Conflicts ────────────────────────────────────────── tidyverse_conflicts() ──
✖ dplyr::filter() masks stats::filter()
✖ dplyr::lag()    masks stats::lag()
ℹ Use the conflicted package (<http://conflicted.r-lib.org/>) to force all conflicts to become errors
df1 <- df %>% purrr::keep(is.numeric)
head(cor(df1))

                         Athlete_ID          Day   Day_of_Week           Age
Athlete_ID             1.0000000000  0.003181412 -0.0007664656  0.0004031379
Day                    0.0031814122  1.000000000  0.2493622621 -0.0011728265
Day_of_Week           -0.0007664656  0.249362262  1.0000000000 -0.0019610525
Age                    0.0004031379 -0.001172827 -0.0019610525  1.0000000000
Training_Duration_Min  0.0114574150 -0.042690978 -0.1966859058  0.0038350557
Training_Intensity     0.0018921492 -0.252388624 -0.6034657392  0.0056685194
                      Training_Duration_Min Training_Intensity
Athlete_ID                      0.011457415        0.001892149
Day                            -0.042690978       -0.252388624
Day_of_Week                    -0.196685906       -0.603465739
Age                             0.003835056        0.005668519
Training_Duration_Min           1.000000000        0.149799320
Training_Intensity              0.149799320        1.000000000
                      Sleep_Duration_Hours Caffeine_Intake_mg
Athlete_ID                     0.023322029         0.01311460
Day                            0.035103726        -0.04486666
Day_of_Week                    0.050663473        -0.10852350
Age                           -0.002782183        -0.01505481
Training_Duration_Min         -0.023468425         0.03138027
Training_Intensity            -0.112471462         0.17425773
                      Resting_Heart_Rate      HRV_ms   Mood_Score
Athlete_ID                   -0.05322872  0.02290712  0.016159233
Day                           0.10697721 -0.10295039 -0.004357812
Day_of_Week                   0.01474716  0.07327046  0.078668108
Age                           0.13403207 -0.42652759  0.003690238
Training_Duration_Min        -0.02084737 -0.07190011 -0.030627136
Training_Intensity            0.13617432 -0.27925458 -0.178189504
                      Muscle_Soreness Energy_Level Recovery_Score
Athlete_ID               -0.006614642  0.013572073     0.02728712
Day                       0.013260126 -0.283157923    -0.14138912
Day_of_Week              -0.337284119 -0.063917091     0.05309288
Age                       0.002450187  0.000595532    -0.22833171
Training_Duration_Min     0.108552400 -0.034244362    -0.06124986
Training_Intensity        0.786741402 -0.272163755    -0.31326421
The data correlation report shows that none of the features are strongly correlated, and in particular there are no features which are strongly correlated with the target variable, Recovery_Score:

cor(df1)[, 14]

           Athlete_ID                   Day           Day_of_Week 
           0.02728712           -0.14138912            0.05309288 
                  Age Training_Duration_Min    Training_Intensity 
          -0.22833171           -0.06124986           -0.31326421 
 Sleep_Duration_Hours    Caffeine_Intake_mg    Resting_Heart_Rate 
           0.61641428           -0.19034180           -0.40623104 
               HRV_ms            Mood_Score       Muscle_Soreness 
           0.82443209            0.54573368           -0.50008110 
         Energy_Level        Recovery_Score 
           0.67280351            1.00000000 
Variable Importance bar chart
Variable importance bar chart
As the bar chart shows, Heart Rate Variability, HRV_ms, is the strongest predictor by far, with a value of 102.4021. The second strongest predictor is sleep_duration_hours, at 64.1145. The third strongest predictor is Energy_Level at 32.1483. Muscle_Soreness has a value of -14.4799.

The NumericEnsembles package was used to analyze the data.
The numeric ensembles package automatically performs all of the steps necessary for an analysis to be completed. The steps include:

• Complete the exploratatory data analysis

• Split the data into train, test and validation (in this case 60%, 20%, 20%)

• Remove features above a user specified value for VIF (I choose 5.00)

• Remove columns above a user specified value for correlation (I chose 0.99)

• Convert strings to numeric values (there are multiple ways to do this in NumericEnsembles).

• Builds 18 individual models and 14 ensembles of models:

Model	Library
1	Bagging	ipred
2	BayesGLM	arm
3	BayesRNN	brnn
4	Cubist	Cubist
5	Earth	earth
6	Elastic	glmnet
7	Generalized Additive Models (with smoothing splines)	gam
8	Gradient Boosted	gbm
9	Lasso	glmnet
10	Linear (tuned)	e1071
11	Neuralnet	nnet
12	Partial Least Squares	pls
13	Principal Components	pls
14	Ridge	glmnet
15	RPart	rpart
16	Support Vector Machines (tuned)	e1071
17	Tree	tree
18	XGBoost	xgb
19	EnsembleBagging	ipred
20	EnsembleBayesGLM	arm
21	EnsembleBayesRNN	brnn
22	EnsembleCubist	Cubist
23	EnsembleEarth	earth
24	EnsembleElastic	glmnet
25	EnsembleGradientBoosted	gbm
26	EnsembleLasso	glmnet
27	EnsembleLinear (tuned)	e1071
28	EnsembleNeuralnet	nnet
29	EnsembleRidge	glmnet
30	EnsembleRPart	rpart
31	EnsembleSVM (tuned)	e1071
32	EnsembleTrees	tree
Function call: (this will take 2-5 minutes to run)
Model evaluations
Accuracy (mean of the root means squared error across all resamples) and one standard deviation bars.



Accuracy barchart
Accuracy plot (Root Mean Squared Error) by model and resample, free scales: The accuracy on each resample for each of the 32 models. This report shows the accuracy for each of the 25 random resamples for each of the 32 models.
The y-axis is the rmse, the x-axis is the number of resample (from 1:25)
Accuracy plot (root-mean squared error on left of each plot)
The accuracy bar chart shows the root mean squared error for every resample on every model. This allows the reader to see all 32 models compared in a way that is neutral, fair and equal.

The y-axis is the individual root mean squared error, the x-axis is the resample number (in this case from 1:25).

Mean bias barchart



Bias is calculated as the error that occurs if a model is too complex and is not able to capture the true patterns in the data.1

 

Closer to 0 is better for bias. The results for EnsembleBayesGLM and EnsembleNeuralnet show virtually zero bias, that adds to their strength as excellent candidates for this data set.

Bias plot: This shows the bias of each result by model and each of the 25 resamples.
Bias by model and resample

Duration barchart
Duration (mean) barchart

This plot shows the man time (in seconds) for each of the 25 models. EnsembleEarth, EnsembleBayesGLM and EnsembleNeuralnet have three of the four fastest times on this chart, making them an excellent choice.

Accuracy table for the ten models with the lowest root mean squared error:

Model	Mean holdout RMSE	RMSE Lower 95% Conf Int	RMSE Upper 95% Conf Int	Overfitting lower 95% CI	Overfitting upper 95% CI	Bias	Mean train RMSE	Mean test RMSE	Mean validation RMSE
Ensemble Cubist	0.0455	0.0440	0.0471	2.3402	2.5090	0e+00	0.0167	0.0270	0.0488
Ensemble Earth	0.0593	0.0573	0.0614	1.0110	1.0839	-4e-04	0.0548	0.0552	0.0558
Ensemble BayesGLM	0.0641	0.0619	0.0664	0.9946	1.0663	-2e-04	0.0609	0.0613	0.0619
Ensemble Neuralnet	0.0641	0.0619	0.0664	0.9946	1.0663	-2e-04	0.0609	0.0613	0.0619
BayesRNN	0.0705	0.0680	0.0729	0.9873	1.0586	-2e-04	0.0688	0.0688	0.0691
Ensemble BayesRNN	0.1045	0.1009	0.1082	1.0039	1.0763	-6e-04	0.1023	0.1023	0.1046
Ensemble Gradient Boosted	0.1895	0.1829	0.1961	1.3070	1.4013	1e-03	0.1403	0.1899	0.1905
Ensemble Lasso	0.8059	0.7778	0.8340	0.9616	1.0310	-3e-03	0.8076	0.8053	0.8065
Ensemble Elastic	1.0647	1.0276	1.1017	0.9536	1.0224	-2e-04	1.0732	1.0720	1.0573

There are a number of surprises in the table of the top ten results:
• Nine of the top ten results are ensembles
• EnsembleCubist has the lowest RMSE, but the highest overfitting, thus it is a poor choice.
• EnsembleEarth has good scores for RMSE, but the overfitting confidence interval (CI) does not include 1.00, therefore it will overfit when used in production and is not a good choice.
• EnsembleBayesGLM and EnsembleNeuralnet have identical scores for accuracy and overfitting. Importantly, the 95% confidence interval for both includes 1.00, so either one is an excellent choice.

Overfitting is calculated by NumericEnsembles as:

 

Closer to 1.00 is better for overfitting. A value close to 1.00 implies the RMSE of the holdout data is approximately the same as the RMSE for the training data, across the resamples.

The NumericEnsembles package also graphs the overfitting values by each of the models, and all the resamples. In this case, the result is:
Overfitting plot by model and each resample

As the plot shows, the overfitting values will vary for each model. This is caused by the random resampling of the data 25 times. The x-axis on the overfitting plots is the overfitting value, the y-axis is the resample number. The solid black line is the mean of the overfitting values, the red line = 1.00 (the best possible score).

This graph allows the reader to see all the overfitting results in a way that is neutral and fair across all models and resamples.

Kolomogorov-Smirnov test barchart
Kogmogorov-Smirnov mean value barchart
The Kolomogorov-Smirnov test measures the likelihood that a given data set (the predicted values from each of the 32 modela) are likely from another distribution (the training set). The chart is sorted with the best results on the left side of the chart.

Our top choices, EnsembleNeuralnet and EnsembleBayesGLM are both above our criteria for decision (either p<0.05 or p<0.10), so both are satisfactory choices.

Diagnostic plots. Each plot shows a set of four charts for each model:
• Predicted vs Actual
• Residuals
• Histogram of Residuals
• Q-Q Plot

Ensemble Cubist summary plots:
Ensemble Cubist summary plots


Ensemble Earth summary plots

Ensemble BayesGLM summary plots

Criteria for decision from top three models:
Measure	EnsembleCubist	EnsembleEarth	EnsembleBayesGLM
Mean error (RMSE)	0.0455	0.0593	0.0641
95% CI for error (RMSE)	0.0440 - 0.0471	0.0573 - 0.0614	0.0619 - 0.0664
Mean overfitting	2.4246	1.0475	1.0305
95% CI for overfitting	2.3402 - 2.5090	1.0110 - 1.0839	0.9946 - 1.0663
Bias	0.00	-0.004	-0.002
K-S Test	Best of all 32 models	5th best of all 32 models	6th best of all 32 models
Duration (seconds)	0.2477	0.0339	0.0337
Rating	Fail, mean overfitting is unacceptable	Weak, 95% range for overfitting is >1.00 for all values.	Best overall result.
Meet our winner: EnsembleBayesGLM
Model coefficients:
EnsembleBayesGLM	x
(Intercept)	0.0574771
Bagging	-0.0256597
BayesGLM	62.5564948
BayesRNN	0.0709828
Cubist	-0.0120467
Earth	0.0169641
Elastic	-1.2504599
GAM	0.0394361
GBM	-0.0345526
Lasso	0.2518555
Linear	0.0036107
Neuralnet	-61.3768755
PCR	-0.0540149
PLS	0.0356316
Ridge	0.0243420
Rpart	0.0036107
SVM	-0.0170501
Tree	0.0036107
XGBoost	-0.0004376
Summary:
The analysis shows that it is possible to predict an athlete’s Recovery_Score (on a scale of 0 - 100) in this data set with a mean error rate of 0.0641 (95% CI: 0.0619 - 0.0664), overfitting mean of 1.0305 (95% CI: 0.9946 - 1.0663), and virtually zero bias, across 25 random resamples, using the NumericEnsembles package.


#DataScience #SportsAnalytics #MachineLearning #Biometrics #AthletePerformance #PredictiveModeling #HealthTech #Kaggle #RStats #SportsScience

Footnotes
https://www.geeksforgeeks.org/machine-learning/bias-vs-variance-in-machine-learning/↩︎
