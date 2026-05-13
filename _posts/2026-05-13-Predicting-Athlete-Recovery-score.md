<h1>Athlete Recovery and Biometric Training Analysis with 32 models, individual and ensembles</h1>

<img alt="Athletes training" src="https://github.com/user-attachments/assets/388e214a-db21-49e8-bc82-3b7f56d10360" />

Athletes training against a Chicago skyline. Image created with assistance from Google Gemini Pro.<br><br>
Russ Conte<br>
<br>
May 13, 2026<br>
<br>
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

Github repository for reproducible results:<br>
https://github.com/InfiniteCuriosity/EnsemblesData/blob/main/athlete_recovery_synthetic.csv

Load the data:

```
df <- read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/EnsemblesData/refs/heads/main/athlete_recovery_synthetic.csv', stringsAsFactors = TRUE)
```

This will load the data into R and make it easy for us to run the analysis using NumericEnsembles.

<h2>Exploratory Data Analysis</h2>

<h4>Head of the data</h4>

```
head(df)
```

These are the top six rows of the data set. All the columns (features) can be observed. The target variable is the last column, Recovery_Score.

| Athlete_ID| Day| Day_of_Week| Age| Gender| Sport_Type| Training_Type| Training_Duration_Min| Training_Intensity| Sleep_Duration_Hours| Caffeine_Intake_mg| Stress_Level| Resting_Heart_Rate| HRV_ms| Mood_Score| Muscle_Soreness| Energy_Level| Recovery_Score|
|----------:|---:|-----------:|---:|------:|----------:|-------------:|---------------------:|------------------:|--------------------:|------------------:|------------:|------------------:|------:|----------:|---------------:|------------:|--------------:|
|       1000|   1|           1|  28|      2|          5|             2|                    46|                7.9|                  7.7|                270|            1|                 59|     75|        4.3|             4.8|          5.9|           51.1|
|       1000|   2|           2|  28|      2|          5|             1|                    71|                7.0|                  7.3|                258|            2|                 55|     79|        5.9|             4.0|          5.8|           63.7|
|       1000|   3|           3|  28|      2|          5|             2|                    45|                7.3|                  7.7|                214|            3|                 57|     77|        5.1|             4.0|          5.6|           71.0|
|       1000|   4|           4|  28|      2|          5|             1|                    86|                7.5|                  7.9|                228|            1|                 61|     71|        3.0|             6.2|          2.7|           37.2|
|       1000|   5|           5|  28|      2|          5|             2|                    28|                9.2|                  6.7|                  0|            1|                 64|     62|        3.3|             7.4|          1.1|           12.3|
|       1000|   6|           6|  28|      2|          5|             1|                    70|                5.5|                  8.4|                149|            2|                 58|     79|        6.2|             5.1|          3.1|           82.3|

<br>
The data has 18 columns and 7774 rows. The data has been cleaned, there are no NAs or other issues with the data. It is a tidy data set.<br>
<h4>Types of data</h4>

```
str(df)
```
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
The str function reports the type of data for each feature. 14 of the columns are integer or numeric, and four are factors. We can see the number of values for each factor:<br>
<br>
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

<h4>Summary of the data</h4>

|   |  Athlete_ID |     Day      | Day_of_Week  |     Age      |    Gender   |  Sport_Type  |Training_Type |Training_Duration_Min |Training_Intensity |Sleep_Duration_Hours |Caffeine_Intake_mg | Stress_Level |Resting_Heart_Rate |    HRV_ms     |  Mood_Score  |Muscle_Soreness | Energy_Level  |Recovery_Score |
|:--|:------------|:-------------|:-------------|:-------------|:------------|:-------------|:-------------|:---------------------|:------------------|:--------------------|:------------------|:-------------|:------------------|:--------------|:-------------|:---------------|:--------------|:--------------|
|Max   |1000 | 1.00 |1.000 |18.00 |1.00 |1.000 |1.000 | 10.00        | 1.00      |5.000        |  0.0      |1.000 |38.00      | 28.00 |1.000 | 1.000  | 1.000 |  0.00 |
|1st Qu   |1074 | 7.00 |2.000 |23.00 |1.00 |2.000 |1.000 | 36.00        | 4.90      |7.000        |125.0      |2.000 |53.00      | 65.00 |4.500 | 3.600  | 2.000 | 30.02 |
|Median   |1149 |14.00 |4.000 |26.00 |2.00 |4.000 |2.000 | 56.00        | 6.60      |7.500        |210.0      |3.000 |57.00      | 75.00 |5.300 | 5.200  | 3.200 | 50.60 |
|Mean   |1149 |14.48 |4.005 |25.84 |1.53 |3.397 |2.635 | 52.61        | 6.18      |7.499        |183.4      |2.391 |57.06      | 74.67 |5.242 | 5.116  | 3.343 | 50.64 |
|3rd Qu   |1224 |22.00 |6.000 |28.00 |2.00 |5.000 |4.000 | 69.00        | 7.90      |8.000        |265.0      |3.000 |61.00      | 84.00 |6.000 | 6.700  | 4.400 | 71.08 |
|Max   |1299 |28.00 |7.000 |41.00 |3.00 |5.000 |5.000 |115.00        |10.00      |9.500        |400.0      |3.000 |79.00      |115.00 |9.500 |10.000  |10.000 |100.00 |
<br>
The summary function gives the min, 1st Qu, Median, Mean, 3rd Qu and Max for each feature. Several are notable. The athletes range from 18 to 41 years old, include five different sport types and training types. The data also reports some athletes having 0.0 caffeine intake, a resting heart rate of 38.00 beats per minute, and recovery scores ranging from 0.00 to 100.00
<br>
<h4>Boxplots</h4>
<img alt="boxplots" src="https://github.com/user-attachments/assets/362aaecd-33aa-4c9b-aee9-4ffa30aaebde" />
<br>
The box plots report a number of unusual findings. The majority of box plots do not have any outliers, but six of them do, as noted by the red dots, which indicate values outside the middle 66% of the values: Age and energy level (with all the values above the inter-quartile range), sleep duration has a number of values below the inter-quartile range, and HRV_ms, mood_score and Resting_heart_rate have values above and below the inter-quartile range.
<br>
<h4>Outliers as measured by Cook’s Distance:</h4>
<br>
<img alt="cooks_distance_plot" src="https://github.com/user-attachments/assets/b7178b4a-9baf-4ada-a502-4f93a5ad272e" />
<br>
This data set clearly has a lot of values that are reported as outliers. This report will keep the outliers, as they are not due to any systemic or methodological errors.
<br>
<img alt="Histograms_of_the_numeric_columns" src="https://github.com/user-attachments/assets/182b1417-32bc-4627-bf1a-4a55510729c4" />
<br>
<h4>Histograms of each numeric column</h4>
We can see many of the values are approximately normally distributed, such as energy level, MRV_ms, mood_score, Muscle_Soreness, Recovery_Score, Resting_Heart_Rate, and Sleep_Duration_Hours. The plots for Training_Duration_Min and Training_Intensity show a slightly non-symmetrical result.

<h4>Recovery_Score (y) vs each predictor</h4>

<img  alt="predictor_vs_target" src="https://github.com/user-attachments/assets/e725744c-b1da-4d38-920c-b6cca864408c" />

The plot of Recovery_Score (on the y-axis) vs each of the predictors is very revealing. The extreme majority of the plots show the data within a clearly defined range.

Correlation of the data

|                      | Athlete_ID|        Day| Day_of_Week|        Age|     Gender| Sport_Type| Training_Type| Training_Duration_Min| Training_Intensity| Sleep_Duration_Hours| Caffeine_Intake_mg| Stress_Level| Resting_Heart_Rate|     HRV_ms| Mood_Score| Muscle_Soreness| Energy_Level|          y|
|:---------------------|----------:|----------:|-----------:|----------:|----------:|----------:|-------------:|---------------------:|------------------:|--------------------:|------------------:|------------:|------------------:|----------:|----------:|---------------:|------------:|----------:|
|Athlete_ID            |  1.0000000|  0.0031814|  -0.0007665|  0.0004031|  0.0400917| -0.0502107|     0.0289372|             0.0114574|          0.0018921|            0.0233220|          0.0131146|    0.0138491|         -0.0532287|  0.0229071|  0.0161592|      -0.0066146|    0.0135721|  0.0272871|
|Day                   |  0.0031814|  1.0000000|   0.2493623| -0.0011728| -0.0001021|  0.0057786|     0.0565785|            -0.0426910|         -0.2523886|            0.0351037|         -0.0448667|    0.0022976|          0.1069772| -0.1029504| -0.0043578|       0.0132601|   -0.2831579| -0.1413891|
|Day_of_Week           | -0.0007665|  0.2493623|   1.0000000| -0.0019611|  0.0024599|  0.0009784|     0.2096680|            -0.1966859|         -0.6034657|            0.0506635|         -0.1085235|    0.0095948|          0.0147472|  0.0732705|  0.0786681|      -0.3372841|   -0.0639171|  0.0530929|
|Age                   |  0.0004031| -0.0011728|  -0.0019611|  1.0000000| -0.0499164|  0.0307213|    -0.0109182|             0.0038351|          0.0056685|           -0.0027822|         -0.0150548|   -0.0179003|          0.1340321| -0.4265276|  0.0036902|       0.0024502|    0.0005955| -0.2283317|
|Gender                |  0.0400917| -0.0001021|   0.0024599| -0.0499164|  1.0000000|  0.0443549|     0.0356624|            -0.0021594|         -0.0012669|           -0.0010199|          0.0128268|    0.0104303|         -0.0085718|  0.2065288| -0.0097076|       0.0014662|   -0.0013178|  0.1135937|
|Sport_Type            | -0.0502107|  0.0057786|   0.0009784|  0.0307213|  0.0443549|  1.0000000|     0.2158523|            -0.0004826|          0.0025764|            0.0024164|         -0.0082671|    0.0057598|          0.3870115|  0.0126879| -0.0025941|       0.0055582|    0.0024988|  0.0082708|
|Training_Type         |  0.0289372|  0.0565785|   0.2096680| -0.0109182|  0.0356624|  0.2158523|     1.0000000|            -0.0409622|         -0.2101101|            0.0102851|         -0.0167047|   -0.0133009|          0.1476617|  0.0829738|  0.0213359|      -0.1484322|    0.0178665|  0.0589932|
|Training_Duration_Min |  0.0114574| -0.0426910|  -0.1966859|  0.0038351| -0.0021594| -0.0004826|    -0.0409622|             1.0000000|          0.1497993|           -0.0234684|          0.0313803|    0.0105616|         -0.0208474| -0.0719001| -0.0306271|       0.1085524|   -0.0342444| -0.0612499|
|Training_Intensity    |  0.0018921| -0.2523886|  -0.6034657|  0.0056685| -0.0012669|  0.0025764|    -0.2101101|             0.1497993|          1.0000000|           -0.1124715|          0.1742577|   -0.0509359|          0.1361743| -0.2792546| -0.1781895|       0.7867414|   -0.2721638| -0.3132642|
|Sleep_Duration_Hours  |  0.0233220|  0.0351037|   0.0506635| -0.0027822| -0.0010199|  0.0024164|     0.0102851|            -0.0234684|         -0.1124715|            1.0000000|         -0.1680287|    0.1330920|         -0.1939376|  0.3181762|  0.4920255|      -0.2170950|    0.3810282|  0.6164143|
|Caffeine_Intake_mg    |  0.0131146| -0.0448667|  -0.1085235| -0.0150548|  0.0128268| -0.0082671|    -0.0167047|             0.0313803|          0.1742577|           -0.1680287|          1.0000000|   -0.0690443|          0.0557858| -0.1225043| -0.1641974|       0.1684516|   -0.1577895| -0.1903418|
|Stress_Level          |  0.0138491|  0.0022976|   0.0095948| -0.0179003|  0.0104303|  0.0057598|    -0.0133009|             0.0105616|         -0.0509359|            0.1330920|         -0.0690443|    1.0000000|         -0.0919971|  0.1536036|  0.2694326|      -0.0674126|    0.1425364|  0.2166632|
|Resting_Heart_Rate    | -0.0532287|  0.1069772|   0.0147472|  0.1340321| -0.0085718|  0.3870115|     0.1476617|            -0.0208474|          0.1361743|           -0.1939376|          0.0557858|   -0.0919971|          1.0000000| -0.3908204| -0.2170930|       0.2448820|   -0.3274638| -0.4062310|
|HRV_ms                |  0.0229071| -0.1029504|   0.0732705| -0.4265276|  0.2065288|  0.0126879|     0.0829738|            -0.0719001|         -0.2792546|            0.3181762|         -0.1225043|    0.1536036|         -0.3908204|  1.0000000|  0.3474121|      -0.3837201|    0.4624718|  0.8244321|
|Mood_Score            |  0.0161592| -0.0043578|   0.0786681|  0.0036902| -0.0097076| -0.0025941|     0.0213359|            -0.0306271|         -0.1781895|            0.4920255|         -0.1641974|    0.2694326|         -0.2170930|  0.3474121|  1.0000000|      -0.2630474|    0.4607392|  0.5457337|
|Muscle_Soreness       | -0.0066146|  0.0132601|  -0.3372841|  0.0024502|  0.0014662|  0.0055582|    -0.1484322|             0.1085524|          0.7867414|           -0.2170950|          0.1684516|   -0.0674126|          0.2448820| -0.3837201| -0.2630474|       1.0000000|   -0.5268779| -0.5000811|
|Energy_Level          |  0.0135721| -0.2831579|  -0.0639171|  0.0005955| -0.0013178|  0.0024988|     0.0178665|            -0.0342444|         -0.2721638|            0.3810282|         -0.1577895|    0.1425364|         -0.3274638|  0.4624718|  0.4607392|      -0.5268779|    1.0000000|  0.6728035|
|y                     |  0.0272871| -0.1413891|   0.0530929| -0.2283317|  0.1135937|  0.0082708|     0.0589932|            -0.0612499|         -0.3132642|            0.6164143|         -0.1903418|    0.2166632|         -0.4062310|  0.8244321|  0.5457337|      -0.5000811|    0.6728035|  1.0000000|

The data correlation report shows that none of the features are strongly correlated, and in particular there are no features which are strongly correlated with the target variable, Recovery_Score:

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
<br>

<img alt="Variable_importance_barchart" src="https://github.com/user-attachments/assets/11cc4ebe-d584-4dec-b908-592f2a6677e8" />

As the bar chart shows, Heart Rate Variability, HRV_ms, is the strongest predictor by far, with a value of 102.4021. The second strongest predictor is sleep_duration_hours, at 64.1145. The third strongest predictor is Energy_Level at 32.1483. Muscle_Soreness has a value of -14.4799.<br>

<h4>The NumericEnsembles package was used to analyze the data.</h4><br>
The numeric ensembles package automatically performs all of the steps necessary for an analysis to be completed. The steps include:<br>
• Complete the exploratatory data analysis<br>
• Split the data into train, test and validation (in this case 60%, 20%, 20%)<br>
• Remove features above a user specified value for VIF (I choose 5.00)<br>
• Remove columns above a user specified value for correlation (I chose 0.99)<br>
• Convert strings to numeric values (there are multiple ways to do this in NumericEnsembles).<br>
• Builds 18 individual models and 14 ensembles of models:<br>
<br>

<h4>List of models and source packages</h4>

|Number| Model | Package source|
|:---:|:---:|:---:|
|1	|Bagging	|ipred |
|2 |	BayesGLM	|arm|
|3	|BayesRNN	|brnn|
|4	|Cubist	|Cubist|
|5	|Earth	|earth|
|6	|Elastic|	glmnet|
|7	|Generalized Additive Models (with smoothing splines)|	gam|
|8	|Gradient Boosted	|gbm|
|9	| Lasso	| glmnet |
|10	|Linear (tuned)	|e1071|
|11 |	Neuralnet |	nnet|
|12	|Partial Least Squares |	pls |
|13 |	Principal Components |	pls|
|14 |	Ridge |	glmnet|
|15	|RPart|	rpart|
|16	|Support Vector Machines (tuned)|	e1071|
|17	|Tree|	tree|
|18	|XGBoost	|xgb|
|19	|EnsembleBagging	|ipred|
|20|	EnsembleBayesGLM|	arm|
|21	|EnsembleBayesRNN|	brnn|
|22	|EnsembleCubist	|Cubist|
|23	|EnsembleEarth|	earth|
|24	|EnsembleElastic|	glmnet|
|25	|EnsembleGradientBoosted	|gbm|
|26	|EnsembleLasso	|glmnet|
|27	|EnsembleLinear (tuned)	|e1071|
|28	|EnsembleNeuralnet	|nnet|
|29	|EnsembleRidge	|glmnet|
|30	|EnsembleRPart	|rpart|
|31	|EnsembleSVM (tuned)	|e1071|
|32	|EnsembleTrees	|tree|


Function call: (this will take 2-5 minutes to run)

```
#install.packages(NumericEnsembles)

start_time <- Sys.time()

library(NumericEnsembles)
Numeric(data = read.csv('https://raw.githubusercontent.com/InfiniteCuriosity/EnsemblesData/refs/heads/main/athlete_recovery_synthetic.csv', stringsAsFactors = TRUE),
  colnum = 18,
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

<h2>Model evaluations</h2>

<h4>Accuracy (mean of the root means squared error across all resamples) and one standard deviation bars.</h4>

<img alt="accuracy_barchart" src="https://github.com/user-attachments/assets/b5929035-64f8-4302-b86c-e4428346a532" />

Accuracy plot (Root Mean Squared Error) by model and resample, free scales: The accuracy on each resample for each of the 32 models. This report shows the accuracy for each of the 25 random resamples for each of the 32 models.
The y-axis is the rmse, the x-axis is the number of resample (from 1:25)

<img alt="accuracy_plot_free_scales" src="https://github.com/user-attachments/assets/38f8d54b-45ec-40c6-b344-86eccffd512b" />

The accuracy bar chart shows the root mean squared error for every resample on every model. This allows the reader to see all 32 models compared in a way that is neutral, fair and equal.

The y-axis is the individual root mean squared error, the x-axis is the resample number (in this case from 1:25).

<img alt="bias_barchart" src="https://github.com/user-attachments/assets/992b8254-3967-4f51-bf36-224b18569cfc" />

Bias is calculated as the error that occurs if a model is too complex and is not able to capture the true patterns in the data.1
Closer to 0 is better for bias. The results for EnsembleBayesGLM and EnsembleNeuralnet show virtually zero bias, that adds to their strength as excellent candidates for this data set.

<img alt="bias_plot" src="https://github.com/user-attachments/assets/b97ef917-81e1-4ce0-927e-97a759a11f07" />

Bias by model and resample

<img alt="duration_barchart" src="https://github.com/user-attachments/assets/292ccaf0-b4b8-484b-b0dd-6b18c0e8fbcb" />

Duration (mean) barchart<br>

This plot shows the man time (in seconds) for each of the 25 models. EnsembleEarth, EnsembleBayesGLM and EnsembleNeuralnet have three of the four fastest times on this chart, making them an excellent choice.

Accuracy table for the ten models with the lowest root mean squared error:

| Model | Mean holdout RMSE | RMSE Lower 95% Conf Int | RMSE Upper 95% Conf Int | Overfitting lower 95% CI | Overfitting upper 95% CI | Bias | Mean train RMSE | Mean test RMSE | Mean validation RMSE |
    |:-------|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|-------:|
    | Ensemble Cubist | 0.0455 | 0.0440 | 0.0471 | 2.3402 | 2.5090 | 0e+00 | 0.0167 | 0.0270 | 0.0488 |
    | Ensemble Earth | 0.0593 | 0.0573 | 0.0614 | 1.0110 | 1.0839 | -4e-04 | 0.0548 | 0.0552 | 0.0558 |
    | Ensemble BayesGLM | 0.0641 | 0.0619 | 0.0664 | 0.9946 | 1.0663 | -2e-04 | 0.0609 | 0.0613 | 0.0619 |
    | Ensemble Neuralnet | 0.0641 | 0.0619 | 0.0664 | 0.9946 | 1.0663 | -2e-04 | 0.0609 | 0.0613 | 0.0619 |
    | BayesRNN | 0.0705 | 0.0680 | 0.0729 | 0.9873 | 1.0586 | -2e-04 | 0.0688 | 0.0688 | 0.0691 |
    | Ensemble BayesRNN | 0.1045 | 0.1009 | 0.1082 | 1.0039 | 1.0763 | -6e-04 | 0.1023 | 0.1023 | 0.1046 |
    | Ensemble Gradient Boosted | 0.1895 | 0.1829 | 0.1961 | 1.3070 | 1.4013 | 1e-03 | 0.1403 | 0.1899 | 0.1905 |
    | Ensemble Lasso | 0.8059 | 0.7778 | 0.8340 | 0.9616 | 1.0310 | -3e-03 | 0.8076 | 0.8053 | 0.8065 |
    | Ensemble Elastic | 1.0647 | 1.0276 | 1.1017 | 0.9536 | 1.0224 | -2e-04 | 1.0732 | 1.0720 | 1.0573 |

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
