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

<h4>Step 1: Looking at the data</h4>
![boxplots](/images/boxplots.jpg)
What are boxplots, and what are they telling us about the credit card fraud data?

Boxplots provide a five number summary of each variable. The five values are:
<br>
• Minimum (or 0%) value. This is the lowest value for the specific feature.<br>
• Maximum (or 100%) value. This is the maximum value for the specific feature.<br>
• Median value. 50% of the data is above the median, and 50% is below the median.<br>
• First Quartile (Q1 or 25th percentile). The median of the lower 50% of the data set.<br>
• Third Quartile (Q3 or75th percentile). The median of the upper 50% of the data set.<br>
<br>
In addition, the Interquartile Range provides:IQR = Q3 - Q1

The boxplots for the Credit Card Fraud data set clearly show the values for V1 through v28 have a very small interquartile range, and number of values above and below that range for virtually all features.The values for each feature vary across the data set. For example, the Boxplots show that V10 has values between approximately -24 and 24, but V24 has values between approximately -3 and 5.The boxplot for y (the target) only has values of 0 and 1, without any interquartile range.

