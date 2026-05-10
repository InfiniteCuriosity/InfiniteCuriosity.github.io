May 10, 2026

Russ Conte, written with assistance by Google Gemini Pro

Are you interested in keeping your health insurance premiums down? So is everyone else, but most people can do much better to get their insurance premiums under control.

There is a famous data set of insurance rates. It's been used for decades (literally), and many people have created predictive models from the data. I also created a set of predictive models, but mine used an extremely diverse set of models, including individual models and ensembles of models, regular learning and deep learning, tuned and not-tuned models. The best resuls here on the hold out data beat the best previous results by a mile.

What actually works to keep your insurance rates low, based on the results from the models? A few things work very well.

1. The Smoke Signal: Smoking Status (Don't do it, or quit if you are a current smoker)

By a landslide, smoking is the most aggressive predictor of high medical charges. I was shocked how much smoking increased the cost of health insurance. The variable importance metrics place it at the absolute top of the hierarchy. When looking at the extremes, the "Top 5%" of spenders are exclusively smokers, while the "Bottom 5%" are virtually all non-smokers. It’s not just a health risk; it’s a financial one.

[Insert: top 5 percent vs bottom 5 percent charges by smoker.jpeg]

2. The Heavyweight Contender: BMI

Body Mass Index (BMI) holds the silver medal for predictability. Our correlation matrix and variable importance charts show a significant "tipping point" where BMI shifts from a mild factor to a primary driver of cost. In the visualization below, you can see the clear density of the highest-cost claimants hovering in the higher BMI brackets.

[Insert: Top 5% vs bottom 5% charges by BMI.jpg]

3. The Relentless March: Age

While we can control our habits, we can't control the calendar. Age remains the third most influential variable. The data shows a steady, linear climb in charges as patients move from their 20s into their 60s. Interestingly, even the "healthiest" seniors often face higher base charges than the "least healthy" young adults, simply due to the biological baseline of aging.

[Insert: top 5% vs bottom 5% charges by age.jpg]

The Technical Verdict

To reach these conclusions, I utilized a variety of models, comparing everything from BayesGLM to Ensemble methods. As shown in the Variable Importance Report, the gap between our top three predictors and the rest of the field (like region or number of children) is substantial.

[Insert: variable_importance_barchart.jpeg]

The Correlation Matrix further validates these relationships, showing how these independent variables intersect to compound total charges.

[Insert: Data Correlation.jpeg]

The Bottom Line

Data science confirms what doctors have said for years, but with the added weight of statistical significance: Lifestyle choices (Smoking and BMI) are the primary levers of insurance costs, often outweighing the inevitable impact of age.

#DataScience #MachineLearning #InsuranceAnalytics #PredictiveModeling #Statistics
