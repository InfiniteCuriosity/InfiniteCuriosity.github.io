May 10, 2026

<h1>AI assisted predicting health insurance costs, and an interactive dashboard!</h1>

<img width="1024" height="559" alt="Insurance_data_graphic" src="https://github.com/user-attachments/assets/e860aba2-e4a0-460e-ab9f-24f151ad8615" />

<i>Image created with assistance from Google Gemini Pro</i>

Russ Conte, written with assistance by Google Gemini Pro

Are you interested in keeping your health insurance premiums down? So is everyone else, but most people can do much better to get their insurance premiums under control.

There is a famous data set of insurance rates. It's been used for decades (literally), and many people have created predictive models from the data. I also created a set of predictive models, but mine used an extremely diverse set of models, including individual models and ensembles of models, regular learning and deep learning, tuned and not-tuned models. The best resuls here on the hold out data beat the best previous results by a mile.

What actually works to keep your insurance rates low, based on the results from the models? A few things work very well.

1. The Smoke Signal: Smoking Status (Don't do it, or quit if you are a current smoker)

By a landslide, smoking is the most aggressive predictor of high medical charges. I was shocked how much smoking increased the cost of health insurance. The variable importance metrics place it at the absolute top of the hierarchy. When looking at the extremes, the "Top 5%" of spenders are exclusively smokers, while the "Bottom 5%" are virtually all non-smokers. It’s not just a health risk; it’s a financial one.

This is the most important result from this study, by far:

<img width="2436" height="1157" alt="top 5 percent vs bottom 5 percent charges by smoker" src="https://github.com/user-attachments/assets/9aa73b19-f230-476c-8c39-c5bb9983a1d4" />

What I did is sort the Charges from largest (bad) to smallest (good). Then I broke it into two groups, the top 5% and bottom 5% of charges. What that chart shows is the top 5% of charges billed the insurance company over $3 million dollars, but the bottom 5% (same number of customers) only billed $102,197. <i>That's a difference of a factor of 30 for smokers vs non-smokers, for the same number of people in each group!</i>

The recommendation here is beyond obvious: Stop smoking today! It's good for your health (as everyone knows) but it's also good for your bank account.

2. The Heavyweight Contender: BMI

Same ideas as #1 (smoking), I looked at BMI (body mass index), and it came in 2nd place for strongly predicting insurance rates. In the visualization below, you can see the clear density of the highest-cost claimants hovering in the higher BMI brackets.

<img width="2436" height="1157" alt="Top 5% vs bottom 5% charges by BMI" src="https://github.com/user-attachments/assets/471351de-d5c6-4bbc-837a-7243d988a561" />

The effect of BMI is smaller than smoking, but still very significant.

3. The Relentless March: Age

I'm sure no one will be surprised that age is the 3rd strongest predictor from this data set.

<img width="2436" height="1157" alt="top 5% vs bottom 5% charges by age" src="https://github.com/user-attachments/assets/0419c97e-fbad-4deb-be66-9b943cfeec23" />

The Technical Verdict

To reach these conclusions, I utilized a large number of models (32 models), comparing everything from BayesGLM to Ensemble methods. The most accurate was a model named EnsembleEarth. You can see how well the predicted vs actual values lined up from this visualization:

<img width="1152" height="720" alt="ensemble_earth_pred_vs_actual" src="https://github.com/user-attachments/assets/7d6392c1-5da5-4689-8fa3-d70b564c9a74" />

<h2>You can play with an interactive dashboard of the results here</h2><br>https://gemini.google.com/share/59e5dbce54b0<br>
(**you might need to click on Preview to get the dashboard to work. You will not need to log in nor need a Google Gemini account to play with the dashboard, so have fun!!**)

Play with the sliders, checkboxes, and all the other interactive ways to connect with the data, to see the effects they have on insurance charges.

The Bottom Line

Data science confirms what doctors have said for years, but with the added weight of statistical significance: Lifestyle choices (Smoking and BMI) are the primary levers of insurance costs, often outweighing the inevitable impact of age.

The biggest issue with these results is that this is only one set of data, and there might be other data that is more accurately predictive of YOUR insurance charges!

#DataScience #MachineLearning #InsuranceAnalytics #PredictiveModeling #Statistics
