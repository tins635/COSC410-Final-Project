# COSC410-Final-Project
We are seeking the most effective model that can accurately classify or predict depression based on survey responses. We hope to identify the most influential features associated with depression risk using the model, and success will be achieved through the attainment of high accuracy, precision, recall, and F1-score.

## Introduction
Depression is one of the most prevalent mental health concerns among college students in the United States and is associated with reduced academic performance, increased dropout risk, social impairment, and poorer overall well-being. Understanding which factors increase students' risk for depression is therefore essential for designing campus-level prevention and intervention strategies. Large-scale survey datasets, such as those from the Healthy Minds Network (HMN), offer an opportunity to examine mental health patterns across tens of thousands of students and to identify which demographic, psychosocial, academic, or clinical features are most strongly associated with depression.

## Dataset
We use the Healthy Minds Study dataset. The data is separated by year, and we requested access to data from the past 10 years. Attached to this repository is a PDF of the 2024-2025 Codebook, which outlines all survey items included in the HMS Student Survey and the variable names in the clean datasets.

## Goals of the Study
1. to determine which features in the HMS student survey dataset most strongly predict whether a student reports having been diagnosed with depression, and
2. to evaluate how accurately a machine learning model can classify students as reporting a depression diagnosis (dx_dep) or not based on their survey responses.

## Milestone 1
Machine learning approaches can analyze high-dimensional data, handle nonlinear interactions, accommodate mixed data types, and provide interpretable measures of feature importance. Inspired by similar existing publications on depression, we decided to run a random forest model for our first analysis. We initially analyzed the data from 2020 to 2025 using both logistic regression and random forest models. However, the program ran into an error: the laptop we used could not handle all the data. Thus, for this milestone, we decided to focus on the 2024-2025 HMN student-level survey data and use the Random Forest model type. We performed a structured modeling pipeline consisting of:
1. Data cleaning and preprocessing, including handling of missing values and selective encoding of categorical variables.
2. Feature engineering and reduction, designed to preserve meaningful predictors while avoiding high-cardinality variables that introduce noise or memory issues.
3. Supervised machine learning model training using a Random Forest classifier to predict the binary outcome variable `dx_dep`.
4. Model evaluation using accuracy, precision, recall, and F1-score to determine how well the model identifies students with diagnosed depression.
5. Feature importance analysis, identifying which survey items (e.g., mental health history, symptom measures, treatment variables, psychological factors) contributed most strongly to depression classification.

The Random Forest model achieved high predictive performance. It had an accuracy of 93.5%, precision of 0.818, recall of 0.959, and F1-score of 0.883. These evaluation results indicate that survey features in the HMN dataset contain strong signals related to whether a student has been diagnosed with depression. Importantly, the most influential predictors included other mental health diagnoses (e.g, anxiety and trauma), therapy and medication history, depression and anxiety symptom scores, loneliness indicators, and measures of impairment or unmet mental health needs. These findings align with established clinical literature and suggest that students experiencing broader mental health burdens or functional impairment are significantly more likely to report a depression diagnosis. See below for the bar graph of the top 20 predictors, which highlights a small subset of features that appear consistently predictive across tens of thousands of students.

<img width="886" height="660" alt="randomForest" src="https://github.com/user-attachments/assets/2f2b0132-6a1a-461d-9c71-6db7a78960e2" />

Overall, this initial analysis demonstrates that machine learning methods can be successfully applied to large-scale student survey data to characterize the factors most strongly associated with depression risks. Specific predictive factors can be found and are informative for campus health professionals seeking to understand key correlates of depression in university populations. 

## Next Steps
We were only able to successfully train the data on a Random Forest model. We hope to be able to train our model with more data from previous years, which can be done by training a different Random Forest model for each year's worth of survey responses. Furthermore, we hope to be able to test at least 3 more different model types on the dataset. This is because we want to be able to compare the different machine learning algorithms and see which one will be the best predictor.

## Progress Made on 11/26
To complement the Random Forest model, we implemented a Logistic Regression classifier to evaluate how well a linear model with regularization can identify depression (dx_dep) among college students. The full modeling pipeline consisted of the same five steps from the Random Forest pipeline and an extra step before model training to scale the features. In other words, we standardized the input space to ensure that coefficient magnitudes are meaningful and that the optimization procedure converges reliably.

The Logistic Regression model demonstrated strong predictive performance. It had an accuracy of 97.81%, precision of 0.947, recall of 0.968, and F1-score of 0.958. Compared to the performance evaluation of the Random Forest model, Logistic Regression is slightly better at identifying depression among college students. See below for the bar graph of the top 20 predictors using Logistic Regression:

<img width="766" height="606" alt="LogistcReg" src="https://github.com/user-attachments/assets/c48473b5-a967-4220-b31e-6137792fdcdf" />

