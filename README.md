# COSC410-Final-Project
We are seeking the most effective model that can accurately classify or predict depression based on survey responses. We hope to identify the most influential features associated with depression risk using the model, and success will be achieved through the attainment of high accuracy, precision, recall, and F1-score.

## Introduction
Depression is one of the most prevalent mental health concerns among college students in the United States and is associated with reduced academic performance, increased dropout risk, social impairment, and poorer overall well-being. Understanding which factors increase students' risk for depression is therefore essential for designing campus-level prevention and intervention strategies. Large-scale survey datasets, such as those from the Healthy Minds Network (HMN), offer an opportunity to examine mental health patterns across tens of thousands of students and to identify which demographic, psychosocial, academic, or clinical features are most strongly associated with depression. Machine learning approaches can analyze high-dimensional data, handle nonlinear interactions, accommodate mixed data types, and provide interpretable measures of feature importance. 

## Dataset
We use the Healthy Minds Study dataset. The data is separated by year, and we requested access to data from the past 10 years. Attached to this repository is a PDF of the 2024-2025 Codebook, which outlines all survey items included in the HMS Student Survey and the variable names in the clean datasets. Use the following link to go to their website and request access to the data if you want to do your own research on it:

https://healthymindsnetwork.org/research/data-for-researchers/ 

## Goals of the Study
1. to determine which features in the HMS student survey dataset most strongly predict whether a student reports having been diagnosed with depression, and
2. to evaluate how accurately a machine learning model can classify students as reporting a depression diagnosis (dx_dep) or not based on their survey responses.

## One Year of Data 
To start off our research, we decided to focus on the 2024-2025 HMN student-level survey data. By doing so, we can compare how model performance and predictions change as we increase the size of our dataset. The 2024-2025 HMN dataset contains [INSERT NUMBER OF ROWS/SURVEY RESPONSES] responses, and we trained 4 different models on this dataset.

### Random Forest Model
We performed a structured modeling pipeline consisting of:
1. Data cleaning and preprocessing, including handling of missing values and selective encoding of categorical variables.
2. Feature engineering and reduction, designed to preserve meaningful predictors while avoiding high-cardinality variables that introduce noise or memory issues.
3. Supervised machine learning model training using a Random Forest classifier to predict the binary outcome variable `dx_dep`.
4. Model evaluation using accuracy, precision, recall, and F1-score to determine how well the model identifies students with diagnosed depression.
5. Feature importance analysis, identifying which survey items (e.g., mental health history, symptom measures, treatment variables, psychological factors) contributed most strongly to depression classification.

The Random Forest model achieved high predictive performance. It had an accuracy of 93.5%, precision of 0.818, recall of 0.959, and F1-score of 0.883. These evaluation results indicate that survey features in the HMN dataset contain strong signals related to whether a student has been diagnosed with depression. Importantly, the most influential predictors included other mental health diagnoses (e.g, anxiety and trauma), therapy and medication history, depression and anxiety symptom scores, loneliness indicators, and measures of impairment or unmet mental health needs. These findings align with established clinical literature and suggest that students experiencing broader mental health burdens or functional impairment are significantly more likely to report a depression diagnosis. See below for the bar graph of the top 20 predictors, which highlights a small subset of features that appear consistently predictive across tens of thousands of students.

<img width="886" height="660" alt="randomForest" src="https://github.com/user-attachments/assets/2f2b0132-6a1a-461d-9c71-6db7a78960e2" />

Overall, this initial analysis demonstrates that machine learning methods can be successfully applied to large-scale student survey data to characterize the factors most strongly associated with depression risks. Specific predictive factors can be found and are informative for campus health professionals seeking to understand key correlates of depression in university populations. 

### Logistic Regression Model
We implemented a Logistic Regression classifier to evaluate how well a linear model with regularization can identify depression (dx_dep) among college students. The full modeling pipeline consisted of the same five steps from the Random Forest pipeline and an extra step before model training to scale the features. In other words, we standardized the input space to ensure that coefficient magnitudes are meaningful and that the optimization procedure converges reliably.

The Logistic Regression model demonstrated strong predictive performance. It had an accuracy of 97.81%, precision of 0.947, recall of 0.968, and F1-score of 0.958. Compared to the performance evaluation of the Random Forest model, Logistic Regression is slightly better at identifying depression among college students. This suggests that [ADD INTERPRETATION HERE]. See below for the bar graph of the top 20 predictors using Logistic Regression:

<img width="766" height="606" alt="LogistcReg" src="https://github.com/user-attachments/assets/c48473b5-a967-4220-b31e-6137792fdcdf" />

### Multi-Layer Perceptron Neural Network w/o Hyperparameter Tuning
Using the same preprocessing steps as the Random Forest and Logistic Regression models, we trained an Multi-Layer Perceptron Neural Network on our depression dataset. Neural networks distribute learned information across multiple hidden layers and neurons, so it does not naturally provide feature importance (i.e., the MLP model cannot be used to find the top 20 predictors of depression among college students). Nonetheless, we implemented this model to test whether nonlinear relationships among student characteristics can improve the prediction of depression diagnoses. In other words, the MLP model provides an additional perspective on whether depression risk is driven by simple additive effects or higher-order patterns in the data. 

We implemented the MLP model with the following architecture:
* Two Hidden Layers
* ReLU Activation Function
* Adam Optimizer (Adaptive Optimization of the Learning Rate)
* Cross-Entropy Loss Function
* L2 Regularization

The MLP was trained on the 80% training split, and it involves repeatedly adjusting internal weights to minimize prediction error using backpropagation. 

*Note*: Since the MLP uses an adaptive optimizer, training converges reliably even without specific hyperparameter tuning.

After training, we evaluated the model on the unseen 20% test set. It had an accuracy of 98.38%, precision of 0.977, recall of 0.959, and F1-score of 0.968. Thus, the MLP model achieved the highest accuracy, precision, and F1-score among the three models we have implemented so far. This indicates that nonlinear machine learning approaches are well-suited for identifying patterns associated with depression in college students. The results suggest that depression risk is influenced by interacting factors, and neural models that capture these interactions can more accurately distinguish students who may be experiencing depressive symptons. See below for the loss function of the MLP model:

<img width="800" height="600" alt="MLP_NN" src="https://github.com/user-attachments/assets/e552c9e3-cbf0-4788-95f9-8c2154a53414" />

### Multi-Layer Perceptron Neural Network w/ Hyperparameter Tuning
To further optimize the performance of the MLP model, we conducted systematic hyperparameter tuning using GridSearchCV with 5-fold cross-validation. The training set was split into five equally sized folds, with four folds used for training and one for validation, rotating through all folds. The search explored a predefined hyperparameter grid that included:
* Hidden Layer Architecture: # of Layers and # of Neurons per Layer
* Activation Function: ReLU versus tanh
* Regularization Strength: L2 penalty parameter (alpha)
* Optimizer Settings: Different Learning Rates
* Training Schedule: # of Training Epochs

For each hyperparameter combination, the model was trained and evaluated on held-out folds, and the average validation score was used to select the best configuration. The final best-performing model was retrained on the full training dataset and then evaluated on the test set. Here were the results:

[ADD THE HYPERPARAMETER TUNING RESULTS HERE]

This approach allows the MLP model to achieve robust predictive performance while minimizing issues such as overfitting and unstable gradients. This new tuned model had [ADD EVALUATION REULTS HERE].

### Summary of Results
Comparing the evaluation metrics of each of the four models, we see that overall, [INSERT MODEL TYPE] performed the best. This implies [ADD INTERPRETATION HERE].

## Three Years of Data
