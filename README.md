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
To start off our research, we decided to focus on the 2024-2025 HMN student-level survey data. By doing so, we can compare how model performance and predictions change as we increase the size of our dataset. The 2024-2025 HMN dataset contains 84,735 responses, and we trained 4 different models on this dataset.

### Random Forest
We performed a structured modeling pipeline consisting of:
1. Data cleaning and preprocessing, including handling of missing values and selective encoding of categorical variables.
2. Feature engineering and reduction, designed to preserve meaningful predictors while avoiding high-cardinality variables that introduce noise or memory issues.
3. Supervised machine learning model training using a Random Forest classifier to predict the binary outcome variable `dx_dep`.
4. Model evaluation using accuracy, precision, recall, and F1-score to determine how well the model identifies students with diagnosed depression.
5. Feature importance analysis, identifying which survey items (e.g., mental health history, symptom measures, treatment variables, psychological factors) contributed most strongly to depression classification.

The Random Forest model achieved high predictive performance. 
* Accuracy: 93.50%
* Precision: 0.818
* Recall: 0.959
* F1-Score: 0.883

These evaluation results indicate that survey features in the HMN dataset contain strong signals related to whether a student has been diagnosed with depression. Importantly, the most influential predictors included other mental health diagnoses (e.g, anxiety and trauma), therapy and medication history, depression and anxiety symptom scores, loneliness indicators, and measures of impairment or unmet mental health needs. These findings align with established clinical literature and suggest that students experiencing broader mental health burdens or functional impairment are significantly more likely to report a depression diagnosis. See below for the bar graph of the top 20 predictors, which highlights a small subset of features that appear consistently predictive across tens of thousands of students.

<img width="1000" height="800" alt="rand_forest" src="https://github.com/user-attachments/assets/808b3195-cd38-4c33-913d-0708ff7343f5" />

[INSERT INTERPRETATION OF GRAPH]

Overall, this initial analysis demonstrates that machine learning methods can be successfully applied to large-scale student survey data to characterize the factors most strongly associated with depression risks. Specific predictive factors can be found and are informative for campus health professionals seeking to understand key correlates of depression in university populations. 

### Logistic Regression
We implemented a Logistic Regression classifier to evaluate how well a linear model with regularization can identify depression (dx_dep) among college students. The full modeling pipeline consisted of the same five steps from the Random Forest pipeline and an extra step before model training to scale the features. In other words, we standardized the input space to ensure that coefficient magnitudes are meaningful and that the optimization procedure converges reliably.

The Logistic Regression model also demonstrated strong predictive performance. 
* Accuracy: 97.81%
* Precision: 0.947
* Recall: 0.968
* F1-Score: 0.958

Compared to the performance evaluation of the Random Forest model, Logistic Regression is slightly better at identifying depression among college students. This suggests that [ADD INTERPRETATION HERE]. See below for the bar graph of the top 20 predictors using Logistic Regression:

<img width="1000" height="800" alt="log_reg" src="https://github.com/user-attachments/assets/6a8cf2fa-a8f7-4895-b235-2b492e48474d" />

[INSERT INTERPRETATION OF GRAPH]

### Multi-Layer Perceptron Neural Network w/o Hyperparameter Tuning
Using the same preprocessing steps as the Random Forest and Logistic Regression models, we trained an Multi-Layer Perceptron Neural Network on our depression dataset. Neural networks distribute learned information across multiple hidden layers and neurons, so it does not naturally provide feature importance (i.e., the MLP model cannot be used to find the top 20 predictors of depression among college students). Nonetheless, we implemented this model to test whether nonlinear relationships among student characteristics can improve the prediction of depression diagnoses. In other words, the MLP model provides an additional perspective on whether depression risk is driven by simple additive effects or higher-order patterns in the data. 

We implemented the MLP model with the following architecture:
* Two Hidden Layers of Sizes 128 and 64, respectively
* ReLU Activation Function
* Adam Optimizer (Adaptive Optimization of the Learning Rate)
* Cross-Entropy Loss Function
* L2 Regularization
* 50 Epochs

The MLP was trained on the 80% training split, and it involves repeatedly adjusting internal weights to minimize prediction error using backpropagation. 

*Note*: Since the MLP uses an adaptive optimizer, training converges reliably even without specific hyperparameter tuning.

After training, we evaluated the model on the unseen 20% test set. Here were the evaluation results of the model:
* Accuracy: 98.32%
* Precision: 0.975
* Recall: 0.959
* F1-Score: 0.967

Thus, the MLP model achieved the highest accuracy, precision, and F1-score among the three models we have implemented so far. This indicates that nonlinear machine learning approaches are well-suited for identifying patterns associated with depression in college students. The results suggest that depression risk is influenced by interacting factors, and neural models that capture these interactions can more accurately distinguish students who may be experiencing depressive symptons. See below for the loss function of the MLP model:

<img width="800" height="600" alt="mlp1" src="https://github.com/user-attachments/assets/01fbf4c8-cb4a-4fc6-aba2-43a2f0b39c9f" />

[INSERT INTERPRETATION OF GRAPH]

### Multi-Layer Perceptron Neural Network w/ Hyperparameter Tuning
To further optimize the performance of the MLP model, we conducted systematic hyperparameter tuning using GridSearchCV with 5-fold cross-validation. The training set was split into five equally sized folds, with four folds used for training and one for validation, rotating through all folds. The search explored a predefined hyperparameter grid that included:
* Hidden Layer Architecture: # of Layers and # of Neurons per Layer
* Activation Function: ReLU versus tanh
* Regularization Strength: L2 penalty parameter (alpha)
* Optimizer Settings: Different Learning Rates
* Training Schedule: # of Training Epochs

For each hyperparameter combination, the model was trained and evaluated on held-out folds, and the average validation score was used to select the best configuration. The final best-performing model was retrained on the full training dataset and then evaluated on the test set. Here were the best hyperparameters found:

{'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (64,), 'learning_rate_init': 0.001, 'max_iter': 200}

Besides the choice of activation function, the hyperparameters are different from the MLP model we trained without hyperparameter tuning. These hyperparameter values should transform our MLP model to achieve robust predictive performance while minimizing issues such as overfitting and unstable gradients. In fact, here were the evaluation results of this new tuned model:
* Accuracy: 98.37%
* Precision: 0.968
* Recall: 0.968
* F1-Score: 0.968

That means this hyperparameter tuned model achieved a higher accuracy, recall, and F1-score compared to our MLP Neural Network model without hyperparameter tuning, which is what we wanted!

See below for the loss function of the MLP model with the best hyperparameters found:

<img width="800" height="600" alt="mlpt1" src="https://github.com/user-attachments/assets/63ee84b6-89be-4e35-8867-be3c05b1f3e9" />

[INSERT INTERPRETATION OF GRAPH]

### Summary of Results
Comparing the evaluation metrics of each of the four models, we see that overall, the hyperparameter tuned multi-layer preceptron neural network performed the best. This implies [ADD INTERPRETATION HERE].

## Three Years of Data
We then tested our four models with three years worth of data: 2022-2025 survey responses. The modeling pipeline for each model remains the stay. In total, the combined dataset contains 205,213 responses. 

### Random Forest
* Accuracy: 92.52%
* Precision: 0.797
* Recall: 0.964
* F1-Score: 0.873

Top 20 Predictors Graph:

<img width="1000" height="800" alt="rand2" src="https://github.com/user-attachments/assets/b7d90d93-d841-4de0-8de2-7a8e6c8c1a1c" />

#### Comparison to 1 Year of Data
[INSERT COMMENTS ON EVAL METRICS AND HOW THE TOP 20 PREDICTORS CHANGED]

### Logistic Regression
* Accuracy: 98.05%
* Precision: 0.961
* Recall: 0.966
* F1-Score: 0.963

Top 20 Predictors Graph:

<img width="1000" height="800" alt="lr2" src="https://github.com/user-attachments/assets/403167b0-bfcf-4faa-8a02-814e32cc4d7e" />

#### Comparison to 1 Year of Data
[INSERT COMMENTS ON EVAL METRICS AND HOW THE TOP 20 PREDICTORS CHANGED]

### Multi-Layer Perceptron Neural Network w/o Hyperparameter Tuning
* Accuracy: 98.36%
* Precision: 0.978
* Recall: 0.960
* F1-Score: 0.969

Loss Curve Graph:

<img width="800" height="600" alt="mlp2" src="https://github.com/user-attachments/assets/9270c48b-afe6-4359-95b2-89db774a6511" />

#### Comparison to 1 Year of Data
[INSERT COMMENTS ON EVAL METRICS AND HOW THE LOSS CURVE CHANGED]

### Multi-Layer Perceptron Neural Network w/ Hyperparameter Tuning
Best Hyperparameters Found:

{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (128, 64, 32), 'learning_rate_init': 0.01, 'max_iter': 200}

* Accuracy: 98.66%
* Precision: 0.992
* Recall: 0.958
* F1-Score: 0.974

Loss Curve Graph:

<img width="800" height="600" alt="mlp" src="https://github.com/user-attachments/assets/7eaae0c9-77b3-4da2-8219-b855e3ada5f7" />

#### Comparison to 1 Year of Data
[INSERT COMMENTS ON EVAL METRICS AND HOW THE LOSS CURVE CHANGED]

### Summary of Results
Comparing the evaluation metrics of each of the four models with 3 years worth of data, we see that overall, the hyperparameter tuned multi-layer preceptron neural network performed the best. This result is similar to when we trained our models on 1 year worth of data. 

## Five Years of Data
