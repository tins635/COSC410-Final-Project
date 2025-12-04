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

## Hypothesis
Based on previous research that used ML algorithms to detect depression, we hypothesize that the Random Forest model will outperform the other machine learning algorithms in predicting depression among college students. We also hypothesize that the top predictors will revolve around stress levels and social support.

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

These evaluation results indicate that survey features in the HMN dataset contain strong signals related to whether a student has been diagnosed with depression. See below for the feature-importance graph, which highlights the most influential predictors of depression identified by the Random Forest model.

<img width="1000" height="800" alt="rand_forest" src="https://github.com/user-attachments/assets/808b3195-cd38-4c33-913d-0708ff7343f5" />

The strongest predictors are features related to prior mental-health diagnoses:
* `dx_any` (any mental-health diagnosis)
* `dx_dep_1`, `dx_dep_5` (various levels of depression diagnoses)
* `dx_anx`, `dx_ax_1` (anxiety diagnoses)
* `dx_none` (lack of diagnosis)

This suggests that a student's diagnostic history, particularly past depression or anxiety diagnoses, is the single most influential indicator of current depression risk. In fact, these findings align with established clinical literature and suggest that students experiencing broader mental health burdens are significantly more likely to report a depression diagnosis. The next three best predictors following past diagnoses are related to the student's experience with counseling or therapy for mental health concerns. We can imply from this that students who report that counseling or therapy has helped them with mental health concerns are also more likely to report a depression diagnosis.

Overall, this initial analysis demonstrates that machine learning methods can be successfully applied to large-scale student survey data to characterize the factors most strongly associated with depression risks. Specific predictive factors can be found and are informative for campus health professionals seeking to understand key correlates of depression in university populations. 

### Logistic Regression
We implemented a Logistic Regression classifier to evaluate how well a linear model with regularization can identify depression (dx_dep) among college students. The full modeling pipeline consisted of the same five steps from the Random Forest pipeline and an extra step before model training to scale the features. In other words, we standardized the input space to ensure that coefficient magnitudes are meaningful and that the optimization procedure converges reliably.

The Logistic Regression model also demonstrated strong predictive performance. 
* Accuracy: 97.81%
* Precision: 0.947
* Recall: 0.968
* F1-Score: 0.958

Compared with the Random Forest model, Logistic Regression performs slightly better at identifying depression among college students. This suggests that the underlying structure of depression risk in this dataset is relatively stable, predictable, and well-described by direct relationships between features and depression. Since patterns in students' mental-health profile are stable across the population, simple and interpretable models can work extremely well with this dataset. See below for the bar graph of the top 20 predictors using Logistic Regression:

<img width="1000" height="800" alt="log_reg" src="https://github.com/user-attachments/assets/6a8cf2fa-a8f7-4895-b235-2b492e48474d" />

Since Logistic Regression is a linear, additive model, each coefficient represents the direct increase in log-odds of depression associated with that predictor. That means larger positive coefficients indicate strong associations with depression risk. The strongest predictors are:
* `dx_dep_1`, `dx_dep_5`, `dx_dep_2`, `dx_dep_4`, `dx_dep_3`
* `dx_any`

These represent various depression-diagnosis categories. Their large positive coefficients indicate that having any prior diagnosis of depression is by far the strongest predictor of current depression. This mirrors the Random Forest results, where diagnostic history was also the top predictors. However, Logistic Regression introduces several new variables in the ranking. The two best predictors that follow prior diagnosis of depression are:
* `stress1`
* `achieve4`

`stress1` captures the students' extent to which they agree that experiencing stress depletes health and vitality, whereas `achieve4` exemplifies the students' agreement with the statement that it is important that the courses they take offer them a challenge. Both of these variables build off each other, indicating that academic-related factors are also linearly related to depression. 

### Multi-Layer Perceptron Neural Network w/o Hyperparameter Tuning
Using the same preprocessing steps as the Random Forest and Logistic Regression models, we trained a Multi-Layer Perceptron Neural Network on our depression dataset. Neural networks distribute learned information across multiple hidden layers and neurons, so it does not naturally provide feature importance (i.e., the MLP cannot be used to find the top 20 predictors of depression among college students). Nonetheless, we implemented this model to test whether nonlinear relationships among student characteristics can improve the prediction of depression diagnoses. In other words, the MLP provides an additional perspective on whether depression risk is driven by simple additive effects or higher-order patterns in the data. 

We implemented the MLP with the following architecture:
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

Thus, the MLP achieved the highest accuracy, precision, and F1-score among the three models we have implemented so far. This indicates that nonlinear machine learning approaches are well-suited for identifying patterns associated with depression in college students. The results suggest that depression risk is influenced by interacting factors, and neural models that capture these interactions can more accurately distinguish students who may be experiencing depressive symptoms. See below for the loss function of the MLP:

<img width="800" height="600" alt="mlp1" src="https://github.com/user-attachments/assets/01fbf4c8-cb4a-4fc6-aba2-43a2f0b39c9f" />

From the loss curve graph, we see that the loss decreases very rapidly within the first few epochs and at around epoch 8, the loss approaches near zero. This may be a sign of mild overfitting.

### Multi-Layer Perceptron Neural Network w/ Hyperparameter Tuning
To further optimize the performance of the MLP, we conducted systematic hyperparameter tuning using GridSearchCV with 5-fold cross-validation. The training set was split into five equally sized folds, with four folds used for training and one for validation, rotating through all folds. The search explored a predefined hyperparameter grid that included:
* Hidden Layer Architecture: # of Layers and # of Neurons per Layer
* Activation Function: ReLU versus tanh
* Regularization Strength: L2 penalty parameter (alpha)
* Optimizer Settings: Different Learning Rates
* Training Schedule: # of Training Epochs

For each hyperparameter combination, the model was trained and evaluated on held-out folds, and the average validation score was used to select the best configuration. The final best-performing model was retrained on the full training dataset and then evaluated on the test set. Here were the best hyperparameters found:

`{'activation': 'relu', 'alpha': 0.01, 'hidden_layer_sizes': (64,), 'learning_rate_init': 0.001, 'max_iter': 200}`

Besides the choice of activation function, the hyperparameters are different from the MLP we trained without hyperparameter tuning. These hyperparameter values help the MLP achieve robust predictive performance while minimizing issues such as overfitting and unstable gradients. In fact, here were the evaluation results of this new tuned model:
* Accuracy: 98.37%
* Precision: 0.968
* Recall: 0.968
* F1-Score: 0.968

That means this hyperparameter tuned model achieved a higher accuracy, recall, and F1-score compared to our MLP Neural Network model without hyperparameter tuning, which is what we wanted! In other words, the tuned model is more balanced, slightly less overfitted, and better at consistently identifying depression cases without excessive false negatives.

See below for the loss function of the MLP with the best hyperparameters found:

<img width="800" height="600" alt="mlpt1" src="https://github.com/user-attachments/assets/63ee84b6-89be-4e35-8867-be3c05b1f3e9" />

Comparing this loss curve with that of the MLP without hyperparameter tuning, we see the MLP with tuning learns more gradually and has a healthier training loss plateau. This is because the MLP with tuning is intentionally constrained to avoid overfitting, while the MLP without tuning fits the training data much more aggressively.

### Summary of Results
Comparing the evaluation metrics of each of the four models, we see that overall, the hyperparameter tuned multi-layer perceptron neural network performed the best. This implies that the relationship between survey features and depression is not purely linear. Whereas Logistic Regression and even Random Forests capture useful structure, the MLP's deeper, non-linear architecture is better able to learn complex interactions among mental-health, demographic, and behavioral variables. In other words, the MLP is able to capture subtle patterns that straightforward models may overlook. Thus, deep learning algorithms can be effective in predicting depression within college students when properly designed. 

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
In terms of the evaluation metrics, we see an increase in recall while there is a decrease in accuracy, precision, and F1-score. This means the Random Forest model became more sensitive and inclusive in detecting depression but at the cost of accuracy and precision due to higher data variability, class imbalance, or changes in underlying student populations over multiple years. In terms of the top 20 predictors, a majority of the top 20 predictors stayed the same, with some changes in their ordering. That means, even as the size of our dataset increases, when using the Random Forest model, the best predictors for depression within college students are past depression and anxiety diagnoses.

### Logistic Regression
* Accuracy: 98.05%
* Precision: 0.961
* Recall: 0.966
* F1-Score: 0.963

Top 20 Predictors Graph:

<img width="1000" height="800" alt="lr2" src="https://github.com/user-attachments/assets/403167b0-bfcf-4faa-8a02-814e32cc4d7e" />

#### Comparison to 1 Year of Data
Unlike the Random Forest model, the Logistic Regression model improved in terms of accuracy, precision, and F1-score. That means Logistic Regression was able to capture stable, linear patterns that held up even as more data was added. This suggests that depression indicators in our dataset have a strong, linear separability component. In terms of the top 20 predictors, the top 6 predictors remained the same as those from the smaller dataset. However, we see a drastic change in the remaining predictors. For example, `stress1` and `achieve4` are no longer part of the top 20 predictors. In fact, more predictors in regards to peer relationships appear in the top 20. This suggests that social factors are also linearly related to depression.

### Multi-Layer Perceptron Neural Network w/o Hyperparameter Tuning
* Accuracy: 98.36%
* Precision: 0.978
* Recall: 0.960
* F1-Score: 0.969

Loss Curve Graph:

<img width="800" height="600" alt="mlp2" src="https://github.com/user-attachments/assets/9270c48b-afe6-4359-95b2-89db774a6511" />

#### Comparison to 1 Year of Data
Performance and the loss curve are nearly identical to the MLP without hyperparameter tuning on 1 year's worth of survey responses. This suggests that the untuned MLP is already very stable and generalizes well. 

### Multi-Layer Perceptron Neural Network w/ Hyperparameter Tuning
Best Hyperparameters Found:

`{'activation': 'relu', 'alpha': 0.0001, 'hidden_layer_sizes': (128, 64, 32), 'learning_rate_init': 0.01, 'max_iter': 200}`

* Accuracy: 98.66%
* Precision: 0.992
* Recall: 0.958
* F1-Score: 0.974

Loss Curve Graph:

<img width="800" height="600" alt="mlp" src="https://github.com/user-attachments/assets/7eaae0c9-77b3-4da2-8219-b855e3ada5f7" />

#### Comparison to 1 Year of Data
There is improvement in accuracy, precision, and F1-score, which suggests that the tuned MLP becomes more confident and more selective. In other words, with more data, we have more refined decision boundaries. 

### Summary of Results
Comparing the evaluation metrics of each of the four models with 3 years' worth of data, we see that overall, the hyperparameter tuned multi-layer perceptron neural network performed the best. This result is similar to when we trained our models on 1 year's worth of data, which suggests that the tuned MLP is the strongest model for detecting depression among college students. We notice that Random Forest got worse with more data in terms of balance and reliability. On the other hand, Logistic Regression becomes even more consistent and trustworthy with more survey responses. MLP without hyperparameter tuning remains unaffected, meaning the model is robust. 

## Five Years of Data



## Results
The Random Forest model exhibited strong performance, though slightly lower than the logistic regression and the multi-layer perceptron (MLP) models. Its high recall across the different sizes of data indicates that the ensemble method is effective at identifying students who are at risk of depression. However, the model's comparatively lower precision and accuracy suggest that while it is sensitive to depressive signals, it is more prone to false positives than the other approaches. This pattern implies that Random Forest may be overfitting subtle noise or less informative features in the data. Nonetheless, the Random Forest model serves as a valuable complementary model, confirming the robustness of key predictors while revealing additional interactions that linear models may understate.

Both the logistic regression model and the MLP demonstrated exceptionally strong predictive performance, suggesting important insights about the underlying structure of depression risk within the dataset of college students. Logistic regression, which is a linear classifier, achieved accuracy, precision, recall, and F1-scores exceeding 96%. This indicates that a substantial portion of the relationship between the survey features and depression risk is effectively captured through linear combinations of predictors. In other words, many risk factors influence depression in a directionally consistent, additive manner, so the core structure of the data is largely linearly separable. 

The MLP models, both with and without hyperparameter tuning, achieved slightly higher performance metrics, most notably in precision and overall F1-score. Neural networks are capable of detecting nonlinear interactions, threshold effects, and more complex patterns that logistic regression cannot capture. Their improved performance indicates that the dataset does contain nonlinear relationships and interactions among predictors, but these patterns are not dominant drivers of depression classification. Instead, the nonlinear components serve to refine and enhance predictive accuracy rather than define it. 

Put together, the strong performance of logistic regression paired with the improvements from the MLP suggest that the factors contributing to depression among college students are primarily linear, with secondary nonlinear interactions that further improve model precision when captured. This provides an important methodological insight: simpler linear models may already offer highly effective screening capabilities, while more complex neural architectures can provide marginal gains by modeling subtler dependencies between variables. This balance supports robustness in the findings and reinforces the reliability of the identified predictors across different modeling approaches. 

## Conclusions
### Research Goal 1
Since the MLP models do not provide interpretable coefficients or feature importance values, our analysis of predictive factors relies solely on the Random Forest and Logistic Regression models. Across these two interpretable approaches, a consistent and highly concentrated set of predictors emerged as the most influential. Nearly all of the strongest predictors were variables directly related to students' prior mental health status, especially:
* previous diagnosis of depression or anxiety,
* history of counseling or therapy, and
* current self-assessment of mental health.

This suggests that college students with prior depressive symptoms or diagnoses are substantially more likely to report a current diagnosis of depression. Other factors, such as stress and peer relationships, also contributed to prediction, but their effect sizes were smaller and far less dominant. The overwhelming predictive strength of prior mental health indicators highlights the continuity and persistence of depression over time within college students. Overall, the models reveal that depression risk in the HMS dataset is driven primarily by students' mental health history. 

### Research Goal 2
All three modeling approaches performed exceptionally well in predicting whether a student reported a depression diagnosis. Logistic regression and MLP models achieved exceptionally high acrruacy, precision, recall, and F1-scores. This demonstrates that depression classification is highly feasible using only survey data. The tuned MLP achieved the highest overall performance, suggesting that although the underlying structure of the data is primarily linear, nonlinear patterns and interactions further enhance predictive ability. Random Forest models achieved strong recall with slightly lower precision and overall accuracy. this indicates that while Random Forest is effective in detecting depressive symptoms, it is more prone to false positives relative to the neural network and logistic regression models. Nonetheless, its feature importance results align closely with those of logistic regression, which reinforces the reliability of the identified predictors. 

### Overall Implications
College institutions can feasibly implement machine learning algorithms as early detection or screening tools to identify students who may be at elevated risk of depression, provided appropriate ethical safeguards are in place. The consistency of key predictors across models highlights potential areas for targeted intervention, particularly previous mental health diagnoses, therapy, stress, peer relationships, and campus support systems.
