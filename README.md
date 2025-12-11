# COSC410-Final-Project
**Predicting Depression in College Students Using Machine Learning**

We aim to identify the most effective machine learning model for predicting depression using large-scale student survey responses. Our goals are:
1. **Prediction**: Determine which models most accurately classify depression.
2. **Interpretation**: Identify which features reliably predict depression risk.

## 1. Introduction
Depression is one of the most prevalent mental health concerns among college students in the United States and is associated with reduced academic performance, increased dropout risk, social impairment, and poorer overall well-being. Understanding which factors increase students' risk for depression is therefore essential for designing campus-level prevention and intervention strategies. Large-scale survey datasets, such as those from the Healthy Minds Network (HMN), offer an opportunity to examine patterns across tens of thousands of students and identify which demographic, psychosocial, academic, or clinical features are most strongly associated with depression. Machine learning is well suited to this task because it can:
- Process high-dimensional mixed data
- Capture nonlinear relationships
- Provide interpretable indicators of risk (for some models)

Our work evaluates multiple machine learning models and compares performance across:
- One Year of Data (2024-2025)
- Three Years of Data (2022-2025)

Later sections will explore a version of the analysis excluding mental-health-related predicors, testing whether depression can still be predicted from nonclinical factors. 

## 2. Dataset
We use the **Healthy Minds Study** student-level survey dataset.
- **1-year dataset (2024-2025)**: 84,735 responses
- **3-year dataset (2022-2025)**: 205,213 responses

A PDF codebook (2024-2025 edition) is included in this repository. To request HMN data, visit: https://healthymindsnetwork.org/research/data-for-researchers/ 

## 3. Goals and Hypotheses
### Goals
1. Identify survey features that most strongly predict depression
2. Evaluate classification performance across multiple ML approaches

### Hypotheses
Based on previous research that used ML algorithms to detect depression, we hypothesize that
- Random Forest will outperform other models.
- Top predictors will be stress and social support.

## 4. Modeling Pipeline
All models share the same core pipeline:
1. Data cleaning & handling missing values
2. Encoding categorical variables
3. Standardization (Logistic Regression & MLP)
4. Model training using the selected algorithm
5. Evaluation using:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
6. (For Random Forest) Feature Importance for Top 20 Predictors
7. (For Logistic Regression) Coefficient Values for Top 20 Predictors
8. (For MLP) Loss Curve Visualization
9. (For Tuned MLP) Additional hyperparameter tuning via GridSearchCV

## 5. Results: 1-Year vs. 3-Years Data
### Random Forest
#### 1-Year Performance
`Accuracy : 0.9350327491591433`

`Precision: 0.8178768745067088`

`Recall   : 0.9588248901226001`

`F1 Score : 0.8827600894473432`

<img width="1000" height="800" alt="rand_forest_preds1" src="https://github.com/user-attachments/assets/6ceef7b7-38ff-4c86-9974-0df263f1f9d6" />

#### 3-Year Performance
`Accuracy : 0.9252491289623078`

`Precision: 0.7974232663887836`

`Recall   : 0.9638178986901164`

`F1 Score : 0.8727604512276046`

<img width="1000" height="800" alt="rand_forest_preds2" src="https://github.com/user-attachments/assets/18f39247-e486-47bc-8c40-0799c9b63696" />

#### Comparison & Interpretation
Across larger data, Random Forest becomes more sensitive (higher recall) but less precise. This suggests that combining years and having more data responses increase population variability and noise, making Random Forest over-inclusive. Importantly, the top predictors remain nearly identical, confirming that diagnostic history is a stable and dominant signal. 

### Logistic Regression
#### 1-Year Performance
`Accuracy : 0.9781082197439075`

`Precision: 0.947463768115942`

`Recall   : 0.9678464029609067`

`F1 Score : 0.9575466300492047`

<img width="1000" height="800" alt="log_reg_preds1" src="https://github.com/user-attachments/assets/98f97dca-1fa3-43dd-8392-106d74955f4a" />

#### 3-Year Performance
`Accuracy : 0.9804838827571084`

`Precision: 0.9610756608933455`

`Recall   : 0.9657415040762114`

`F1 Score : 0.9634029332480467`

<img width="1000" height="800" alt="log_reg_preds2" src="https://github.com/user-attachments/assets/50d56f8a-78a1-4634-8646-32811b970e06" />

#### Comparison & Interpretation
Logistic Regression improves with more data, indicating the underlying structure is strongly linear and scales well. Some predictors shift (`stress1` drops out), but diagnostic history still dominates, with social/peer-relationship features becoming stronger.

### Multi-Layer Perceptron (No Hyperparameter Tuning)
#### 1-Year Performance
`Accuracy : 0.9832418717177082`

`Precision: 0.9752883031301482`

`Recall   : 0.9585935692805921`

`F1 Score : 0.966868875408306`

<img width="800" height="600" alt="MLP_Loss1" src="https://github.com/user-attachments/assets/d30e1e74-e383-4819-ad64-97becfdf5fb9" />

#### 3-Year Performance
`Accuracy : 0.9836269278561509`

`Precision: 0.9779789120089577`

`Recall   : 0.9600622881744069`

`F1 Score : 0.9689377831191641`

<img width="800" height="600" alt="MLP_Loss2" src="https://github.com/user-attachments/assets/aa9350f4-08e4-4a3c-9222-87c54d9bdee0" />

#### Comparison & Interpretation
Performance is almost unchanged. The model already generalizes well, suggesting that even the untuned MLP captures stable nonlinear patterns without being heavily influenced by dataset expansion.

### Multi-Layer Perceptron (With Hyperparameter Tuning)
#### 1-Year Performance


#### 3-Year Performance


#### Comparison & Interpretation
The tuned MLP becomes even stronger with more data, developing sharper decision boundaries and achieving the highest precision of any model. This suggests that depression prediction includes nonlinear interactions that deep learning captures best.

### Summary Across All Models
Across both datasets, the MLP with hyperparameter tuning consistently performs the best, followed closely by the untuned MLP and Logistic Regression. Random Forest is the only model that degrades slightly with additional years of data. Across all models, prior mental-health diagnoses dominate the top 20 predictors, confirming that past clinical history is the strongest predictor of depression in this dataset.

## 6. Excluding Mental-Health Predictors
To evaluate whether depression can still be predicted using non-mental-health variables, we constructured an alternative preprocessing pipeline that removes all features directly tied to prior mental-health history. Below are the performances of each model with 1 year of data.

### Random Forest
`Accuracy : 0.8521862276509117`

`Precision: 0.7011061946902655`

`Recall   : 0.7330557483229239`

`F1 Score : 0.7167250932941309`

<img width="1000" height="800" alt="RF_FILTERED" src="https://github.com/user-attachments/assets/89443500-ae14-4224-8aab-92039605d798" />

### Logistic Regression
`Accuracy : 0.8266359827698118`

`Precision: 0.6174325928438189`

`Recall   : 0.8422391857506362`

`F1 Score : 0.7125244618395304`

<img width="1000" height="800" alt="LF_FILTERED" src="https://github.com/user-attachments/assets/69528517-7163-4925-b4d1-4a2badc174e2" />

### MLP w/o Hyperparameter Tuning
`Accuracy : 0.8251017879270668`

`Precision: 0.6621331424481031`

`Recall   : 0.6419153365718251`

`F1 Score : 0.6518675123326286`

<img width="800" height="600" alt="MLP1_FILTERED" src="https://github.com/user-attachments/assets/d1f97be6-bb9d-42a6-ad7e-394df78345cc" />

### MLP w/ Hyperparameter Tuning
`Accuracy : 0.854310497433174`

`Precision: 0.7498652291105121`

`Recall   : 0.6435345824658801`

`F1 Score : 0.6926428482509647`

<img width="800" height="600" alt="MLP2_FILTERED" src="https://github.com/user-attachments/assets/903ff0fe-fd9e-435e-bfdd-0d8b37e1aac1" />

### Analysis
Even after excluding all prior mental-health indicators, the filtered models identified several variables that remained moderately predictive of depression. 
- Random Forest emphasized perceived need for support, general disability indicators, `talk1_1` (communication/connection), satisfaction metrics, and several behavorial items. These predictors reflect broader functional impairments and psychosocial factors rather than direct mental-health history.
- Logistic Regression highlighted lifestyle and academic engagement variables such as drug use categories, satisfaction measures, academic achievement, loneliness, and peer interactions. These coefficients suggest that social isolation, academic pressure, and substance-use-related behaviors may correlate with depression risk even without explicit clinical features.

The exclusion of mental-healh variables expectedly reduced predictive performance across all models compared to the original full-feature models. However, accuracy and F1-scores remained suprisingly strong, indicating that non-clinical variables still preserve meaningful signal related to depression risk.
- Random Forest delivered the strongest overall performance, which is the complete opposite of the original full-feature performance.
- Logistic Regression achieved the highest recall, meaning it identified a larger proportion of individuals with depression, though at the cost of reduced precision.
- The performance of the MLP w/o Hyperparameter Tuning declined mostly, indicating a loss of strong nonlinear mental-health predictors.
- The tuned MLP achieved the highest overall accuracy, though with more conservative recall.

These new models suggest that depression correlates with broader social, behavorial, and functional dimensions, not just clinical indicators. However, with the decline in performance, we see that non-clinical factors help, but they do not replace the value of mental-health-specific information. This filtered-feature experiment demonstrates the potential (and limitations) of early-screening models that rely soley on general behavorial or demographic data. It also highlights opportunities for improving the fairness and accessibility of predictive screening tools by reducing reliance on clinical histories that not all individuals have equal access to.
