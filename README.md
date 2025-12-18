Project Title: Customer Churn Detection

Introduction

This project focuses on the development and comparative evaluation of machine learning and deep learning models for predicting customer churn in the telecommunications domain using the Telco Customer Churn dataset. The dataset contains a combination of numerical and categorical customer attributes, including demographic information, service usage patterns, contract details, and billing information. A robust data preprocessing pipeline was implemented to handle missing values, normalize numerical features, and encode categorical variables using one-hot encoding, ensuring compatibility across all models. Three predictive approaches were investigated: Logistic Regression as a baseline linear model, Random Forest as a non-linear ensemble method, and a deep learning–based Multilayer Perceptron (MLP) implemented in PyTorch and integrated with scikit-learn using the Skorch framework. The deep learning model incorporates multiple fully connected layers with ReLU activations, dropout-based regularization, and a class-weighted binary cross-entropy loss function to effectively address class imbalance. Hyperparameter optimization for all models was conducted using GridSearchCV with cross-validation, and model performance was assessed using F1 score and ROC-AUC metrics to account for the imbalanced nature of the churn prediction task. The project culminates in a comprehensive comparative analysis, identifying the most effective modeling approach and producing a deployable end-to-end prediction pipeline for real-world churn prediction applications.

Features Used and Input Shape

In our project, features are automatically categorized by the pipeline into numerical and categorical types.

Raw Input Features

The model uses data from the IBM Telco Customer Churn dataset, excluding the customerID and the target Churn.

•	Numerical Features: tenure, MonthlyCharges, and TotalCharges.

•	Categorical Features: gender, SeniorCitizen, Partner, Dependents, PhoneService, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies, Contract, PaperlessBilling, and PaymentMethod.

Processed Feature Count

While the raw dataset has 19 features, the OneHotEncoder expands categorical variables into multiple binary columns.

•	Encoding Expansion: For example, a feature like PaymentMethod (with 4 levels) is split into 4 separate columns.

•	Total Model Input: After this preprocessing, the number of features increases significantly. Based on standard encoding for this dataset, your num_features variable (used by the MLP) is typically around 45 to 50 features.

Feature Distribution Curves (KDE)

The added graph uses Kernel Density Estimation (KDE) to visualize how the data is spread:

•	Smoothing: The KDE curve provides a smooth line over the histogram, making it easier to see the probability density of a feature.

•	Churn vs. No Churn: By coloring the curves by the Churn label, we can see that customers with low tenure have a much higher "density" of Churning (the peaks are separated), while customers with high tenure are more likely to stay.

•	Bimodal Nature: The Total Charges curve for "No Churn" often shows two peaks, indicating the model is seeing two distinct types of loyal customers (low-spending and high-spending).

<img width="1285" height="349" alt="image" src="https://github.com/user-attachments/assets/61626330-12c9-4974-879d-3e6a2ab00133" />

Correlation Heatmap

<img width="975" height="763" alt="image" src="https://github.com/user-attachments/assets/4ccf8405-e608-4b64-a8c6-2c443d845a45" />

What is a Correlation Heatmap?

A correlation heatmap uses the Pearson Correlation Coefficient ($r$) to measure the strength of a linear relationship between two variables. The values range from -1 to +1.

•	+1.0: Perfect positive correlation (as one goes up, the other goes up).

•	0.0: No linear relationship.

•	-1.0: Perfect negative correlation (as one goes up, the other goes down).

• Tenure vs. Churn (Negative Correlation): Usually around -0.35. This is one of the most important findings. It means as tenure (length of stay) increases, the likelihood of Churn decreases. Long-term customers are loyal.

• Monthly Charges vs. Churn (Positive Correlation): Usually around +0.19. This shows that as the monthly bill increases, the probability of churn also increases. Customers paying more are more sensitive to the service value.

• Total Charges vs. Tenure (High Positive Correlation): Usually around +0.83. This is expected (collinearity). The longer a customer stays, the more they have paid in total.

Why is this important:

• Feature Selection: It helps you identify "Red Flags." If a feature has a very low correlation with Churn (near 0), it might not be very useful for the model.

• Multicollinearity: If two input features are too highly correlated (like tenure and TotalCharges), it can sometimes confuse linear models like Logistic Regression. This is why we use a StandardScaler to help normalize these relationships.





