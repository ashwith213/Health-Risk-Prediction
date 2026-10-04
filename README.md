Health Risk Prediction using Supervised Machine Learning
📌 Project Overview
This project focuses on predicting health risk using Supervised Machine Learning techniques.
The objective is to develop a classification model that can classify individuals into healthy and unhealthy categories based on health, lifestyle, physiological, and medical-history indicators.
The project is based on a dataset containing health records of 9,800 individuals. 

🎯 Problem Statement
A biomedical research institute, NovaGen Research Labs, conducts large-scale population health studies to understand how underlying health conditions influence disease risk and long-term health outcomes.
The available health data contains numerical and categorical indicators such as physiological measurements, lifestyle factors, and medical history.
The goal of this project is to build a Machine Learning classification model that can distinguish between healthy and unhealthy individuals based on the available health data. 
The prediction can support:

Selection of eligible participants for clinical trials
Longitudinal health studies
Population stratification
Risk-based analysis
Comparison of health outcomes 


📊 Dataset
The dataset contains 9,800 individual health records.
Each record represents a unique participant, with a combination of numerical and categorical health indicators. 
Features



Feature
Description




Age
Age of the individual in years


BMI
Body Mass Index


Blood_Pressure
Systolic blood pressure


Cholesterol
Cholesterol level


Glucose_Level
Blood glucose level


Heart_Rate
Resting heart rate


Sleep_Hours
Average sleep hours per day


Exercise_Hours
Average exercise hours per day


Water_Intake
Daily water intake


Stress_Level
Stress level


Smoking
Smoking habit


Alcohol
Alcohol consumption


Diet
General diet category


MentalHealth
Mental health score/indicator


PhysicalActivity
Overall physical activity level


MedicalHistory
Previous medical conditions


Allergies
Known allergies


Diet_Type__Vegan
Vegan diet indicator


Diet_Type__Vegetarian
Vegetarian diet indicator


Blood_Group_AB
Blood group AB indicator


Blood_Group_B
Blood group B indicator


Blood_Group_O
Blood group O indicator


Target
Health outcome/risk target



The feature descriptions are based on the provided project statement. 

🧠 Machine Learning Approach
The project follows a supervised classification workflow:
Dataset
   ↓
Data Preprocessing
   ↓
Feature Selection
   ↓
Train/Test Split
   ↓
Model Training
   ↓
Hyperparameter Experimentation
   ↓
Model Prediction
   ↓
Model Evaluation


🤖 Models Used
1. Decision Tree Classifier
A Decision Tree classifier was implemented to establish a tree-based classification approach.
Different hyperparameters were experimented with, including:

max_depth
min_samples_split

2. Random Forest Classifier
A Random Forest classifier was implemented using multiple decision trees to improve classification performance.
The model used:
RandomForestClassifier(
    n_estimators=301
)

The model was trained using the training dataset and evaluated using the testing dataset.

📈 Random Forest Performance
The Random Forest model achieved the following results on the test dataset:



Metric
Score




Accuracy
94.45%


Recall
95.23%


F1-score — Class 0
94%


F1-score — Class 1
95%


Macro Average F1
94%


Weighted Average F1
94%


Test Samples
2,865



Classification Report
              precision    recall    f1-score    support

0                 0.95      0.94        0.94       1356
1                 0.94      0.95        0.95       1509

accuracy                              0.94       2865
macro avg          0.94      0.94        0.94       2865
weighted avg       0.94      0.94        0.94       2865

The Random Forest achieved approximately 94.45% test accuracy and 95.23% recall.

🛠️ Technologies Used

Python
Jupyter Notebook
Pandas
NumPy
Scikit-learn
Matplotlib
Seaborn


📂 Project Structure
health-risk-prediction-ml/
│
├── README.md
├── health_risk_prediction.ipynb
├── requirements.txt
├── LICENSE
├── .gitignore
│
└── images/

If the dataset is permitted to be shared publicly, it can also be included as:
├── novagen_dataset.csv


▶️ How to Run the Project
1. Clone the repository
git clone https://github.com/YOUR_USERNAME/health-risk-prediction-ml.git

2. Navigate to the project directory
cd health-risk-prediction-ml

3. Install the required libraries
pip install -r requirements.txt

4. Open the Jupyter Notebook
jupyter notebook health_risk_prediction.ipynb

5. Run all cells
Make sure the dataset is available at the path expected by the notebook before running the model.

📌 Key Learning Outcomes
Through this project, I gained practical experience in:

Supervised Machine Learning
Binary classification
Data preprocessing
Train/test splitting
Decision Tree algorithms
Random Forest algorithms
Hyperparameter experimentation
Model evaluation
Accuracy and recall
Classification reports
Feature-based health-risk prediction


🚀 Future Improvements
Possible future improvements include:

Testing additional classification algorithms
Performing systematic hyperparameter tuning
Using cross-validation
Handling class imbalance if required
Improving feature engineering
Comparing multiple models using consistent test metrics
Deploying the final model as a web application
Developing an interactive health-risk prediction interface


⚠️ Disclaimer
This project is developed for educational and Machine Learning practice purposes.
The predictions generated by this model should not be considered medical advice, diagnosis, or a substitute for professional medical evaluation.

👨‍💻 Author
Ashwith Reddy
B.Tech Student
Interested in Machine Learning, Embedded Systems, VLSI and AI

⭐ Project Highlights

🌲 Random Forest Classifier
🎯 94.45% Test Accuracy
📊 95.23% Recall
👥 2,865 Test Samples
🐍 Python + Scikit-learn

If you like this project, consider giving the repository a ⭐ on GitHub.
Available next action: 
