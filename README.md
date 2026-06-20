# Donor Engagement Prediction using Machine Learning

## Project Overview

This project was developed for NayePankh Foundation to demonstrate how Machine Learning can help organizations make better fundraising and engagement decisions.

The objective is to predict whether an individual is likely to become a donor based on demographic information, membership status, involvement indicators, and historical donation records.

---

## Problem Statement

Non-profit organizations often interact with thousands of individuals but have limited resources for outreach. Predicting potential donors can help focus campaigns on the most promising supporters.

This project uses Machine Learning models to identify donor behavior patterns and support data-driven fundraising decisions.

---

## Dataset

* Total Records: 34,508
* Features: 23
* Target Variable: DONOR_IND

  * Y = Donor
  * N = Non-Donor

Dataset attributes include:

* Age
* Gender
* Marital Status
* Membership Status
* Alumni Indicator
* Parent Indicator
* Email Presence
* Previous Donation History
* Wealth Rating
* Degree Level
* Years of Contribution

---

## Data Preprocessing

The following preprocessing steps were performed:

1. Removed unnecessary columns
2. Converted donation amounts from text to numeric values
3. Handled missing values
4. Encoded categorical variables using one-hot encoding
5. Removed data leakage features
6. Split dataset into training and testing sets

---

## Machine Learning Models Used

### 1. Logistic Regression

A supervised learning algorithm used for binary classification problems.

### 2. Random Forest Classifier

An ensemble learning method that combines multiple decision trees for classification.

---

## Results

| Model               | Accuracy |
| ------------------- | -------- |
| Logistic Regression | 64.71%   |
| Random Forest       | 64.29%   |

Logistic Regression achieved slightly better performance on this dataset.

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* VS Code

---

## How to Run

1. Install required libraries:

pip install pandas scikit-learn matplotlib

2. Run the project:

python main.py

---

## Impact for NayePankh Foundation

This project demonstrates how Machine Learning can help:

* Identify potential donors
* Improve campaign targeting
* Increase donor engagement
* Optimize fundraising efforts
* Support data-driven decision making

---

## Author

Aarya Mule
B.Tech Student
