import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD DATASET
# =========================

data = pd.read_csv("donor_dataset.csv")

print("Original Shape:", data.shape)

# =========================
# TARGET VARIABLE
# =========================

data["DONOR_IND"] = data["DONOR_IND"].map({
    "Y": 1,
    "N": 0
})

# =========================
# REMOVE DATA LEAKAGE
# =========================

drop_cols = [
    "ID",
    "BIRTH_DATE",
    "CurrFYGiving",
    "TotalGiving"
]

data = data.drop(columns=drop_cols, errors="ignore")

# =========================
# CLEAN DONATION COLUMNS
# =========================

money_cols = [
    "PrevFYGiving",
    "PrevFY1Giving",
    "PrevFY2Giving",
    "PrevFY3Giving",
    "PrevFY4Giving"
]

for col in money_cols:
    if col in data.columns:
        data[col] = (
            data[col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
        )

        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

# =========================
# HANDLE MISSING VALUES
# =========================

for col in data.columns:

    if pd.api.types.is_numeric_dtype(data[col]):
        data[col] = data[col].fillna(
            data[col].median()
        )
    else:
        data[col] = data[col].fillna(
            "Unknown"
        )

# =========================
# ENCODE CATEGORICAL DATA
# =========================

data = pd.get_dummies(
    data,
    drop_first=True
)

print("Processed Shape:", data.shape)

# =========================
# FEATURES & TARGET
# =========================

X = data.drop("DONOR_IND", axis=1)
y = data["DONOR_IND"]

# =========================
# TRAIN TEST SPLIT
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =========================
# LOGISTIC REGRESSION
# =========================

lr = LogisticRegression(
    max_iter=5000
)

lr.fit(X_train, y_train)

lr_pred = lr.predict(X_test)

print("\n===== Logistic Regression =====")
print(
    "Accuracy:",
    round(
        accuracy_score(y_test, lr_pred) * 100,
        2
    ),
    "%"
)

print(classification_report(
    y_test,
    lr_pred
))

# =========================
# RANDOM FOREST
# =========================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

print("\n===== Random Forest =====")
print(
    "Accuracy:",
    round(
        accuracy_score(y_test, rf_pred) * 100,
        2
    ),
    "%"
)

print(classification_report(
    y_test,
    rf_pred
))
import matplotlib.pyplot as plt

models = ["Logistic Regression", "Random Forest"]
accuracies = [64.71, 64.29]

plt.bar(models, accuracies)

plt.ylabel("Accuracy (%)")
plt.title("Model Comparison")

plt.show()