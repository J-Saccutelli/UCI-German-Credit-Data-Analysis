import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

#Figure Display 
sb.set_theme(style="whitegrid")


#Read in data 
data_frame = pd.read_csv('German_Credit_Data.csv')


#Check for null or duplicated entries
data_frame.isna().sum().sum()
data_frame.duplicated().sum()


#Describe the data:
data_frame.describe().T

data_frame.describe(include="object").T



#Display some important data:
print(data_frame.columns)
data_frame[["Age (years)", "Credit Amount", "Duration"]].hist(bins=20, edgecolor= "black", figsize=(10,5))
plt.suptitle("Distribution of Numerical Features", fontsize=14)
plt.show()

plt.figure(figsize=(15,5))
for r, c in enumerate(["Age (years)", "Credit Amount", "Duration"]):
    plt.subplot(1, 3, r + 1)
    sb.boxplot(y = data_frame[c], color="Red")
    plt.title(c)
plt.tight_layout()
plt.show()

#Categorial data columns:
plt.figure(figsize=(15,5))
cat_c = ["Personal status and sex", "Job", "Housing", "Savings account/bonds", "Status of existing checking account", "Purpose"]
for r, c in enumerate(cat_c):
    plt.subplot(3, 3, r+1)
    sb.countplot(data=data_frame, x = c, palette="Set2", order=data_frame[c].value_counts().index)
    plt.title("Distribution of {col}")
    plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


#Correlation q1 visualized:
correlation = data_frame[["Age (years)", "Job", "Credit Amount", "Duration"]]
sb.pairplot(correlation, hue='Job', palette='dark')
plt.show()

#Differences in the data based on key factor visualized:
plt.figure(figsize=(15,8))
sb.scatterplot(data=data_frame, x="Age (years)", y="Credit Amount", hue="Personal status and sex")
plt.title("Credit Amount vs Age in Years by Sex and personal status")
plt.show()

#Differences in the credit data based on key factor visualized:
sb.violinplot(data=data_frame, x="Savings account/bonds", y = "Credit Amount")
plt.title("Credit Amount Distribution by Savings account/bonds")
plt.show()

#Differences in the Risk by Age visualized:
plt.figure(figsize=(15,4))
for r, c in enumerate(["Age (years)", "Credit Amount", "Duration"]):
    plt.subplot(1, 3, r + 1)
    sb.boxplot(data=data_frame, x="Target (1 = Good, 2 = Bad)", y=c, hue=data_frame["Job"])
    plt.title(f"{c} by Risk")
plt.tight_layout()
plt.show()

#Label Encoder:

#Target column name
target_col = "Target (1 = Good, 2 = Bad)"

le_cat = data_frame.copy()

#Get all categorical feature columns (strictly excluding target)
le_cat_col = [c for c in le_cat.select_dtypes(include="object").columns if c != target_col]

#Label encode feature columns correctly
le_cat_c_dict = {}
for c in le_cat_col:
    le = LabelEncoder()
    le_cat[c] = le.fit_transform(le_cat[c])
    # Store encoder in dict, do NOT overwrite le_cat[c]
    le_cat_c_dict[c] = le  

#Label encode target column
le_t = LabelEncoder()
le_cat[target_col] = le_t.fit_transform(le_cat[target_col])
le_cat_c_dict[target_col] = le_t

#Define feature matrix X and target vector y AFTER all encoding is done
X = le_cat.drop(columns=[target_col])
y = le_cat[target_col]

#Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=1)

#Model Training Function
def train_m(m, param_g, X_train, y_train, X_test, y_test):
    grid = GridSearchCV(m, param_g, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)
    best_m = grid.best_estimator_
    y_pred = best_m.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    # Return best parameters instead of repeating estimator
    return best_m, acc, grid.best_params_  
#Run Decision Tree
dt = DecisionTreeClassifier(random_state=1, class_weight="balanced")
dt_param_g = {
    "max_depth": [3, 5, 7, 10, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4]
}

best_dt, acc_dt, best_params_dt = train_m(dt, dt_param_g, X_train, y_train, X_test, y_test)
print("Decision tree acc: ", acc_dt)
print("Decision tree best param: ", best_params_dt)

#Random Forest:
rf = RandomForestClassifier(random_state=1, class_weight = "balanced", n_jobs = -1)
rf_param_g = {"n_estimators": [100, 200], "max_depth": [5, 7, 10, None], "min_samples_split": [2,5,10], "min_samples_leaf": [1,2,4]}
best_rf, acc_rf, param_rf = train_m(rf, rf_param_g, X_train, y_train, X_test, y_test)
print("Decision tree acc: ", acc_rf)
print("Decision tree best param: ", param_rf)

#Extra Trees Classifier:

extra_t = ExtraTreesClassifier(random_state=1, class_weight="balanced", n_jobs=-1)
extra_t_param_g = {
"n_estimators": [100, 200], "max_depth": [5, 7, 10, None], "min_samples_split": [2,5,10], "min_samples_leaf": [1,2,4]
}

best_extra_t, acc_extra_t, param_extra_t = train_m(extra_t, extra_t_param_g, X_train, y_train, X_test, y_test)
print("Extra trees acc: ", acc_extra_t)
print("Extra trees best param: ", best_extra_t)

