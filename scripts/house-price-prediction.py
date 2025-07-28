# %% [markdown]
# 📌 Step 1: Import Libraries

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib

# %% [markdown]
# 📌 Step 2: Load Dataset

# %%
# Load California housing data
cali = fetch_california_housing()
X = pd.DataFrame(cali.data, columns=cali.feature_names)
y = pd.Series(cali.target, name="MedianHouseValue")

# Save to CSV for portability
df = pd.concat([X, y], axis=1)
df.to_csv("../data/cali_housing.csv", index=False)

print(df.head())

# %% [markdown]
# 📌 Step 3: Exploratory Data Analysis (EDA)

# %%
# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# Target distribution
sns.histplot(df["MedianHouseValue"], bins=50, kde=True)
plt.title("House Value Distribution")
plt.show()

# %% [markdown]
# 📌 Step 4: Preprocessing

# %%
X = df.drop("MedianHouseValue", axis=1)
y = df["MedianHouseValue"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# %% [markdown]
# 📌 Step 5: Train-Test Split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# %% [markdown]
# 📌 Step 6: Train the Model

# %%
model = LinearRegression()
model.fit(X_train, y_train)

# %% [markdown]
# 📌 Step 7: Evaluate the Model

# %%
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R^2 Score: {r2:.2f}")

# Scatter plot
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Value")
plt.ylabel("Predicted Value")
plt.title("Actual vs Predicted Values")
plt.savefig("../outputs/results.png")
plt.show()

# %% [markdown]
# 📌 Step 8: Save the Model

# %%
joblib.dump(model, "../outputs/model.pkl")

# %% [markdown]
# 📌 Step 9: requirements.txt

# %% [markdown]
# pandas  
# numpy  
# matplotlib  
# seaborn  
# scikit-learn  
# joblib  


