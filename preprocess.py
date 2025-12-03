import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

# LOADING THE DATA
paths = [
    "HMS_2016-2017.csv",
    "HMS_2017-2018.csv",
    "HMS_2018-2019.csv",
    "HMS_2019-2020.csv",
    "HMS_2020-2021.csv",
    "HMS_2021-2022.csv",
    "HMS_2022-2023.csv",
    "HMS_2023-2024.csv",
    "HMS_2024-2025.csv"
]
dfs = []
for p in paths:
    try:
        dfs.append(pd.read_csv(p, low_memory=False))
        print(f"Loaded: {p}")
    except Exception as e:
        print(f"Error loading {p}: {e}")

df = pd.concat(dfs, ignore_index=True)

print("Number of Survey Responses (# of Rows): ", len(df))

df["dx_dep"] = df["dx_dep"].fillna(0).astype(int)
y = df["dx_dep"].to_numpy()
Xdf = df.drop(columns=["dx_dep"])

# PREPROCESSING THE DATA

# convert mixed-type columns (i.e., converting "object" to numeric when possible)
for col in Xdf.columns:
    if Xdf[col].dtype == "object":
        try:
            Xdf[col] = pd.to_numeric(Xdf[col])
        except:
            pass

numeric_cols = Xdf.select_dtypes(include=["number"]).columns.tolist()
object_cols = Xdf.select_dtypes(include=["object"]).columns.tolist()

# only encode small categorical columns and drop high-cardinality object columns
small_cat_cols = [c for c in object_cols if Xdf[c].nunique() <= 15]
large_cat_cols = [c for c in object_cols if c not in small_cat_cols]
Xdf = Xdf.drop(columns=large_cat_cols)

# one-hot encode small categorical columns
if len(small_cat_cols) > 0:
    ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
    X_cat = ohe.fit_transform(Xdf[small_cat_cols])
    cat_feature_names = ohe.get_feature_names_out(small_cat_cols)
else:
    X_cat = np.empty((len(Xdf), 0))
    cat_feature_names = []

X_num = Xdf[numeric_cols].to_numpy()

# final feature matrix - combining numeric and encoded categorical features
X = np.hstack([X_num, X_cat])
feature_names = numeric_cols + list(cat_feature_names)

# replace all NaN with 0
X = np.nan_to_num(X, nan=0.0)

# save artifacts
np.save("X.npy", X)
np.save("y.npy", y)
np.save("feature_names.npy", np.array(feature_names, dtype=object))

print("Preprocessing complete.")
print("Saved: X.npy, y.npy, feature_names.npy")