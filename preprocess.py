import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings("ignore")

FILTER_VARIABLES = [

    # --- Diagnosis variables (except dx_dep target) ---
    "dx_bip","dx_anx","dx_ocd","dx_trauma","dx_neurodev",
    "dx_ea","dx_psy","dx_pers","dx_sa","dx_none","dx_dk","dx_any",

    # Diagnosis detail sub-items
    "dx_dep_1","dx_dep_2","dx_dep_3","dx_dep_4","dx_dep_5",
    *[f"dx_bip_{i}" for i in range(1,6)],
    *[f"dx_ax_{i}" for i in range(1,8)],
    *[f"dx_ocd_{i}" for i in range(1,8)],
    *[f"dx_trauma_{i}" for i in range(1,6)],
    *[f"dx_neurodev_{i}" for i in range(1,6)],
    *[f"dx_ea_{i}" for i in range(1,8)],
    *[f"dx_psy_{i}" for i in range(1,8)],
    *[f"dx_perso_{i}" for i in range(1,13)],
    *[f"dx_sa_{i}" for i in range(1,5)],

    # Medication variables
    *[f"meds_{i}" for i in range(1,10)],
    "meds_dis",
    *[f"meds_w_{i}" for i in range(1,6)],
    *[f"meds_cur_{i}" for i in range(1,9)],
    *[f"meds_time_{i}" for i in range(1,8)],
    "meds_any","meds_help_me","meds_help_others",

    # Stimulant/anti-anxiety reasons
    *[f"stim_reason_{i}" for i in range(1,6)],
    *[f"antianx_reason_{i}" for i in range(1,6)],

    # Misuse variables
    "stim_misuse","stim_sold","stim_prescriber",

    # Summary MH flags
    "anymhprob","dep_any","dep_maj","dep_oth","deprawsc",
    "anx_any","anx_mod","anx_sev","anx_score",
    "ed_any_sde","sib_any","sui_idea",
    "positiveMH","flourish","dep_or_anx",
    "needmet_temp","tx_any","inf_any","meds_any","ther_any",
]

def preprocess_filtered(df):
    """
    Remove only variables that directly reveal a depression diagnosis
    or prior MH treatment. Keep symptom-based predictors.
    """
    cols_to_drop = []

    for col in df.columns:
        # Never drop target
        if col.lower() == "dx_dep":
            continue

        # Remove exact matches
        if col in FILTER_VARIABLES:
            cols_to_drop.append(col)

    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    return df

df = pd.read_csv("HMS_2024-2025.csv")
df = preprocess_filtered(df)

# # LOADING THE DATA
# paths = [
#     "HMS_2022-2023.csv",
#     "HMS_2023-2024.csv",
#     "HMS_2024-2025.csv"
# ]
# dfs = []
# for p in paths:
#     try:
#         dfs.append(pd.read_csv(p, low_memory=False))
#         print(f"Loaded: {p}")
#     except Exception as e:
#         print(f"Error loading {p}: {e}")
# df = pd.concat(dfs, ignore_index=True)

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