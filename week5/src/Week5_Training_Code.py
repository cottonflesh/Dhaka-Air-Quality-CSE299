import os
import pandas as pd, numpy as np, joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "Dhaka_Air_Quality_Dates_Fixed_2012_2024.csv"
OUT_DIR = "Week5_Dhaka_Air_Quality"
os.makedirs(OUT_DIR, exist_ok=True)
df = pd.read_csv(DATA_PATH)
df["Date_parsed"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
df["Time_parsed"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors="coerce")
df["Month"] = df["Date_parsed"].dt.month; df["Day"] = df["Date_parsed"].dt.day
df["Hour"] = df["Time_parsed"].dt.hour; df["DayOfWeek"] = df["Date_parsed"].dt.dayofweek
target = "PM2.5"
num = ["PM10", "Temperature", "Humidity", "Wind Speed", "Wind Direction", "CO", "NO₂", "SO₂", "O₃", "Rainfall", "Month", "Day", "Hour", "DayOfWeek"]
cat = ["Location"]
for c in [target] + num: df[c] = pd.to_numeric(df[c], errors="coerce")
df["Location"] = df["Location"].fillna("Unknown").astype(str).str.strip()
work = df.dropna(subset=[target, "Date_parsed"]).sort_values("Date_parsed")
train_all = work[work.Date_parsed < pd.Timestamp("2024-01-01")]; test = work[work.Date_parsed >= pd.Timestamp("2024-01-01")]
if len(test) < 100 or len(train_all) < 100:
    i = int(len(work)*0.8); train_all, test = work.iloc[:i], work.iloc[i:]
train = train_all.sample(n=35000, random_state=42).sort_values("Date_parsed") if len(train_all)>35000 else train_all.copy()
features = num + cat; Xtr, ytr = train[features], train[target]; Xte, yte = test[features], test[target]
pre = ColumnTransformer([("num", SimpleImputer(strategy="median"), num), ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat)])
model = Pipeline([("preprocess", pre), ("model", RandomForestRegressor(n_estimators=35, random_state=42, n_jobs=2, min_samples_leaf=3, max_features=0.8))])
model.fit(Xtr, ytr); pred = model.predict(Xte); base = np.full(len(yte), ytr.mean())
def metrics(name, y, p): return {"Model": name, "Test rows": len(y), "MAE": mean_absolute_error(y,p), "RMSE": np.sqrt(mean_squared_error(y,p)), "R²": r2_score(y,p)}
pd.DataFrame([metrics("Baseline (training mean)",yte,base), metrics("Random Forest Regressor",yte,pred)]).to_csv(f"{OUT_DIR}/Week5_Model_Evaluation.csv", index=False)
joblib.dump(model, f"{OUT_DIR}/Week5_Random_Forest_PM25_Model.joblib")
print("Done. This experiment predicts PM2.5, not AQI, because the dataset has no AQI target.")
