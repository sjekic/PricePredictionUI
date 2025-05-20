import joblib

model = joblib.load("xgb_best_model.joblib")
feature_names = model.get_booster().feature_names

print(f"Number of features: {len(feature_names)}")
print(feature_names)