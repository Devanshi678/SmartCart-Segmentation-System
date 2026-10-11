import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import joblib

def load_and_clean_data(filepath="smartcart_customers.csv"):
    df = pd.read_csv(filepath)
    
    # 1. Impute missing income and drop extreme outliers
    df["Income"] = df["Income"].fillna(df["Income"].median())
    df = df[df["Income"] < 200000]
    
    # 2. Feature Engineering
    df["Age"] = 2026 - df["Year_Birth"]
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], dayfirst=True)
    ref_date = df["Dt_Customer"].max()
    df["cutomer_tenure_days"] = (ref_date - df["Dt_Customer"]).dt.days
    
    df["Total_spending"] = (
        df["MntWines"] + df["MntFruits"] + df["MntMeatProducts"] + 
        df["MntFishProducts"] + df["MntSweetProducts"] + df["MntGoldProds"]
    )
    df["Total_childrens"] = df["Kidhome"] + df["Teenhome"]
    
    df["Education"] = df["Education"].replace({
        "Basic": "Undergraduate", "2n Cycle": "Undergraduate",
        "Graduation": "Graduate",
        "PhD": "Post-Graduate", "Master": "Post-Graduate"
    })
    
    df["Living_with"] = df["Marital_Status"].replace({
        "Married": "Couple", "Together": "Couple",
        "Divorced": "Single", "Widow": "Single", "Alone": "Single", 
        "Absurd": "Single", "YOLO": "Single"
    })
    
    # Target modeling features
    selected_cols = [
        "Age", "Income", "Total_spending", "Total_childrens", 
        "cutomer_tenure_days", "NumWebPurchases", "NumCatalogPurchases", 
        "NumStorePurchases", "NumWebVisitsMonth", "Education", "Living_with"
    ]
    
    return df[selected_cols].copy().reset_index(drop=True)


def build_and_save_model():
    # Load clean dataset
    df = load_and_clean_data("smartcart_customers.csv")
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=["Education", "Living_with"], drop_first=False)
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_encoded)
    
    # Fit Agglomerative Clustering
    clustering_model = AgglomerativeClustering(n_clusters=4, linkage="ward")
    cluster_labels = clustering_model.fit_predict(scaled_features)
    
    # Optimize KNN classifier via GridSearchCV across k = 1 to 15
    param_grid = {"n_neighbors": list(range(1, 16))}
    grid_search = GridSearchCV(
        estimator=KNeighborsClassifier(),
        param_grid=param_grid,
        cv=5,
        scoring="accuracy"
    )
    grid_search.fit(scaled_features, cluster_labels)
    
    best_classifier = grid_search.best_estimator_
    best_k = grid_search.best_params_["n_neighbors"]
    
    # Attach assigned clusters to dataset
    df["Cluster"] = cluster_labels
    
    # Save artifacts
    joblib.dump(scaler, "scaler.joblib")
    joblib.dump(best_classifier, "classifier.joblib")
    joblib.dump(list(df_encoded.columns), "model_columns.joblib")
    df.to_csv("smartcart_segmented_customers.csv", index=False)
    
    print("✅ Model build complete!")
    print(f"Optimal k found by GridSearchCV: {best_k}")
    print("Saved: scaler.joblib, classifier.joblib, model_columns.joblib, smartcart_segmented_customers.csv")

if __name__ == "__main__":
    build_and_save_model()