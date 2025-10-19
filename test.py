import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import CountVectorizer
from datetime import datetime
from pathlib import Path
import os
from ml25.P01_customer_purchases.boilerplate.negative_generation import gen_all_negatives

DATA_DIR = Path(__file__).resolve().parent

COLOR_MAP = {
    "imgb": "blue", "imgbl": "black", "imgg": "green",
    "imgp": "pink", "imgw": "white", "imgo": "orange",
    "imgy": "yellow", "imgr": "red"
}
ADJ_LIST = [
    "premium", "casual", "modern", "stylish", "exclusive",
    "elegant", "classic", "lightweight", "durable"
]

def read_csv(filename: str):
    file = os.path.join(DATA_DIR, f"{filename}.csv")
    return pd.read_csv(file)

def save_df(df, filename: str):
    file = os.path.join(DATA_DIR, filename)
    df.to_csv(file, index=False)

def extract_customer_features(df):
    df = df.copy()
    today = datetime.strptime("2025-21-09", "%Y-%d-%m")
    for col in ["customer_date_of_birth", "customer_signup_date"]:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    df["customer_age_years"] = ((today - df["customer_date_of_birth"]).dt.days / 365).fillna(0).astype(int)
    df["customer_tenure_months"] = ((today - df["customer_signup_date"]).dt.days / 30.44).fillna(0).astype(int)
    df["customer_id"] = df["customer_id"].astype(str).str.replace("CUST_", "", regex=False)
    group = df.groupby("customer_id")
    cust_feat = pd.DataFrame({
        "customer_id": group["customer_id"].first(),
        "customer_age_years": group["customer_age_years"].first(),
        "customer_tenure_months": group["customer_tenure_months"].first(),
        "customer_gender": group["customer_gender"].first() if "customer_gender" in df.columns else "unknown",
    }).reset_index(drop=True)
    save_df(cust_feat, "customer_features.csv")
    return cust_feat

def preprocess(df, training=False, preprocessor=None, cat_names=None, text_names=None):
    df = df.copy()
    if "customer_gender" not in df.columns:
        df["customer_gender"] = "unknown"
    if "item_img_filename" in df.columns:
        df["item_img_filename"] = df["item_img_filename"].map(COLOR_MAP).fillna("unknown")
    if "item_title" in df.columns:
        df["title_features"] = df["item_title"].apply(lambda x: " ".join([w for w in str(x).lower().split() if w in ADJ_LIST]))
        df.drop(columns=["item_title"], inplace=True)
    else:
        df["title_features"] = ""
    if "customer_prefered_cat" in df.columns:
        df["customer_cat_is_prefered"] = df["item_category"] == df["customer_prefered_cat"]
    dropcols = [
        "purchase_id", "purchase_timestamp", "customer_date_of_birth",
        "customer_signup_date", "customer_item_views", "purchase_item_rating",
        "purchase_device", "item_release_date", "item_avg_rating",
        "item_num_ratings", "item_id"
    ]
    df.drop(columns=[c for c in dropcols if c in df.columns], inplace=True, errors="ignore")
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    df["customer_id"] = df["customer_id"].astype(str).str.replace("CUST_", "", regex=False)
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna("unknown")
        else:
            df[col] = df[col].fillna(df[col].median())
    numerical_features = ["item_price", "customer_age_years", "customer_tenure_months"]
    categorical_features = ["customer_gender", "item_category", "item_img_filename"]
    free_text_features = ["title_features"]
    if training:
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", StandardScaler(), numerical_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
                ("text", CountVectorizer(), "title_features"),
            ],
            remainder="passthrough"
        )
        processed = preprocessor.fit_transform(df)
        cat_names = preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
        text_names = [f"title_{t}" for t in preprocessor.named_transformers_["text"].get_feature_names_out()]
    else:
        processed = preprocessor.transform(df)
    output_columns = [f"num_{col}" for col in numerical_features]
    if cat_names is not None:
        output_columns.extend(cat_names)
    if text_names is not None:
        output_columns.extend(text_names)
    passthrough_cols = [c for c in df.columns if c not in numerical_features + categorical_features + free_text_features]
    output_columns.extend(passthrough_cols)
    processed_df = pd.DataFrame(processed.toarray() if hasattr(processed, "toarray") else processed, columns=output_columns)
    return processed_df, preprocessor, cat_names, text_names

def read_train_data():
    train_df = read_csv("customer_purchases_train")
    customer_feat = extract_customer_features(train_df)
    item_cols = ["item_id", "item_title", "item_category", "item_price", "item_img_filename"]
    item_info = train_df[item_cols].drop_duplicates(subset="item_id")
    all_neg = gen_all_negatives(train_df)
    all_neg = pd.merge(all_neg, item_info, on="item_id", how="left")
    n_pos = len(train_df)
    all_neg = all_neg.sample(n=min(4 * n_pos, len(all_neg)), random_state=42)
    all_neg.drop_duplicates(subset=["customer_id", "item_id"], inplace=True)
    train_df["customer_id"] = train_df["customer_id"].astype(str).str.replace("CUST_", "", regex=False)
    train_df_cust = pd.merge(train_df, customer_feat, on="customer_id", how="left")
    if "customer_gender_x" in train_df_cust.columns or "customer_gender_y" in train_df_cust.columns:
        train_df_cust["customer_gender"] = train_df_cust.get("customer_gender_x", train_df_cust.get("customer_gender_y", "unknown"))
        train_df_cust.drop(columns=["customer_gender_x", "customer_gender_y"], inplace=True, errors="ignore")
    processed_pos, preprocessor, cat_names, text_names = preprocess(train_df_cust, training=True)
    processed_pos["label"] = 1
    all_neg["customer_id"] = all_neg["customer_id"].astype(str).str.replace("CUST_", "", regex=False)
    processed_neg = pd.merge(all_neg, customer_feat, on="customer_id", how="left")
    if "customer_gender_x" in processed_neg.columns or "customer_gender_y" in processed_neg.columns:
        processed_neg["customer_gender"] = processed_neg.get("customer_gender_x", processed_neg.get("customer_gender_y", "unknown"))
        processed_neg.drop(columns=["customer_gender_x", "customer_gender_y"], inplace=True, errors="ignore")
    processed_neg, _, _, _ = preprocess(processed_neg, training=False, preprocessor=preprocessor, cat_names=cat_names, text_names=text_names)
    processed_neg["label"] = 0
    processed_full = pd.concat([processed_pos, processed_neg], axis=0).sample(frac=1, random_state=42)
    save_df(processed_full, "train_final.csv")
    print({"balance": processed_full["label"].value_counts().to_dict()})
    return preprocessor, cat_names, text_names

def read_test_data(preprocessor, cat_names, text_names):
    test_df = read_csv("customer_purchases_test")
    customer_feat = read_csv("customer_features")
    test_df["customer_id"] = test_df["customer_id"].astype(str).str.replace("CUST_", "", regex=False)
    customer_feat["customer_id"] = customer_feat["customer_id"].astype(str)
    merged = pd.merge(test_df, customer_feat, on="customer_id", how="left")
    if "customer_gender_x" in merged.columns or "customer_gender_y" in merged.columns:
        merged["customer_gender"] = merged.get("customer_gender_x", merged.get("customer_gender_y", "unknown"))
        merged.drop(columns=["customer_gender_x", "customer_gender_y"], inplace=True, errors="ignore")
    processed, _, _, _ = preprocess(merged, training=False, preprocessor=preprocessor, cat_names=cat_names, text_names=text_names)
    save_df(processed, "test_final.csv")

if __name__ == "__main__":
    preprocessor, cat_names, text_names = read_train_data()
    read_test_data(preprocessor, cat_names, text_names)
