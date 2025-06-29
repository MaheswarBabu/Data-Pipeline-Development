import pandas as pd
import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Enable logging to track the steps
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def load_data(filepath):
    # Load the CSV file into a pandas DataFrame
    logging.info(f"Loading data from {filepath}")
    return pd.read_csv(filepath)

def build_pipeline():
    # Define numeric and categorical columns
    numeric_cols = ['math score', 'reading score', 'writing score']
    categorical_cols = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']

    # Process numeric columns: fill missing and scale
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Process categorical columns: fill missing and one-hot encode
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combine both pipelines
    full_pipeline = ColumnTransformer([
        ('num', num_pipeline, numeric_cols),
        ('cat', cat_pipeline, categorical_cols)
    ])

    return full_pipeline, numeric_cols, categorical_cols

def run_pipeline(input_file, output_file):
    # Load data
    df = load_data(input_file)

    # Build pipeline
    pipeline, numeric_cols, categorical_cols = build_pipeline()

    # Transform the data
    logging.info("Processing the data...")
    processed = pipeline.fit_transform(df)

    # Get column names for final DataFrame
    cat_names = pipeline.named_transformers_['cat']['encoder'].get_feature_names_out(categorical_cols)
    columns = numeric_cols + list(cat_names)

    # Create cleaned DataFrame
    cleaned_df = pd.DataFrame(processed, columns=columns)

    # Save to new CSV
    cleaned_df.to_csv(output_file, index=False)
    logging.info(f"Cleaned data saved to {output_file}")

# Run when the script is executed
if __name__ == "__main__":
    run_pipeline("StudentsPerformance.csv", "StudentsPerformance_cleaned.csv")