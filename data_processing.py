import requests
import pandas as pd
from sklearn.impute import SimpleImputer

# API endpoint
crop_api = "http://localhost:8000/crops"

# Function to fetch data from the API
def fetchdata(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        return None

# Fetch data
crops_data = fetchdata(crop_api)

if crops_data:
    # Create DataFrame
    crops_df = pd.DataFrame(crops_data)
    
    # Data overview
    print("Dataset Overview:")
    print(crops_df.describe())  # Summary statistics
    print("\nShape of the DataFrame:", crops_df.shape)  # Shape of the data
    print("\nData Types and Non-Null Counts:")
    crops_df.info()  # Data types and non-null counts
    
    # Check for null values
    print("\nNull Value Summary:")
    print(crops_df.isnull().sum())
    
    # Impute missing numeric data using the mean
    numeric_cols = crops_df.select_dtypes(include='number').columns
    imputer = SimpleImputer(strategy='mean')
    crops_df[numeric_cols] = imputer.fit_transform(crops_df[numeric_cols])
    
    # Remove duplicates
    crops_df = crops_df.drop_duplicates()
    
    # Change data types to numeric if applicable
    for col in crops_df.columns:
        if pd.api.types.is_object_dtype(crops_df[col]) and crops_df[col].str.isnumeric().all():
            crops_df[col] = pd.to_numeric(crops_df[col])
    
    # Save cleaned data
    crops_df.to_csv("cleaned_crops_data.csv", index=False)
    print("\nData cleaned and saved to 'cleaned_crops_data.csv'.")
else:
    print("No data fetched, exiting.")





