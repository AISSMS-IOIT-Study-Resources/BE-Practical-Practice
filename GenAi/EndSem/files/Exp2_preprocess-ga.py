import pandas as pd
import numpy as np 
from sklearn.preprocessing import MinMaxScaler

def load_sample_data():
    data = {
        'image_id': ['img1', 'img2', 'img3', 'img4', 'img5'],
        'category': ['cat', 'dog', np.nan, 'cat', 'dog'],
        'size_kb': [1200, 850, 900, np.nan, 1150],
        'resolution': ['1024x768', '800x600', '1024x768', '640x480', np.nan],
        'label': ['outdoor', 'indoor', 'indoor', np.nan, 'outdoor']
    }

    df = pd.DataFrame(data)
    print(df)
    return df

def handle_missing_data(df):
    df['category'] = df['category'].fillna('unknown')
    df['label'] = df['label'].fillna('unknown')

    median_size = df['size_kb'].median()
    df['size_kb'] = df['size_kb'].fillna(median_size)

    mode_res = df['resolution'].mode()[0]
    df['resolution'] = df['resolution'].fillna(mode_res)

    return df

# def normalize_numerical(df, cols):
#     for col in cols:
#         min_val = df[col].min()
#         max_val = df[col].max()
#         df[col] = (df[col] - min_val) / (max_val - min_val)
#     return df

def normalize_numerical_sklearn(df, cols):
    scaler = MinMaxScaler()
    df_normalized = df.copy()
    for col in cols:
        df_normalized[col] = scaler.fit_transform(df[[col]]).flatten()
    return df_normalized

def encode_categorical(df, cols):
    df_encoded = pd.get_dummies(df, columns=cols, drop_first=True)
    return df_encoded

def main():
    df = load_sample_data()
    print("Original Data:")
    print(df,"\n")

    df = handle_missing_data(df)
    print("After Handling Missing Data:")
    print(df,"\n")

    df = normalize_numerical_sklearn(df, cols=['size_kb'])
    print("After Normalizing Numerical Features:")
    print(df,"\n")

    df_encoded = encode_categorical(df, cols=['category','resolution','label'])
    print("After Encoding Categorical Variables:")
    print(df_encoded)

main()