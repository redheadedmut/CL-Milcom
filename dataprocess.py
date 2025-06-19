# Initial data processing
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# Global label encoder instance
label_encoder = LabelEncoder()

def get_label_encoder():
    """
    Returns the global label encoder instance.
    """
    return label_encoder

def limit_class_samples(data, max_class):
    # Group by label and sample at most `max_class` samples per class
    return data.groupby('label').apply(lambda x: x.sample(min(len(x), max_class))).reset_index(drop=True)

def upsample_classes(data, min_class):
    majority = data.groupby('label').filter(lambda x: len(x) >= min_class)
    minority = data.groupby('label').filter(lambda x: len(x) < min_class)
    
    # Upsample each minority class
    minority_upsampled = (
        minority.groupby('label')
        .apply(lambda x: resample(x, replace=True, n_samples=min_class, random_state=42))
        .reset_index(drop=True)
    )
    
    # Combine the majority and upsampled minority classes
    return pd.concat([majority, minority_upsampled]).reset_index(drop=True)

def preprocess_data(data, max_class, min_class):
    """
    Preprocess the data by limiting samples per class and upsampling minority classes.
    Returns the processed DataFrame, the number of classes and features, and the label encoder.
    """
    # Limit the dataset to max_class samples per class
    df_limited = limit_class_samples(data, max_class)

    # Upsample classes below the min_class threshold
    df = upsample_classes(df_limited, min_class)

    # Use the global label encoder
    global label_encoder
    df["protocol"] = label_encoder.fit_transform(df["protocol"])
    df["label"] = label_encoder.fit_transform(df["label"])

    num_classes = df['label'].nunique()
    num_features = df.drop(columns=['label']).shape[1]
    print(f"Number of classes: {num_classes}, Number of features: {num_features}")

    return df, num_classes, num_features

def split_data(df, batch_size=100):
    """
    Split the preprocessed DataFrame into train, validation, and test sets.
    Returns DataLoaders for each set.
    """
    # Split the data into features (X) and labels (y)
    X = df.drop(columns=['label'])
    y = df['label']

    # Split into training (80%) and temporary (20%) sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Split the temporary set into validation (50% of 20% = 10%) and test (50% of 20% = 10%) sets
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Print dataset sizes
    print(f"Training set size: {X_train.shape}, Validation set size: {X_val.shape}, Test set size: {X_test.shape}")

    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)

    X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create DataLoaders for batching
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def process_data(data, max_class, min_class, batch_size=100):
    """
    Complete data processing pipeline that combines preprocessing and splitting.
    Returns both the processed DataFrame, the DataLoaders, and the label encoder.
    """
    # Preprocess the data
    df, num_classes, num_features, encoder = preprocess_data(data, max_class, min_class)
    
    # Split the data into train/test sets
    train_loader, val_loader, test_loader = split_data(df, batch_size)
    
    return df, train_loader, val_loader, test_loader, num_classes, num_features, encoder


