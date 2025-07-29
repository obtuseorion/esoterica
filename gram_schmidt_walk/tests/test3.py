import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from data.balanced_data_splitter import BalancedDataSplitter

# Load the dataset
data = pd.read_csv('mbpp100.csv')

# Extract features from the text descriptions
vectorizer = TfidfVectorizer(max_features=100)
X_features = vectorizer.fit_transform(data['text']).toarray()

# Create balanced splitter
splitter = BalancedDataSplitter(
    validation_size=0.2,
    random_state=42,
    balance_features=True, 
    balance_statistics=True,
    balance_labels=False  # Since we don't have labels
)

# Get indices for balanced splits
train_indices, val_indices = splitter.split_indices(X_features, verbose=True)

# Get the actual data rows for both splits
train_data = data.iloc[train_indices]
val_data = data.iloc[val_indices]

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")

# Display sample from each split
print("\nSample from training set:")
print(train_data[['task_id', 'text']].head(2))

print("\nSample from validation set:")
print(val_data[['task_id', 'text']].head(2))

# Instead of using compare_with_random, just create a simple visualization
# of how the data is distributed in each split

# Let's examine some characteristics of the text data
import matplotlib.pyplot as plt

# 1. Text length distribution
train_text_lengths = train_data['text'].apply(len)
val_text_lengths = val_data['text'].apply(len)

plt.figure(figsize=(10, 6))
plt.hist(train_text_lengths, alpha=0.5, label='Training set')
plt.hist(val_text_lengths, alpha=0.5, label='Validation set')
plt.xlabel('Text Length (characters)')
plt.ylabel('Count')
plt.title('Distribution of Text Lengths')
plt.legend()
plt.savefig('text_length_distribution.png')
plt.close()

# 2. Code complexity (measured by length)
train_code_lengths = train_data['code'].apply(len)
val_code_lengths = val_data['code'].apply(len)

plt.figure(figsize=(10, 6))
plt.hist(train_code_lengths, alpha=0.5, label='Training set')
plt.hist(val_code_lengths, alpha=0.5, label='Validation set')
plt.xlabel('Code Length (characters)')
plt.ylabel('Count')
plt.title('Distribution of Code Lengths')
plt.legend()
plt.savefig('code_length_distribution.png')
plt.close()

print("\nCreated visualizations for text and code length distributions:")
print("- text_length_distribution.png")
print("- code_length_distribution.png")