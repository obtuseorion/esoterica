import pandas as pd
import numpy as np
from data.balanced_data_splitter import BalancedDataSplitter

# Load the dataset
data = pd.read_csv('mbpp100.csv')

# In this case, let's consider 'text' as our feature and 'code' as our target
# We'll use text embeddings as features

# First we need to convert text to numerical features
from sklearn.feature_extraction.text import TfidfVectorizer

# Extract features from the text descriptions
vectorizer = TfidfVectorizer(max_features=100)  # Limit features for simplicity
X = vectorizer.fit_transform(data['text']).toarray()

# For this example, we're not using 'code' as a target since it's complex text
# Instead, we'll just create balanced splits of the data itself

# Create balanced splitter
splitter = BalancedDataSplitter(
    validation_size=0.2,
    random_state=42,
    balance_features=True,
    balance_statistics=True,
    processor_kwargs={
        'handle_categorical': False,  # No categorical features in our vector representation
        'scaling': 'standard'
    }
)

# Create balanced splits - without a target
X_train_indices, X_val_indices = splitter.split(X, verbose=True)

# Get the actual data rows for both splits
train_data = data.iloc[X_train_indices]
val_data = data.iloc[X_val_indices]

print(f"Training set size: {len(train_data)}")
print(f"Validation set size: {len(val_data)}")

# You could also evaluate how well distributed your splits are
comparison = splitter.compare_with_random(X)
splitter.visualize_comparison(comparison, title="MBPP Dataset Split Comparison")