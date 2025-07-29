from data.balanced_data_splitter import quick_balanced_split
import pandas as pd

# Load your dataset
data = pd.read_csv('mbpp100.csv')
X = data.drop(columns=['target'])
y = data['target']

# Create balanced splits
X_train, X_val, y_train, y_val = quick_balanced_split(
    X, y, 
    validation_size=0.2,
    verbose=True
)

# Use splits for model training and validation
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier()
model.fit(X_train, y_train)
val_score = model.score(X_val, y_val)
print(f"Validation accuracy: {val_score:.4f}")