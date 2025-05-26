import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('research_sample.csv')

# Preprocess data
data['learning_style'] = data['learning_style'].map({'Visual': 0, 'Auditory': 1, 'Kinesthetic': 2, 'Reading/Writing': 3})
features = data[['academic_score', 'attendance_rate', 'learning_style', 'mutual_score']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Apply K-means clustering
kmeans = KMeans(n_clusters=15, random_state=42)
data['group'] = kmeans.fit_predict(scaled_features)

# Save results
data.to_csv('research_sample.csv', index=False)