import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # Importing the Support Vector Classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# Split the data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# --- Random Forest Classifier ---
rf_model = RandomForestClassifier()  # Random Forest model
rf_model.fit(x_train, y_train)  # Train the model
rf_y_predict = rf_model.predict(x_test)  # Predict on the test data
rf_score = accuracy_score(rf_y_predict, y_test)  # Calculate accuracy
print('Random Forest Classifier Accuracy: {:.2f}%'.format(rf_score * 100))

# --- Support Vector Classifier (SVC) ---
svc_model = SVC()  # Support Vector Classifier model
svc_model.fit(x_train, y_train)  # Train the model
svc_y_predict = svc_model.predict(x_test)  # Predict on the test data
svc_score = accuracy_score(svc_y_predict, y_test)  # Calculate accuracy
print('Support Vector Classifier Accuracy: {:.2f}%'.format(svc_score * 100))

# Save both models
with open('models.p', 'wb') as f:
    pickle.dump({'rf_model': rf_model, 'svc_model': svc_model}, f)
