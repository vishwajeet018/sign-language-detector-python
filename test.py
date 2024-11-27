import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# --- Step 1: Load Data ---
data_dict = pickle.load(open('./data.pickle', 'rb'))
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])

# --- Step 2: Pre-process Data ---
# Reduce noise further to keep accuracy high
noise = np.random.normal(0, 0.05, data.shape)  # Even lower noise
data_noisy = data + noise

# Normalize the data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_noisy)

# Split data into train and test sets
x_train, x_test, y_train, y_test = train_test_split(data_scaled, labels, test_size=0.2, shuffle=True, stratify=labels)

# --- Step 3: Train Models ---

# 1. Random Forest with increased complexity
rf_model = RandomForestClassifier(n_estimators=500, max_depth=30, random_state=42, class_weight='balanced')
rf_model.fit(x_train, y_train)

# 2. Support Vector Classifier (SVC) with high C and optimized gamma
svc_model = SVC(kernel='rbf', C=100, gamma=0.01, random_state=42)
svc_model.fit(x_train, y_train)

# --- Step 4: Evaluate the Models ---
# Random Forest predictions
rf_y_predict = rf_model.predict(x_test)

# Support Vector Classifier predictions
svc_y_predict = svc_model.predict(x_test)

# Calculate Accuracy
rf_accuracy = accuracy_score(y_test, rf_y_predict)
svc_accuracy = accuracy_score(y_test, svc_y_predict)

print(f'Random Forest Accuracy: {rf_accuracy * 100:.2f}%')
print(f'Support Vector Classifier Accuracy: {svc_accuracy * 100:.2f}%')

# Display Classification Reports
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_y_predict))

print("\nSupport Vector Classifier Classification Report:")
print(classification_report(y_test, svc_y_predict))

# --- Step 5: Plot Confusion Matrix ---
# Plot Confusion Matrix for Random Forest
rf_conf_matrix = confusion_matrix(y_test, rf_y_predict)
plt.figure(figsize=(8, 8))
plt.imshow(rf_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Random Forest Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(labels)))
plt.xticks(tick_marks, np.unique(labels))
plt.yticks(tick_marks, np.unique(labels))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Annotating the confusion matrix values on the graph
thresh = rf_conf_matrix.max() / 2.
for i in range(len(np.unique(labels))):
    for j in range(len(np.unique(labels))):
        plt.text(j, i, f'{rf_conf_matrix[i, j]}', horizontalalignment="center",
                 color="white" if rf_conf_matrix[i, j] > thresh else "black")
plt.show()

# Plot Confusion Matrix for Support Vector Classifier
svc_conf_matrix = confusion_matrix(y_test, svc_y_predict)
plt.figure(figsize=(8, 8))
plt.imshow(svc_conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Support Vector Classifier Confusion Matrix')
plt.colorbar()
plt.xticks(tick_marks, np.unique(labels))
plt.yticks(tick_marks, np.unique(labels))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Annotating the confusion matrix values on the graph
thresh = svc_conf_matrix.max() / 2.
for i in range(len(np.unique(labels))):
    for j in range(len(np.unique(labels))):
        plt.text(j, i, f'{svc_conf_matrix[i, j]}', horizontalalignment="center",
                 color="white" if svc_conf_matrix[i, j] > thresh else "black")
plt.show()

# --- Step 6: Simulate Epoch Training and Plot Accuracy Graph ---
# Placeholder for epoch training accuracy for both models
rf_train_accuracies = []
svc_train_accuracies = []
rf_test_accuracies = []
svc_test_accuracies = []

# Simulate epoch-like training by incrementally training the models with subsets of data
for epoch in range(1, 11):  # Simulate 10 epochs
    rf_model.fit(x_train[:epoch * int(len(x_train) / 10)], y_train[:epoch * int(len(y_train) / 10)])
    svc_model.fit(x_train[:epoch * int(len(x_train) / 10)], y_train[:epoch * int(len(y_train) / 10)])

    rf_train_accuracy = accuracy_score(y_train[:epoch * int(len(y_train) / 10)], rf_model.predict(x_train[:epoch * int(len(x_train) / 10)]))
    svc_train_accuracy = accuracy_score(y_train[:epoch * int(len(y_train) / 10)], svc_model.predict(x_train[:epoch * int(len(x_train) / 10)]))

    rf_test_accuracy = accuracy_score(y_test, rf_model.predict(x_test))
    svc_test_accuracy = accuracy_score(y_test, svc_model.predict(x_test))

    rf_train_accuracies.append(rf_train_accuracy)
    svc_train_accuracies.append(svc_train_accuracy)
    rf_test_accuracies.append(rf_test_accuracy)
    svc_test_accuracies.append(svc_test_accuracy)

# Plot Accuracy Graph for training and testing over "epochs"
plt.figure(figsize=(6, 4))
plt.plot(range(1, 11), rf_train_accuracies, label='Random Forest Training Accuracy', marker='o')
plt.plot(range(1, 11), rf_test_accuracies, label='Random Forest Test Accuracy', marker='o')
plt.plot(range(1, 11), svc_train_accuracies, label='SVC Training Accuracy', marker='x')
plt.plot(range(1, 11), svc_test_accuracies, label='SVC Test Accuracy', marker='x')

plt.title('Training and Test Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
