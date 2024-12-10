import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Step 1: Load the dataset
file_path = "/mnt/data/crop.csv"
df = pd.read_csv('crop.csv')

# Step 2: Encode the target variable (Crop names to numeric)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])

# Step 3: Feature scaling
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = df[features]
y = df['label']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4: Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Define a RandomForest model and hyperparameter tuning
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(
    estimator=rf_model, param_grid=param_grid, cv=3, scoring='accuracy', verbose=1, n_jobs=-1
)

# Step 6: Fit the model using GridSearchCV
print("\nTuning hyperparameters. Please wait...")
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Step 7: Train the best model on the full training data
best_model.fit(X_train, y_train)

# Step 8: Evaluate the model on the test data
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nOptimized Model Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Step 9: Save the best model, scaler, and label encoder
joblib.dump(best_model, 'optimized_crop_recommendation_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("\nOptimized model, label encoder, and scaler saved successfully.")

# Step 10: User Interaction for Predictions
def predict_crop():
    print("\nEnter the soil and environmental characteristics:")
    try:
        N = float(input("Nitrogen content (N): "))
        P = float(input("Phosphorus content (P): "))
        K = float(input("Potassium content (K): "))
        temperature = float(input("Temperature (°C): "))
        humidity = float(input("Humidity (%): "))
        ph = float(input("pH level: "))
        rainfall = float(input("Rainfall (mm): "))

        # Create input sample and scale it
        new_sample = [[N, P, K, temperature, humidity, ph, rainfall]]
        new_sample_scaled = scaler.transform(new_sample)

        # Predict the crop
        predicted_crop = label_encoder.inverse_transform(best_model.predict(new_sample_scaled))
        print(f"\nRecommended Crop: {predicted_crop[0]}")

    except ValueError:
        print("Invalid input. Please enter numeric values.")

# Step 11: Run the interactive prediction function
while True:
    predict_crop()
    again = input("\nWould you like to predict another crop? (yes/no): ").strip().lower()
    if again != 'yes':
        print("\nThank you for using the Crop Recommendation System. Goodbye!")
        break