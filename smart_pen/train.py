import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# ==========================================
# 0. HELPER FUNCTION FOR ACADEMIC PLOTS
# ==========================================
def plot_learning_curve(estimator, title, X, y, cv=5):
    """
    Generates a plot comparing Training Score vs Cross-Validation Score.
    Crucial for detecting Overfitting vs Underfitting.
    """
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.grid()
    
    # Plot the bands (Standard Deviation)
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    
    # Plot the lines
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    
    plt.legend(loc="best")
    filename = title.replace(" ", "_") + ".png"
    plt.savefig(filename)
    print(f"   [Saved Plot] {filename}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, classes, title):
    """
    Saves a heatmap of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    filename = title.replace(" ", "_") + ".png"
    plt.savefig(filename)
    print(f"   [Saved Plot] {filename}")
    plt.close()

def train_and_evaluate(X, y, model_name):
    print(f"\n{'='*40}")
    print(f" TRAINING MODEL: {model_name}")
    print(f"{'='*40}")

    # 1. Split Data (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Initialize SVM (RBF Kernel is standard for Sensor Data)
    # C=10 gives a stricter boundary, gamma='scale' auto-adjusts to data variance
    svm = SVC(kernel='rbf', C=10, gamma='scale')

  
    print("Running 5-Fold Cross-Validation...")
    cv_scores = cross_val_score(svm, X_train, y_train, cv=5)
    print(f"   > Mean CV Accuracy: {cv_scores.mean() * 100:.2f}% (+/- {cv_scores.std() * 100:.2f}%)")

    # 4. Final Training
    svm.fit(X_train, y_train)

    # 5. Prediction on Test Set
    y_pred = svm.predict(X_test)
    
    # 6. Accuracy Score
    test_acc = accuracy_score(y_test, y_pred)
    print(f"   > Test Set Accuracy: {test_acc * 100:.2f}%")

    # 7. Detailed Classification Report (Precision, Recall, F1)
    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_test, y_pred))

    # 8. Plot Confusion Matrix
    unique_classes = np.unique(y) # Get names like 'circle', 'square'
    plot_confusion_matrix(y_test, y_pred, unique_classes, f"Confusion Matrix - {model_name}")

    # 9. Plot Learning Curve (Checks for Overfitting)

    print("Generating Learning Curve...")
    plot_learning_curve(svm, f"Learning Curve - {model_name}", X_train, y_train)

    # 10. Save Model
    filename = f"model_{model_name.lower()}.pkl"
    with open(filename, 'wb') as f: pickle.dump(svm, f)
    print(f"Model saved as {filename}")

# MAIN EXECUTION

if __name__ == "__main__":
    # Load Data
    print("Loading preprocessed data...")
    try:
        with open('X_data.pkl', 'rb') as f: X = pickle.load(f)
        with open('y_shape.pkl', 'rb') as f: y_shape = pickle.load(f)
        with open('y_quality.pkl', 'rb') as f: y_quality = pickle.load(f)
    except FileNotFoundError:
        print("Error: .pkl files not found! Run the preprocessing code first.")
        exit()

    # Train Model A: SHAPE
    train_and_evaluate(X, y_shape, "Shape_Detector")

    # Train Model B: QUALITY
    train_and_evaluate(X, y_quality, "Quality_Detector")

    print("\nAll tasks finished. Check your folder for the PNG images and PKL models.")