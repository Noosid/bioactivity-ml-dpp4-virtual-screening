"""
Model training, cross-validation, multi-classifier comparison,
SMOTE-based rebalancing, and evaluation reporting.
"""
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix


def train_random_forest(X_train, y_train, param_grid=None, cv=None, n_jobs=-1):
    """Train a Random Forest classifier with grid search over n_estimators."""
    if param_grid is None:
        param_grid = {'n_estimators': [100, 200, 1000]}
    if cv is None:
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=cv, scoring='accuracy', n_jobs=n_jobs)
    grid_search.fit(X_train, y_train)
    print("Best parameters:", grid_search.best_params_)
    return grid_search.best_estimator_, grid_search


def train_baseline_classifiers(X_train, y_train):
    """
    Train SVM, Gaussian Naive Bayes, and KNN alongside Random Forest for
    comparison, matching the multi-classifier evaluation in the notebook.
    Returns a dict of fitted models keyed by name.
    """
    models = {
        'svm': SVC(),
        'naive_bayes': GaussianNB(),
        'knn': KNeighborsClassifier(),
    }
    fitted = {}
    for name, clf in models.items():
        clf.fit(X_train, y_train)
        fitted[name] = clf
        print(f"Trained {name}")
    return fitted


def evaluate_model(model, X_test, y_test, label="model"):
    """Print a classification report and confusion matrix for a fitted model."""
    y_pred = model.predict(X_test)
    print(f"--- {label} ---")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))
    return y_pred


def cross_validate_roc_auc(model, X, y, cv=None, n_jobs=-1):
    """Report mean ROC-AUC across cross-validation folds."""
    if cv is None:
        cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, n_jobs=n_jobs, scoring='roc_auc')
    print("ROC-AUC per fold:", scores)
    print("Mean ROC-AUC:", scores.mean())
    return scores


def train_with_smote(X_train, y_train, param_grid=None, cv=None, n_jobs=-1):
    """
    Rebalance the training set with SMOTE oversampling, then run the same
    Random Forest grid search used elsewhere. Requires imbalanced-learn.
    """
    from imblearn.over_sampling import SMOTE

    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    print(f"After SMOTE: {X_resampled.shape[0]} samples "
          f"(was {X_train.shape[0]})")
    return train_random_forest(X_resampled, y_resampled, param_grid=param_grid, cv=cv, n_jobs=n_jobs)


def save_model(model, path):
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(path):
    return joblib.load(path)


if __name__ == "__main__":
    print("Modeling module ready.")
