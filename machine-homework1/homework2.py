import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt


def load_data(data_folder):
    train_data = pd.read_csv(f"{data_folder}/train1_icu_data.csv")
    train_labels = pd.read_csv(f"{data_folder}/train1_icu_label.csv", header=None)
    test_data = pd.read_csv(f"{data_folder}/test1_icu_data.csv")
    test_labels = pd.read_csv(f"{data_folder}/test1_icu_label.csv", header=None)

    return train_data, train_labels, test_data, test_labels


train_data, train_labels, test_data, test_labels = load_data("data1")
feature_names = train_data.columns
def preprocess_data(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data), scaler

train_data, scaler = preprocess_data(train_data)
test_data = scaler.transform(test_data)
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model
# print(train_labels.values.flatten()[1:20])
model = train_model(train_data, train_labels.values.flatten()[1:])
def calculate_errors(model, X_train, y_train):
    train_error = 1 - model.score(X_train, y_train)
    cv_error = np.mean(cross_val_score(model, X_train, y_train, cv=5))
    return train_error, cv_error

train_error, cv_error = calculate_errors(model, train_data, train_labels.values.flatten()[1:])
print(f"Training Error: {train_error}")
print(f"Cross-validation Error: {(1-cv_error)}")
def test_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    test_error = 1 - accuracy_score(y_test, predictions)
    return test_error

test_error = test_model(model, test_data, test_labels.values.flatten()[1:])
print(f"Test Error: {test_error}")


def plot_roc_curve(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


plot_roc_curve(model, test_data, test_labels.values.flatten()[1:].astype(int))


def plot_roc_curve(model, X_test, y_test):
    y_probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


plot_roc_curve(model, test_data, test_labels.values.flatten()[1:].astype(int))

def analyze_feature_importance(model, feature_names, output_file_path):
    coefficients = model.coef_[0]
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': coefficients
    })
    # 按系数的绝对值降序排列
    feature_importance = feature_importance.sort_values(by='Coefficient', ascending=False)
    # 保存到 CSV 文件
    feature_importance.to_csv(output_file_path, index=False)
    print(f"Feature importance has been saved to {output_file_path}")

# 调用函数并传入输出文件路径
csv_file_path = "feature_importance.csv"  # 您可以根据需要修改文件路径和名称
analyze_feature_importance(model, feature_names, csv_file_path)
