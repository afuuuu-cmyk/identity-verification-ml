
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Face ID Classification System", layout="wide")
st.title("🔍 Facial Recognition Identity Classification")

# Load Data
@st.cache_data
def load_data():
    url = "https://hebbkx1anhila5yf.public.blob.vercel-storage.com/lfw_arnie_nonarnie%281%29%20%281%29-Dl4miBqO1gGI8nj6UMN44bSrXc5tsy.csv"
    return pd.read_csv(url)

df = load_data()
st.success("✅ Data Loaded Successfully!")

# Sidebar for EDA options
with st.sidebar:
    st.header("🔎 Exploration Panel")
    show_data = st.checkbox("Show Raw Data")
    show_distribution = st.checkbox("Show Label Distribution")
    show_pca = st.checkbox("Show PCA Visualization")
    run_model = st.button("🚀 Train & Evaluate Models")

if show_data:
    st.subheader("📄 Raw Dataset Preview")
    st.dataframe(df.head())

if show_distribution:
    st.subheader("🟧 Label Distribution")
    fig1, ax1 = plt.subplots()
    df['Label'].value_counts().plot(kind='bar', color=['skyblue', 'orange'], ax=ax1)
    st.pyplot(fig1)

if show_pca:
    st.subheader("🧬 PCA Visualization (2D)")
    pca_viz = PCA(n_components=2)
    X_pca = pca_viz.fit_transform(df.iloc[:, :-1])
    fig2, ax2 = plt.subplots()
    scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=df['Label'], cmap='viridis', alpha=0.6)
    fig2.colorbar(scatter, ax=ax2)
    st.pyplot(fig2)

# Preprocessing
X = df.iloc[:, :-1].values
y = df['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if run_model:
    st.subheader("⚙️ Model Training & Evaluation")

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'SVM': SVC(probability=True, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
        'Neural Network': MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=(100, 50))
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        results[name] = {'model': model, 'accuracy': accuracy, 'auc': auc}

    best_model_name = max(results, key=lambda k: results[k]['accuracy'])
    best_model = results[best_model_name]['model']

    st.success(f"🏆 Best Model: {best_model_name} | Accuracy: {results[best_model_name]['accuracy']:.4f}")

    # ROC Curve
    st.subheader("📈 ROC Curve of Best Model")
    fpr, tpr, _ = roc_curve(y_test, best_model.predict_proba(X_test_scaled)[:, 1])
    fig3, ax3 = plt.subplots()
    ax3.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, best_model.predict_proba(X_test_scaled)[:, 1]):.3f}")
    ax3.plot([0, 1], [0, 1], 'k--')
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('ROC Curve')
    ax3.legend()
    st.pyplot(fig3)

    # Classification Report
    st.subheader("🧾 Classification Report")
    y_pred_best = best_model.predict(X_test_scaled)
    st.text(classification_report(y_test, y_pred_best))

    # Confusion Matrix
    st.subheader("🧮 Confusion Matrix")
    fig4, ax4 = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred_best), annot=True, fmt='d', cmap='Blues', ax=ax4)
    st.pyplot(fig4)
