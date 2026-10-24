"""
Lab 7: Naive Bayes Classification
Dataset: Iris (scikit-learn built-in) - Gaussian Naive Bayes
Also: SMS-style spam text sample - Multinomial Naive Bayes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

# ---------------------------------------------------------------
# Part A: Gaussian Naive Bayes on the Iris dataset
# ---------------------------------------------------------------
iris = load_iris(as_frame=True)
X, y = iris.data, iris.target
print("Dataset shape:", X.shape)
print("Classes:", [str(c) for c in iris.target_names])
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y)
print("\nTrain size:", len(X_train), " Test size:", len(X_test))

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nClass priors:", np.round(model.class_prior_, 3))
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

ConfusionMatrixDisplay(cm, display_labels=iris.target_names).plot(cmap="Blues")
plt.title("Gaussian NB - Iris Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, facecolor="white")
plt.close()

sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=X.columns)
print("\nPrediction for", sample.values[0].tolist(), "->",
      iris.target_names[model.predict(sample)[0]])
print("Probabilities:", np.round(model.predict_proba(sample)[0], 4))

# ---------------------------------------------------------------
# Part B: Multinomial Naive Bayes for text (spam / ham)
# ---------------------------------------------------------------
texts = [
    "win a free prize now", "free entry claim your reward", "urgent cash offer click now",
    "congratulations you won a lottery", "get cheap loans instantly", "free gift voucher limited offer",
    "meeting at 10 am tomorrow", "please submit the lab report", "lunch at the canteen today",
    "assignment deadline is friday", "call me when you reach home", "project discussion with the team",
    "claim free cash prize today", "see you in class tomorrow", "viva scheduled for monday morning",
    "exclusive offer win money now",
]
labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]  # 1 = spam, 0 = ham

vec = CountVectorizer()
Xt = vec.fit_transform(texts)
Xt_tr, Xt_te, yt_tr, yt_te = train_test_split(
    Xt, labels, test_size=0.25, random_state=1, stratify=labels)

mnb = MultinomialNB(alpha=1.0)
mnb.fit(Xt_tr, yt_tr)
print("\n--- Text classification (Multinomial NB) ---")
print("Test accuracy:", accuracy_score(yt_te, mnb.predict(Xt_te)))

for msg in ["free cash prize offer", "submit the report before the viva"]:
    p = mnb.predict(vec.transform([msg]))[0]
    pr = mnb.predict_proba(vec.transform([msg]))[0]
    print(f"'{msg}' -> {'SPAM' if p else 'HAM'}  (P(ham)={pr[0]:.3f}, P(spam)={pr[1]:.3f})")
