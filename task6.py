import pandas as pd

# Load data files
def load_data():
    with open("C:/Users/Admin/PycharmProjects/task1/Hygiene/hygiene.dat", "r", encoding="utf-8") as f:
        reviews = f.readlines()
    with open("C:/Users/Admin/PycharmProjects/task1/Hygiene/hygiene.dat.labels", "r", encoding="utf-8") as f:
        labels = f.readlines()
    additional = pd.read_csv("C:/Users/Admin/PycharmProjects/task1/Hygiene/hygiene.dat.additional", header=None)
    return reviews, labels, additional

reviews, labels, additional = load_data()


# Split data
train_reviews = reviews[:546]
train_labels = [int(label.strip()) for label in labels[:546]]
test_reviews = reviews[546:]
test_labels = labels[546:]  # These are all "[None]"

# Split additional features
train_additional = additional.iloc[:546, :]
test_additional = additional.iloc[546:, :]


from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(reviews):
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    text_features = tfidf.fit_transform(reviews)
    return text_features

X_train_text = preprocess_text(train_reviews)
X_test_text = preprocess_text(test_reviews)


from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Process cuisines (Column 0)
cuisines_encoder = OneHotEncoder(handle_unknown="ignore")
train_cuisines = cuisines_encoder.fit_transform(train_additional.iloc[:, 0].values.reshape(-1, 1))
test_cuisines = cuisines_encoder.transform(test_additional.iloc[:, 0].values.reshape(-1, 1))

# Normalize review count and average rating (Columns 2 & 3)
scaler = MinMaxScaler()
train_numeric = scaler.fit_transform(train_additional.iloc[:, 2:])
test_numeric = scaler.transform(test_additional.iloc[:, 2:])

# Combine non-text features
import scipy.sparse as sp
X_train_non_text = sp.hstack([train_cuisines, train_numeric])
X_test_non_text = sp.hstack([test_cuisines, test_numeric])


X_train = sp.hstack([X_train_text, X_train_non_text])
X_test = sp.hstack([X_test_text, X_test_non_text])
y_train = train_labels


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score

# Train the classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Cross-validation (for training data)
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(clf, X_train, y_train, scoring="f1_macro", cv=5)
print("Cross-Validation F1 Score:", cv_scores.mean())

# Predict for the test set
y_pred = clf.predict(X_test)

# Save predictions to a file for submission
nickname = "YourNickname"
output_file = "submission.txt"
with open(output_file, "w") as f:
    f.write(f"{nickname}\n")
    f.writelines([f"{label}\n" for label in y_pred])

print(f"Predictions saved to {output_file}.")


