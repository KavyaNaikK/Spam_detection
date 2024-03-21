
import os
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as me

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw')
nltk.download('omw-1.4')


def load_training_email_data(folder):
    emails = []
    labels = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r', encoding='latin1') as file:
            content = file.read()
            emails.append(content)
            labels.append(1 if folder == 'spam' else 0)
    return emails, labels


def preprocess_text(text):
    # Remove non-alphanumeric characters and HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Convert to lowercase
    text = text.lower() 
    
    # Tokenize text
    tokens = word_tokenize(text)
     
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    # print('------------->',lemmatized_tokens)
    
    
    # Join tokens back into text
    preprocessed_text = ' '.join(lemmatized_tokens)
    # print(preprocessed_text)
    
    return preprocessed_text


def build_XGBClassifier(X, y):
        
        xgb_classifier = XGBClassifier()
        xgb_classifier.fit(X, y)

        return xgb_classifier

def train_model(emails, labels):
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(emails)
    y = np.array(labels)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train Naive Bayes classifier
    clf = build_XGBClassifier(X_train, y_train)

    # Evaluate the model
    y_test_pred = clf.predict(X_test)
    accuracy = me.accuracy_score(y_test, y_test_pred)
    print("Accuracy:", accuracy)

    return vectorizer, clf


def load_prediction_email_data(folder):
    emails = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r', encoding='latin1') as file:
            content = file.read()
            emails.append(content)
    return emails

if __name__ == '__main__':

    ham_emails, ham_labels = load_training_email_data('ham')
    spam_emails, spam_labels = load_training_email_data('spam')
    emails = ham_emails + spam_emails
    labels = ham_labels + spam_labels

    preprocessed_emails = [preprocess_text(email) for email in emails]
    vectorizer, clf = train_model(preprocessed_emails, labels)

   
    new_emails = load_prediction_email_data('Prediction_data')
    i = 0
    for new_email in new_emails:
        i+=1
        preprocessed_new_email = preprocess_text(new_email)
        X_pred = vectorizer.transform([preprocessed_new_email])
        prediction = clf.predict(X_pred)
        if prediction == 0:
            print(f'The email at position {i} is Not Spam')
        else:
            print(f'The email at position {i} is a Spam')
        

    print("Precision:", me.precision_score)
    print("Recall:", me.recall_score)
    print("Confusion Matrix:", me.confusion_matrix)
    print("Accuracy:", me.accuracy_score)
    print("F1 Score:", me.f1_score)





import os
import re
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as me

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw')
nltk.download('omw-1.4')

# Function to load email data from a folder
def load_training_email_data(folder):
    emails = []
    labels = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r', encoding='latin1') as file:
            content = file.read()
            emails.append(content)
            # Assign label 1 for spam, 0 for ham
            labels.append(1 if folder == 'spam' else 0)
    return emails, labels

# Function to preprocess text data
def preprocess_text(text):
    # Remove non-alphanumeric characters and HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    # Convert to lowercase
    text = text.lower() 
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Lemmatize tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into text
    preprocessed_text = ' '.join(lemmatized_tokens)
    
    return preprocessed_text

# Function to build XGBoost classifier
def build_XGBClassifier(X, y):
    xgb_classifier = XGBClassifier()
    xgb_classifier.fit(X, y)
    return xgb_classifier

# Function to train the model
def train_model(emails, labels):
    # Convert text data into numerical features using CountVectorizer
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(emails)
    y = np.array(labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train XGBoost classifier
    clf = build_XGBClassifier(X_train, y_train)

    # Evaluate the model
    y_test_pred = clf.predict(X_test)
    accuracy = me.accuracy_score(y_test, y_test_pred)
    print("Accuracy:", accuracy)

    return vectorizer, clf

# Function to load email data for prediction
def load_prediction_email_data(folder):
    emails = []
    for filename in os.listdir(folder):
        with open(os.path.join(folder, filename), 'r', encoding='latin1') as file:
            content = file.read()
            emails.append(content)
    return emails

if __name__ == '__main__':
    # Load training email data
    ham_emails, ham_labels = load_training_email_data('ham')
    spam_emails, spam_labels = load_training_email_data('spam')
    emails = ham_emails + spam_emails
    labels = ham_labels + spam_labels

    # Preprocess the email data
    preprocessed_emails = [preprocess_text(email) for email in emails]

    # Train the model
    vectorizer, clf = train_model(preprocessed_emails, labels)

    # Load prediction email data
    new_emails = load_prediction_email_data('Prediction_data')

    # Predict labels for new emails and print the results
    # Taking the emails one after the other from the prediction_data folder, to predict if they are spam or ham.
    i = 0
    for new_email in new_emails:
        i += 1
        preprocessed_new_email = preprocess_text(new_email)
        X_pred = vectorizer.transform([preprocessed_new_email])
        prediction = clf.predict(X_pred)
        if prediction == 0:
            print(f'The email at position {i} is Not Spam')
        else:
            print(f'The email at position {i} is a Spam')

    exit()

