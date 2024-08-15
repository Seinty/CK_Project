import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os

data = pd.read_csv('../IMDB Dataset.csv')

# Загрузка стоп-слов
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Лемматизация текста
def lemmatize_text(text):
    return ' '.join([lemmatizer.lemmatize(word) for word in text.split()])

def clean_text(
        text : str
)->str:
    """
    Очищает текст от HTML тегов, специальных символов и приводит его к нижнему регистру.
    Параметры:
    text (str): Исходный текст.
    Возвращает:
    str: Очищенный текст.
    """
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = re.sub(r'(.)\1+', r'\1', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    text = lemmatize_text(text)
    return text

data["review"] = data['review'].apply(clean_text)
data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

X_train, X_test, y_train, y_test = train_test_split(data['review'], data['sentiment'], test_size=0.2, random_state=42)

# Разделяю текст на униграммы\биграммы
tfidf = TfidfVectorizer(ngram_range=(1,3),max_features=8000)
X_train_tfidf = tfidf.fit_transform(X_train).toarray()
X_test_tfidf = tfidf.transform(X_test).toarray()

train_data = {
    'features': X_train_tfidf,
    'labels': y_train
}

test_data = {
    'features': X_test_tfidf,
    'labels': y_test
}


os.makedirs("./files",exist_ok=True)

# Сохранение тренировочных данных в pickle
with open('./files/train_data.pkl', 'wb') as f:
    pickle.dump(train_data, f)

# Сохранение тестовых данных в pickle
with open('./files/test_data.pkl', 'wb') as f:
    pickle.dump(test_data, f)
