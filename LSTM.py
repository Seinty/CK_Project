from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import accuracy_score, classification_report,f1_score
import pandas as pd
import pickle


data = pd.read_csv('./files/Cleaned_Data.csv')
data = data.iloc[:,1:]

tokenizer = Tokenizer(num_words=8000)
tokenizer.fit_on_texts(data.review)
sequences = tokenizer.texts_to_sequences(data.review)
dat = pad_sequences(sequences, maxlen=150)

X_train, X_test, y_train, y_test = train_test_split(dat, data['sentiment'], test_size=0.2, random_state=42)

# Пример LSTM модели
model = Sequential()
model.add(Embedding(input_dim=8000, output_dim=100, input_length=150))
model.add(LSTM(100, return_sequences = True))
model.add(LSTM(50))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

prediction = model.predict(X_test)

y_pred = [1 if pred > 0.5 else 0 for pred in prediction]

acc = accuracy_score(y_test,y_pred)
print(acc)

f1 = f1_score(y_test, y_pred)
print(f'F1 Score: {f1:.2f}')

# Отчет по классификации
print('Classification Report:')
print(classification_report(y_test, y_pred))
