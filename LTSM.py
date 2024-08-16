import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.nn.utils.rnn import pad_sequence
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.utils import get_tokenizer
import plotly.express as px

data = pd.read_csv('IMDB Dataset.csv')

def clean_text(
        text: str
) -> str:
    """
    Очищает текст от HTML тегов, специальных символов и приводит его к нижнему регистру.
    Параметры:
    text (str): Исходный текст.
    Возвращает:
    str: Очищенный текст.
    """
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = text.lower()
    return text

data['cleaned_review'] = data['review'].apply(clean_text)
data['sentiment'] = data['sentiment'].apply(lambda x: 1 if x == 'positive' else 0)

tokenizer = get_tokenizer('basic_english')
data['tokenized_review'] = data['cleaned_review'].apply(tokenizer)

X_train, X_test, y_train, y_test = train_test_split(data['tokenized_review'], data['sentiment'], test_size=0.2, random_state=42)

vocab = build_vocab_from_iterator(X_train, specials=["<pad>"])
vocab.set_default_index(vocab["<pad>"])

def text_to_tensor(
        text : str,
        vocab
):
    """
    Преобразует токенизированный текст в тензор индексов на основе словаря.
    Параметры:
    text (list[str]): Токенизированный текст.
    vocab (Vocab): Объект словаря для преобразования слов в индексы.
    Возвращает:
    torch.Tensor: Тензор индексов.
    """
    return torch.tensor(vocab(text), dtype=torch.long)

X_train_tensor = [text_to_tensor(text, vocab) for text in X_train]
X_test_tensor = [text_to_tensor(text, vocab) for text in X_test]

# Паддинг последовательностей
X_train_tensor = pad_sequence(X_train_tensor, batch_first=True, padding_value=vocab["<pad>"])
X_test_tensor = pad_sequence(X_test_tensor, batch_first=True, padding_value=vocab["<pad>"])

y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

class ReviewDataset(Dataset):
    """
    Класс, представляющий датасет для отзывов.
    Параметры:
    X (torch.Tensor): Матрица признаков (TF-IDF).
    y (torch.Tensor): Метки классов.
    Методы:
    __len__: Возвращает количество образцов в датасете.
    __getitem__: Возвращает образец и его метку по индексу.
    """
    def __init__(self, X, y)->None:
        """
        Инициализирует объект ReviewDataset.
        Параметры:
        X (torch.Tensor): Матрица признаков.
        y (torch.Tensor): Метки классов.
        """
        self.X = X
        self.y = y

    def __len__(self)->int:
        """
        Возвращает количество образцов в датасете.
        Возвращает:
        int: Количество образцов.
        """
        return len(self.y)

    def __getitem__(self,
                    idx : int
    )->tuple:
        """
        Возвращает образец и его метку по индексу.
        Параметры:
        idx (int): Индекс образца.
        Возвращает:
        tuple: Образец (torch.Tensor) и его метка (torch.Tensor).
        """
        return self.X[idx], self.y[idx]

train_dataset = ReviewDataset(X_train_tensor, y_train_tensor)
test_dataset = ReviewDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class LSTMSentimentModel(nn.Module):
    """
    Нейронная сеть на основе LSTM для анализа настроений.
    Атрибуты:
    embedding (nn.Embedding): Слой эмбеддингов для преобразования индексов токенов в вектора.
    lstm (nn.LSTM): LSTM слой для обработки последовательностей.
    fc (nn.Linear): Полносвязный слой для классификации.
    sigmoid (nn.Sigmoid): Сигмоидальная функция для получения вероятности.
    Методы:
    forward: Прямой проход через сеть.
    """
    def __init__(self,
                 vocab_size : int,
                 embed_dim : int,
                 hidden_dim : int,
                 output_dim : int
    )->None:
        """
        Инициализирует модель LSTMSentimentModel.
        Параметры:
        vocab_size (int): Размер словаря (количество уникальных токенов).
        embed_dim (int): Размерность эмбеддингового слоя.
        hidden_dim (int): Размерность скрытого слоя LSTM.
        output_dim (int): Размерность выходного слоя (1 для бинарной классификации).
        """
        super(LSTMSentimentModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self,
                x : torch.Tensor
    )->torch.Tensor:
        """
        Прямой проход через нейронную сеть.
        Параметры:
        x (torch.Tensor): Входной тензор (последовательность индексов токенов).
        Возвращает:
        torch.Tensor: Вероятность положительного класса.
        """
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        out = self.fc(hidden[-1])
        return self.sigmoid(out)

vocab_size = len(vocab)
embed_dim = 128
hidden_dim = 128
output_dim = 1

model = LSTMSentimentModel(vocab_size, embed_dim, hidden_dim, output_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}')

model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        predicted = (outputs.squeeze() > 0.5).float()
        y_pred.extend(predicted.tolist())
        y_true.extend(labels.tolist())

accuracy = accuracy_score(y_true, y_pred)
print(f'Accuracy: {accuracy}')
print(classification_report(y_true, y_pred))

conf_matrix = confusion_matrix(y_true, y_pred)
fig = px.imshow(conf_matrix, text_auto=True, color_continuous_scale='Blues', title='Confusion Matrix')
fig.show()

