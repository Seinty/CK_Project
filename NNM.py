import pandas as pd
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import pickle


with open('./files/train_data.pkl', 'rb') as f:
    loaded_train_data = pickle.load(f)

# Загрузка тестовых данных
with open('./files/test_data.pkl', 'rb') as f:
    loaded_test_data = pickle.load(f)

# Извлечение признаков и меток
X_train_tfidf = loaded_train_data['features']
y_train = loaded_train_data['labels']

X_test_tfidf = loaded_test_data['features']
y_test = loaded_test_data['labels']

X_train_tensor = torch.tensor(X_train_tfidf, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_tfidf, dtype=torch.float32)
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

    def __init__(self, X, y) -> None :
        """
        Инициализирует объект ReviewDataset.
        Параметры:
        X (torch.Tensor): Матрица признаков.
        y (torch.Tensor): Метки классов.
        """
        self.X = X
        self.y = y

    def __len__(self) -> int:
        """
        Возвращает количество образцов в датасете.
        Возвращает:
        int: Количество образцов.
        """
        return len(self.y)

    def __getitem__(
            self,
            idx : int
    )->tuple[torch.Tensor, torch.Tensor]:
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

class SentimentModel(nn.Module):
    """
    Нейронная сеть для анализа настроений на основе логистической регрессии.
    Параметры:
    input_dim (int): Размер входного слоя (количество признаков).
    Методы:
    forward: Прямой проход через сеть. Возвращает вероятность положительного класса.
    """

    def __init__(
            self,
            input_dim : int
    ) -> None:
        """
        Инициализирует модель SentimentModel.
        Параметры:
        input_dim (int): Размер входного слоя.
        """
        super(SentimentModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(
            self,
            x : torch.Tensor
    ) ->torch.Tensor:
        """
        Прямой проход через нейронную сеть.
        Параметры:
        x (torch.Tensor): Входной тензор.
        Возвращает:
        torch.Tensor: Вероятность положительного класса.
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

input_dim = X_train_tensor.shape[1]
model = SentimentModel(input_dim)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 15
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
