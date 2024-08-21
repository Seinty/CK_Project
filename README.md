# Анализ данных и сравнение моделей
____
В этом проекте был проведен подробный анализ данных и оценка производительности трёх различных моделей машинного обучения: метода опорных векторов (SVM), простой нейронной сети и сети с длинной краткосрочной памятью (LSTM). Основная цель проекта — создать модель машинного обучения, которая сможет автоматически классифицировать отзывы по настроениям.

## Обзор моделей
### Метод опорных векторов (SVM)

SVM — это популярная модель с учителем, используемая для задач классификации. Она работает, находя гиперплоскость, которая лучше всего разделяет классы в пространстве признаков. Т.к. SVM особенно эффективен в пространствах с высокой размерностью мы решили использовать именно его в качестве модели классического ML.

### Простая нейронная сеть

Простая полносвязанная нейронная сеть с 2мя скрытыми слоями.

### Сеть с длинной краткосрочной памятью (LSTM)

LSTM — это тип рекуррентной нейронной сети (RNN), которая хорошо работает с данными, в которых важно запоминать какие-либо предыдущие значения, чтобы лучше улавливать контекст.

## Анализ данных и предобработка
Перед обучением моделей был проведён анализ данных для понимания распределения и характеристик данных.
В представленном нам датасете содержались отзывы с метками тональности отзыва. Предварительный анализ данных показал, что дизбаланса классов в нашем случае нет, и мы имеем дело с 25000 экземплярами каждого типа отзывов.
Также мы построили облака слов по каждому из типов отзывов, на котором мы можем заметить, что действительно в негативных отзывах чаще используются другие слова чем в позитивных, хотя встречаются и похожие.
Также анализ данных показал, что изначальный датасет содержит различные малоинформативные слова и некоторые шумовые символы, которые были в последствие удалены.
## Обучение и оценка моделей
Каждая из трёх моделей была обучена на обработанных данных, а их производительность оценивалась с использованием следующих метрик:

Точность (Accuracy): Доля правильно предсказанных наблюдений среди всех наблюдений.
Точность предсказания (Precision): Доля истинных положительных предсказаний среди всех положительных предсказаний.
Полнота (Recall): Доля истинных положительных предсказаний среди всех фактических положительных.
F1-мера (F1-Score): Гармоническое среднее между точностью предсказания и полнотой, обеспечивающее баланс между ними.
ROC-AUC: Площадь под кривой ошибок (ROC-кривая), измеряющая способность модели различать классы.

| Метрика          | SVM            | Нейронная сеть  | LSTM           |
|------------------|----------------|-----------------|----------------|
| **Accuracy**     |      0.89      |    0.8741       |    0.8658      |
| **F1-мера**      |      0.89      |     0.87        |      0.87      |
| **ROC-AUC**      |      0.96      |       -         |      0.93      |
