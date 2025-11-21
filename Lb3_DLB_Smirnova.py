import torch
import random

# 1 Создайте тензор x целочисленного типа, хранящий случайное значение
x_int = torch.randint(low=0, high=4, size=(1,), dtype=torch.int32)
print(x_int)

# 2 Преобразуйте тензор к типу float32
x_float = x_int.to(dtype=torch.float32)
print(x_float)

# Для вычисления производной
x_float.requires_grad_(True)

# 3 Проведите с тензором x ряд операций:
# – возведение в степень n, где n = 3, если ваш номер по списку группы в ЭИОС – четный 
# и n = 2, если ваш номер по списку группы в ЭИОС – нечетный
student_id = 21 
n = 3 if student_id % 2 == 0 else 2
print("n =", n)
x_powered = x_float ** n
print(x_powered)

# – умножение на случайное значение в диапазоне от 1 до 10;
random_multiplier = random.uniform(1, 10)
x_multiplied = x_powered * random_multiplier
print ("случайное число=", random_multiplier, "итог =", x_multiplied)

# – взятие экспоненты от полученного числа.
x_exp = torch.exp(x_multiplied)
print(x_exp)

# 4 Вычислите и выведите на экран значение производной для полученного в п.3 значения по x.
# обратное распространение ошибки
x_exp.backward()

# Градиент будет храниться в x_float.grad
print(x_float.grad)


#задание 2: на основе кода обучения линейного алгоритма создать код для решения задачи 
#классификации цветков ириса из лабораторной работы №2

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader


df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)
df.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# Признаки (X), целевая переменная (y)
X_data = df.iloc[:, 0:4].values
y_data = df.iloc[:, 4].values

# Преобразование строковых меток классов в числовые (0, 1, 2)
le = LabelEncoder()
y_labels_encoded = le.fit_transform(y_data) # [0, 1, 2]


# Разделение данных на обучаемую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X_data, y_labels_encoded, test_size=0.2, random_state=42, stratify=y_labels_encoded)

# Масштабирование признаков 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Преобразование данных в тензоры PyTorch
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long) # Для CrossEntropyLoss нужны Long тензоры
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(dataset=train_dataset, batch_size=16, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(dataset=test_dataset, batch_size=16, shuffle=False)

#Модель
class LinearClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LinearClassifier, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

input_size = X_train_tensor.shape[1]
num_classes = len(le.classes_)      
model = LinearClassifier(input_size, num_classes)

print ('w: ', model.linear.weight.data)
print ('b: ', model.linear.bias.data)

#Функция потерь и оптимизатора
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#Обучение модели
num_epochs = 100 

for epoch in range(num_epochs):
    model.train() 
    for batch_X, batch_y in train_loader:
        pred = model(batch_X)
        loss = criterion(pred, batch_y)

        # Обратное распространение и оптимизация
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print('эпоха: ', epoch+1, 'ошибка',loss.item())

#Оценка модели на тестовых данных
model.eval() #модель в режим оценки
with torch.no_grad():
    correct = 0
    total = 0
    for batch_X, batch_y in test_loader:
        outputs = model(batch_X)
        _, predicted = torch.max(outputs.data, 1) 
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

    accuracy = 100 * correct / total
    print('точность модели на тестовой выборке:', accuracy)

#Визуализация
print ('w: ', model.linear.weight.data)
print ('b: ', model.linear.bias.data)

# Пример предсказания для первого тестового образца
sample_input = X_test_tensor[0].unsqueeze(0) #добавляем размерность батча
true_label = y_test_tensor[0].item()

with torch.no_grad():
    model.eval()
    prediction_output = model(sample_input)
    _, predicted_class_idx = torch.max(prediction_output, 1)
    predicted_class_name = le.inverse_transform([predicted_class_idx.item()])[0]
    true_class_name = le.inverse_transform([true_label])[0]

print('Входные данные (масштабированные):', X_test_scaled[0])
print('Истинный класс:', true_class_name)
print('Предсказанный класс:', predicted_class_name)

