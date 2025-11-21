import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split 


df = pd.read_csv('dataset_simple.csv') 
df.columns = ['Возраст', 'Доход', 'Купит']

X = torch.tensor(df[['Возраст', 'Доход']].values, dtype=torch.float32)
y = torch.tensor(df['Купит'].values, dtype=torch.float32).view(-1, 1)

print('X shape:', {X.shape})
print('y shape', {y.shape})

#Масштабирование данных
scaler = StandardScaler()
X_scaled_np = scaler.fit_transform(X.numpy())
X_scaled = torch.tensor(X_scaled_np, dtype=torch.float32)

#создание нейрона
class NNet_classification(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(NNet_classification, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
    
    def forward(self, X):
        pred = self.layers(X)
        return pred 

inputSize = 2       
hiddenSizes = 32    
outputSize = 1     

net = NNet_classification(inputSize, hiddenSizes, outputSize)

#Функция потерь и оптимизатора
lossFn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.01) # Adam обычно лучше SGD для начала

#Обучение модели
epochs = 1000

print("\nНачинаем обучение...")
for i in range(epochs):
    pred_logits = net(X_scaled)   
    loss = lossFn(pred_logits, y)  
    
    optimizer.zero_grad()          
    loss.backward()               
    optimizer.step()              
    
    if i % 100 == 0:
       #print(f'Ошибка на {i+1} итерации: {loss.item():.4f}')
        print('итерация: ', i+1, 'ошибка', loss.item())

#Оценка модели после обучения
with torch.no_grad():
    pred_logits_final = net(X_scaled)
    probabilities = torch.sigmoid(pred_logits_final) 
    predicted_classes = (probabilities >= 0.5).float() 

err_count = (predicted_classes != y).sum().item() # Количество ошибок
accuracy = (1 - err_count / y.shape[0]) * 100 # Точность в процентах

print('Предсказания:', predicted_classes[0:10].flatten().tolist()) 
print('Истинные метки:', y[0:10].flatten().tolist())
print('Всего ошибок:', err_count)
print('Точность модели', accuracy)

