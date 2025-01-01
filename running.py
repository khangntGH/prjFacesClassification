import torch
from torch.utils.data import DataLoader
from model import FaciesClassification
import torch.nn as nn
from data.process_data import CustomDataset
import pandas as pd 

name = 'NOLAN'
data = pd.read_csv('./data/Data.csv')
test_data = data[data['Well Name'] == name]
training_data = data[data['Well Name'] != name]


train_dataset = CustomDataset(training_data)
test_dataset = CustomDataset(test_data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = FaciesClassification(num_categories=2, num_numerical_features=7, n_class=9)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

def train_model(model, data_loader, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        for category_data, numerical_data, targets in data_loader:
            optimizer.zero_grad()
            outputs = model(category_data, numerical_data)
            loss = criterion(outputs, targets-1)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

def evaluate_model(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for category_data, numerical_data, targets in data_loader:
            outputs = model(category_data, numerical_data)
            loss = criterion(outputs, targets-1)
            total_loss += loss.item()
    avg_loss = total_loss / len(data_loader)
    print(f'Average Test Loss: {avg_loss}')

# Training
train_model(model, train_loader, num_epochs=50)
# Evaluation
evaluate_model(model, test_loader)