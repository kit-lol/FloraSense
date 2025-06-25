import json
import torch
from torch import nn
from torch.optim import Adam
from tqdm import tqdm  # <<< Обратите внимание на изменение тут!
from dataset1 import load_data
from model1 import PlantDiseaseClassifier

if __name__ == "__main__":
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Загружаем данные
    train_loader, test_loader = load_data(config)
    
    # Инициализация модели
    model = PlantDiseaseClassifier(config['num_classes']).to(device)
    
    # Критерий потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=config['learning_rate'])
    
    # Цикл обучения
    best_accuracy = 0.0
    for epoch in range(config['epochs']):
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        # Тренировочная фаза
        model.train()
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["epochs"]}')  # << Теперь тут стандартный прогрессбар
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)
            
            accuracy = correct_predictions / total_samples
            progress_bar.set_postfix({'loss': loss.item(), 'accuracy': accuracy})
        
        # Проверочная фаза
        model.eval()
        val_correct_predictions = 0
        val_total_samples = 0
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                val_correct_predictions += (predicted == targets).sum().item()
                val_total_samples += targets.size(0)
        
        val_accuracy = val_correct_predictions / val_total_samples
        print(f'\nValidation Accuracy after Epoch {epoch + 1}: {val_accuracy:.4f}')
        
        # Сохранение лучшей модели
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), 'saved_models/best_model.pth')
            print(f'Saved new best model with validation accuracy of {best_accuracy:.4f}')