import torch
from torch.utils.data import DataLoader
from dataset import LeafSegmentationDataset
from transforms import transform
from model import UNet  # импортируем нашу модель U-Net

# Параметры обучения
BATCH_SIZE = 4
EPOCHS = 10
LEARNING_RATE = 0.001

# Инициализация модели и устройства
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UNet().to(device)

# Критерий и оптимизатор
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Загружаем данные
dataset = LeafSegmentationDataset('data', transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Обучение
for epoch in range(EPOCHS):
    model.train()
    for batch_idx, (images, masks) in enumerate(loader):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        print(f'Эпоха {epoch}, партия {batch_idx}, потеря: {loss.item()}')

torch.save(model.state_dict(), 'saved_models/model_epoch_{}.pt'.format(epoch))