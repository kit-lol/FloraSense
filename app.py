from flask import Flask, request, render_template
import torch
from torchvision import transforms
from PIL import Image
from model1 import PlantDiseaseClassifier
from model import UNet  # Импортируем модели
import base64
import numpy as np
from io import BytesIO  # Импортируем BytesIO

# Инициализация Flask-приложения
app = Flask(__name__)

# Загрузка моделей
segmentation_model = UNet()
segmentation_model.load_state_dict(torch.load('saved_models/model_epoch_9.pt'))
segmentation_model.eval()

classification_model = PlantDiseaseClassifier(num_classes=38)
classification_model.load_state_dict(torch.load('saved_models/best_model.pth'))
classification_model.eval()

# Преобразования для изображений
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Главная страница с формой загрузки и результатом
@app.route('/', methods=['GET', 'POST'])
def index():
    segmentation_mask = None  # Инициализируем переменные
    predicted_class = None
    original_image = None

    if request.method == 'POST':
        # Получаем изображение из запроса
        file = request.files['file']
        image = Image.open(file).convert('RGB')
        tensor_image = transform(image).unsqueeze(0)

        # Прогон через модели
        with torch.no_grad():
            # Сегментация
            segmentation_output = segmentation_model(tensor_image)
            segmentation_mask = torch.sigmoid(segmentation_output).squeeze().detach().numpy()

            # Классификация
            classification_output = classification_model(tensor_image)
            predicted_class = torch.argmax(classification_output, dim=1).item()

        # Преобразуем segmentation_mask в строку
        segmentation_mask_str = segmentation_mask.tobytes()  # Исправлено на tobytes()

        # Преобразуем исходное изображение в base64 для отображения
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        original_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template('index.html', segmentation_mask=segmentation_mask_str, predicted_class=predicted_class, original_image=original_image)

# Маршрут для обработки POST-запроса на /predict
@app.route('/predict', methods=['POST'])
def predict():
    segmentation_mask = None  # Инициализируем переменные
    predicted_class = None
    original_image = None

    # Получаем изображение из запроса
    file = request.files['file']
    image = Image.open(file).convert('RGB')
    tensor_image = transform(image).unsqueeze(0)

    # Прогон через модели
    with torch.no_grad():
        # Сегментация
        segmentation_output = segmentation_model(tensor_image)
        segmentation_mask = torch.sigmoid(segmentation_output).squeeze().detach().numpy()

        # Классификация
        classification_output = classification_model(tensor_image)
        predicted_class = torch.argmax(classification_output, dim=1).item()

    # Преобразуем segmentation_mask в строку
    segmentation_mask_str = segmentation_mask.tobytes()  # Исправлено на tobytes()

    # Преобразуем исходное изображение в base64 для отображения
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    original_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    return render_template('index.html', segmentation_mask=segmentation_mask_str, predicted_class=predicted_class, original_image=original_image)

# Запуск приложения
if __name__ == '__main__':
    app.run(debug=True)