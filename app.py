import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from flask import Flask, render_template, request, jsonify
from model import UNet  # Модель сегментации
from model1 import PlantDiseaseClassifier  # Модель классификации
import json
import io
import base64
import logging

app = Flask(__name__)

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Загрузка конфигурации
with open('config.json', 'r') as f:
    config = json.load(f)

# Настройка устройства
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Загрузка модели классификации
classification_model = PlantDiseaseClassifier(config['num_classes'])
classification_model.load_state_dict(torch.load('saved_models/best_model.pth', map_location=device))
classification_model.to(device)
classification_model.eval()  # Режим оценки
logger.info("Classification model loaded")

# Загрузка модели сегментации
segmentation_model = UNet(n_channels=3, n_classes=1)
segmentation_model.load_state_dict(torch.load('saved_models/model_epoch_9.pt', map_location=device))
segmentation_model.to(device)
segmentation_model.eval()  # Режим оценки
logger.info("Segmentation model loaded")

# Трансформация изображений
transform = transforms.Compose([
    transforms.Resize(tuple(config["input_size"])),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Список классов с ручным переводом
classes = [
    "Парша яблони", "Чёрная гниль яблони", "Ржавчина яблони", "Здоровый яблоневый лист",
    "Здоровый лист черники", "Мучнистая роса вишни", "Здоровый лист вишни",
    "Пятнистость листьев кукурузы", "Обыкновенная ржавчина кукурузы", "Северная пятнистость листьев кукурузы",
    "Здоровый лист кукурузы", "Чёрная гниль винограда", "Эска винограда", "Пятнистость листьев винограда",
    "Здоровый лист винограда", "Зеленение цитрусовых", "Бактериальная пятнистость персика",
    "Здоровый лист персика", "Бактериальная пятнистость сладкого перца", "Здоровый лист сладкого перца",
    "Ранняя пятнистость картофеля", "Поздняя пятнистость картофеля", "Здоровый лист картофеля",
    "Здоровый лист малины", "Здоровый лист сои", "Мучнистая роса кабачка", "Ожог листьев клубники",
    "Здоровый лист клубники", "Бактериальная пятнистость томата", "Ранняя пятнистость томата",
    "Поздняя пятнистость томата", "Мучнистая роса томата", "Пятнистость листьев томата",
    "Паутинный клещ томата", "Пятнистость томата", "Вирус скручивания листьев томата",
    "Вирус мозаики томата", "Здоровый лист томата"
]

# Создаем папку для загруженных изображений, если ее нет
UPLOAD_FOLDER = 'static/uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    logger.info(f"Created upload folder: {UPLOAD_FOLDER}")

def save_segmentation_image(mask, filename):
    """Сохраняет изображение сегментации как файл PNG"""
    try:
        # Создаем изображение с цветовой картой
        plt.figure(figsize=(5, 5))
        plt.imshow(mask, cmap='viridis')
        plt.axis('off')
        
        # Сохраняем в буфер
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        plt.close()
        buf.seek(0)
        
        # Сохраняем как файл
        image_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(image_path, 'wb') as f:
            f.write(buf.getbuffer())
        
        return image_path
    except Exception as e:
        logger.error(f"Error saving segmentation image: {str(e)}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        # Сохраняем загруженное изображение
        filename = file.filename
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        logger.info(f"Image saved: {file_path}")

        # Открываем изображение
        image = Image.open(file_path).convert('RGB')
        logger.info(f"Image size: {image.size}, mode: {image.mode}")

        # Подготавливаем изображение для моделей
        input_tensor = transform(image).unsqueeze(0).to(device)
        logger.info(f"Input tensor shape: {input_tensor.shape}")

        # Прогоняем изображение через модель сегментации
        with torch.no_grad():
            segmentation_output = segmentation_model(input_tensor)
            segmentation_mask = torch.sigmoid(segmentation_output).squeeze().cpu().numpy()
            
            # Логируем статистику маски
            logger.info(f"Mask stats: min={segmentation_mask.min():.4f}, max={segmentation_mask.max():.4f}, mean={segmentation_mask.mean():.4f}")

        # Прогоняем изображение через модель классификации
        with torch.no_grad():
            classification_output = classification_model(input_tensor)
            probabilities = F.softmax(classification_output, dim=1)[0]
            logger.info(f"Classification probabilities: {probabilities}")

        # Получаем наиболее вероятный класс
        top_prob, top_class = torch.topk(probabilities, k=1)
        predicted_class = classes[top_class.item()]
        probability = top_prob.item() * 100
        logger.info(f"Predicted class: {predicted_class}, probability: {probability:.2f}%")

        # Создаем и сохраняем изображение сегментации
        segmentation_filename = f"segmentation_{filename.split('.')[0]}.png"
        segmentation_path = save_segmentation_image(segmentation_mask, segmentation_filename)
        logger.info(f"Segmentation image saved: {segmentation_path}")

        # Формируем ответ
        response = {
            'diagnosis': predicted_class,
            'probability': f"{probability:.2f}",
            'segmentation_image': f"/static/uploads/{segmentation_filename}"
        }

        return jsonify(response)

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)