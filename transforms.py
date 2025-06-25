from torchvision import transforms

# Преобразования для изображений (включают нормализацию)
image_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Масштабируем до 256×256
    transforms.ToTensor(),           # Преобразуем в тензор
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Нормализуем RGB-канал
])

# Преобразования для масок (только резайз и преобразование в тензор)
mask_transforms = transforms.Compose([
    transforms.Resize((256, 256)),  # Масштабируем до 256×256
    transforms.ToTensor()            # Преобразуем в тензор
])

# Словарь трансформаций
transform = {"image": image_transforms, "mask": mask_transforms}