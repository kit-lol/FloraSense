document.addEventListener('DOMContentLoaded', function() {
    // Элементы интерфейса
    const fileInput = document.getElementById('fileInput');
    const previewContainer = document.getElementById('preview-container');
    const preview = document.getElementById('preview');
    const submitButton = document.getElementById('submitImage');
    const welcomeSection = document.getElementById('welcome');
    const aboutSection = document.getElementById('about');
    const featuresSection = document.getElementById('features');
    const uploadForm = document.querySelector('.upload-form');
    const resultsContainer = document.getElementById('results-container');
    const originalImage = document.getElementById('original-image');
    const segmentationImage = document.getElementById('segmentation-image');
    const diagnosisText = document.getElementById('diagnosis-text');
    const probabilityText = document.getElementById('probability-text');
    const newAnalysisButton = document.getElementById('newAnalysis');
    const btnAbout = document.getElementById('btnAbout');
    const btnFeatures = document.getElementById('btnFeatures');
    const header = document.querySelector('header');

    // Фиксация шапки при прокрутке
    window.addEventListener('scroll', function() {
        if (window.scrollY > 30) {
            header.classList.add('scrolled');
            document.body.classList.add('scrolled');
        } else {
            header.classList.remove('scrolled');
            document.body.classList.remove('scrolled');
        }
    });

    // Показ/скрытие информационных секций
    btnAbout.addEventListener('click', function() {
        if (aboutSection.classList.contains('hidden')) {
            aboutSection.classList.remove('hidden');
            welcomeSection.classList.add('hidden');
            featuresSection.classList.add('hidden');
        } else {
            aboutSection.classList.add('hidden');
            welcomeSection.classList.remove('hidden');
        }
    });

    btnFeatures.addEventListener('click', function() {
        if (featuresSection.classList.contains('hidden')) {
            featuresSection.classList.remove('hidden');
            welcomeSection.classList.add('hidden');
            aboutSection.classList.add('hidden');
        } else {
            featuresSection.classList.add('hidden');
            welcomeSection.classList.remove('hidden');
        }
    });

    // Загрузка изображения
    fileInput.addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(event) {
                preview.src = event.target.result;
                preview.style.display = 'block';
                previewContainer.classList.remove('hidden');
            }
            reader.readAsDataURL(file);
        }
    });

    // Отправка изображения на анализ
    submitButton.addEventListener('click', function() {
        if (!fileInput.files[0]) {
            alert('Пожалуйста, выберите изображение');
            return;
        }

        // Скрываем все секции
        welcomeSection.classList.add('hidden');
        aboutSection.classList.add('hidden');
        featuresSection.classList.add('hidden');
        uploadForm.classList.add('hidden');
        previewContainer.classList.add('hidden');

        // Показываем экран загрузки
        showLoadingScreen();

        // Формируем данные для отправки
        const formData = new FormData();
        formData.append('file', fileInput.files[0]);

        // Отправляем запрос на сервер
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Ошибка сервера: ' + response.status);
            }
            return response.json();
        })
        .then(data => {
            // Скрываем экран загрузки
            hideLoadingScreen();

            if (data.error) {
                alert('Ошибка: ' + data.error);
                return;
            }

            // Отображаем результаты
            originalImage.src = URL.createObjectURL(fileInput.files[0]);
            segmentationImage.src = data.segmentation_image;
            diagnosisText.textContent = data.diagnosis;
            probabilityText.textContent = `Точность: ${data.probability}%`;
            resultsContainer.classList.remove('hidden');
            
            // Прокрутка к результатам
            resultsContainer.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            hideLoadingScreen();
            console.error('Ошибка:', error);
            alert('Произошла ошибка: ' + error.message);
        });
    });

    // Кнопка нового анализа
    newAnalysisButton.addEventListener('click', function() {
        resultsContainer.classList.add('hidden');
        welcomeSection.classList.remove('hidden');
        uploadForm.classList.remove('hidden');
        fileInput.value = '';
        preview.style.display = 'none';
        previewContainer.classList.add('hidden');
        
        // Прокрутка к началу
        window.scrollTo({ top: 0, behavior: 'smooth' });
    });

    // Функции для экрана загрузки
    function showLoadingScreen() {
        const overlay = document.createElement('div');
        overlay.className = 'loading-overlay';
        overlay.innerHTML = '<div class="spinner"></div>';
        document.body.appendChild(overlay);
    }

    function hideLoadingScreen() {
        const overlay = document.querySelector('.loading-overlay');
        if (overlay) overlay.remove();
    }
});