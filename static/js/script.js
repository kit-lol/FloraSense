document.addEventListener('DOMContentLoaded', function() {
    // Безопасное получение элементов
    const elements = {
        btnAbout: document.getElementById('btnAbout'),
        btnFeatures: document.getElementById('btnFeatures'),
        logoLink: document.getElementById('logoLink'),
        fileInput: document.getElementById('fileInput'),
        welcomeSection: document.getElementById('welcome'),
        preview: document.getElementById('preview')
    };

    // Проверяем и инициализируем только существующие элементы
    if (elements.btnAbout && elements.btnFeatures) {
        // Обработчики кнопок навигации
        elements.btnAbout.addEventListener('click', function() {
            toggleSection('about');
        });

        elements.btnFeatures.addEventListener('click', function() {
            toggleSection('features');
        });
    }

    // Обработчик логотипа с проверкой
    if (elements.logoLink) {
        elements.logoLink.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Безопасное применение стилей
            if (this.style) {
                this.style.transform = 'scale(0.95)';
            }
            
            if (document.body.style) {
                document.body.style.cursor = 'wait';
            }
            
            setTimeout(() => {
                window.location.href = '/';
            }, 200);
        });
    }

    // Превью изображения с проверкой
    if (elements.fileInput && elements.preview) {
        elements.fileInput.addEventListener('change', function(event) {
            const file = event.target.files[0];
            if (!file) return;

            const reader = new FileReader();
            
            reader.onload = function(e) {
                elements.preview.src = e.target.result;
                elements.preview.style.display = 'block';
            };
            
            reader.readAsDataURL(file);
        });
    }

    // Функция переключения секций
    function toggleSection(sectionId) {
        if (!elements.welcomeSection) return;
        
        // Скрываем welcome-секцию если она видна
        if (!elements.welcomeSection.classList.contains('hidden')) {
            elements.welcomeSection.classList.add('hidden');
        }
        
        // Скрываем все анимированные секции
        document.querySelectorAll('.animated-section').forEach(sec => {
            if (sec.classList) {
                sec.classList.add('hidden');
            }
        });
        
        // Показываем выбранную секцию
        const targetSection = document.getElementById(sectionId);
        if (targetSection && targetSection.classList) {
            targetSection.classList.remove('hidden');
        }
    }
});