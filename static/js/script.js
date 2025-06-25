document.addEventListener('DOMContentLoaded', function () {
    const btnAbout = document.getElementById('btnAbout');
    const btnFeatures = document.getElementById('btnFeatures');
    const logoLink = document.getElementById('logoLink');

    btnAbout.addEventListener('click', function () {
        toggleSection('about');
    });

    btnFeatures.addEventListener('click', function () {
        toggleSection('features');
    });

    logoLink.addEventListener('click', function (e) {
        e.preventDefault(); // предотвращаем переход по ссылке

        // Применяем небольшие изменения стилей для визуального эффекта
        document.querySelector('.wheat-left').style.transform = 'translateX(-20px) rotate(-10deg)';
        document.querySelector('.wheat-right').style.transform = 'translateX(20px) rotate(10deg) scaleX(-1)';

        // Пауза перед реальной перезагрузкой
        setTimeout(function () {
            window.location.reload();
        }, 500); // Задержка 0.5 секунды
    });

    function toggleSection(sectionId) {
        const welcomeSection = document.getElementById("welcome");
        if (!welcomeSection.classList.contains("hidden")) {
            welcomeSection.classList.add("hidden");
        }

        const sections = document.querySelectorAll('.animated-section');
        sections.forEach(sec => sec.classList.add('hidden'));
        document.getElementById(sectionId).classList.remove('hidden');
    }
});

document.addEventListener('DOMContentLoaded', function() {
    const logoLink = document.getElementById('logoLink');
    
    logoLink.addEventListener('click', function(e) {
        e.preventDefault(); // предотвращаем переход по ссылке
        
        // Применяем небольшие изменения стилей для визуального эффекта
        document.querySelector('.wheat-left').style.transform = 'translateX(-20px) rotate(-10deg)';
        document.querySelector('.wheat-right').style.transform = 'translateX(20px) rotate(10deg) scaleX(-1)';
        
        // задерживаем реальную перезагрузку на полсекунды
        setTimeout(() => {
            location.reload();
        }, 500);
    });
});

// Добавляем JavaScript-код для отображения превью изображения при выборе файла
document.getElementById('fileInput').addEventListener('change', function(event) {
    const file = event.target.files[0];
    const reader = new FileReader();

    reader.onload = function(e) {
        document.getElementById('preview').src = e.target.result;
        document.getElementById('preview').style.display = 'block';
    };

    reader.readAsDataURL(file);
});