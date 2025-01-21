# Car Plate Parser - Система обработки штрафов ПДД

# WARNING! TODO: Распознавание tesseract не работает должным образом. Работаю над добавлением распознавания NomeroffNet  


## Описание
Система автоматизированной обработки PDF-документов со штрафами ПДД. Программа извлекает информацию из PDF-файлов, распознает номера автомобилей с помощью OCR и формирует отчеты с возможностью email-уведомлений.

## Функциональные возможности
- Автоматическая загрузка PDF-файлов из Google Drive
- Извлечение текста и изображений из PDF
- Распознавание автомобильных номеров (OCR)
- Формирование отчетов в формате DOCX
- Отправка email-уведомлений о нарушениях
- Генерация ежедневной статистики

## Требования к системе
- Python 3.8 или выше
- Tesseract OCR
- Достаточно свободного места на диске для загрузки и обработки файлов
- Доступ в интернет для загрузки файлов и отправки email

### Установка зависимостей

#### Ubuntu/Debian
```bash
# Установка системных зависимостей
sudo apt-get update
sudo apt-get install -y python3-pip python3-venv
sudo apt-get install -y tesseract-ocr tesseract-ocr-rus

# Создание виртуального окружения
python3 -m venv venv
source venv/bin/activate

# Установка Python-зависимостей
pip install -r requirements.txt
```
## Настройка

### 1. Конфигурация
Создайте файл `config/config.yaml` со следующими настройками:

```yaml
google_drive:
  folder_url: "ваша_ссылка_на_папку_google_drive"
  download_dir: "downloaded_pdfs/RDF"

email:
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  sender_email: "ваша.почта@gmail.com"
  sender_password: "ваш_пароль_приложения"
  recipients:
    cargo: "получатель1@example.com"
    tech: "получатель2@example.com"
  alerts:
    fine_threshold: 5000
    send_daily_stats: true
    include_images: true

ocr:
  tesseract:
    lang: "rus"
    psm: 7
    whitelist: "АВЕКМНОРСТУХ0123456789"
    path: "путь_к_tesseract"  # например: "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
```

### 2. Структура проекта
```
car_plate_parser/
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── processors.py
│   ├── exceptions.py
│   └── ocr/
│       ├── __init__.py
│       └── recognizer.py
├── config/
│   └── config.yaml
├── requirements.txt
└── README.md
```

## Запуск приложения

### Метод 1: Прямой запуск
```bash
python -m src.main
```

### Метод 2: Установка как пакет
```bash
pip install -e .
car-plate-parser
```

## Логирование и отчеты

### Логи
- Основной лог: `app.log`
- Уровень логирования настраивается в конфигурации

### Отчеты
- Таблица нарушений: `output_table.docx`
- Ежедневная статистика: `daily_statistics.docx`

## Устранение неполадок

### Частые проблемы и решения

1. Ошибка "Tesseract not found":
   - Проверьте путь к Tesseract в config.yaml
   - Убедитесь, что Tesseract установлен корректно

2. Ошибки при загрузке файлов:
   - Проверьте доступность URL Google Drive
   - Проверьте права доступа к папке загрузки

3. Ошибки отправки email:
   - Проверьте настройки SMTP
   - Убедитесь, что пароль приложения корректен
