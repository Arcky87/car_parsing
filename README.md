Here's a Russian version of README.md:

```markdown
# Обработчик PDF-файлов штрафов ГИБДД

Программа для автоматической обработки PDF-файлов со штрафами ГИБДД, извлечения информации и отправки уведомлений.

## Установка

1. Установите зависимости системы:
```bash
sudo apt update
sudo apt install tesseract-ocr-rus libtesseract-dev python3-pip
```

2. Установите необходимые Python-пакеты:
```bash
pip install -r requirements.txt
```

## Настройка

1. Откройте файл `config.yaml` и настройте:
   - URL папки Google Drive с PDF-файлами
   - Данные для отправки email (SMTP-сервер, логин, пароль)
   - Адреса получателей уведомлений

Пример конфигурации:
```yaml
google_drive:
  folder_url: "https://drive.google.com/drive/folders/your_folder_id"
  download_dir: "downloaded_pdfs"

email:
  smtp_server: "smtp.gmail.com"
  smtp_port: 587
  sender_email: "your_email@gmail.com"
  sender_password: "your_app_password"
  recipients:
    cargo: "cargo-web@o2rus.ru"
    tech: "o2rus.tech@gmail.com"
```

## Использование

Запустите программу командой:
```bash
python main.py
```

## Функциональность

- Извлечение текста из PDF-файлов
- Распознавание номеров автомобилей с помощью OCR
- Определение суммы штрафа
- Отправка email-уведомлений
- Создание сводной таблицы в формате DOCX

## Алгоритм работы

1. Программа сканирует указанную папку на наличие PDF-файлов
2. Из каждого PDF извлекается:
   - Текстовая информация о штрафе
   - Фотография автомобиля
   - Изображение номерного знака
3. Выполняется OCR номерного знака
4. Создается сводная таблица с результатами
5. Отправляются уведомления:
   - На cargo-web@o2rus.ru, если номер распознан верно и штраф > 5000 руб
   - На o2rus.tech@gmail.com в остальных случаях

## Структура проекта

```
pdf_processor/
├── README.md           # Документация
├── requirements.txt    # Зависимости Python
├── config.yaml        # Конфигурационный файл
├── main.py           # Основной скрипт
└── utils.py          # Вспомогательные функции
```

## Возможные проблемы

1. Если возникает ошибка с Tesseract:
   - Проверьте установку tesseract-ocr-rus
   - Убедитесь, что путь к Tesseract правильно прописан в системе

2. При проблемах с отправкой email:
   - Проверьте настройки SMTP
   - Для Gmail используйте специальный пароль приложения
   - Проверьте доступ к SMTP-серверу
