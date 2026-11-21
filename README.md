# Titanic — инженерные практики ML

Учебный проект по бинарной классификации. По классу билета, полу,
возрасту, тарифу и числу родственников на борту предсказываем, выжил ли пассажир.

Источник данных — [Titanic, OpenML №40945, версия 1](https://www.openml.org/d/40945).
Лицензия в [метаданных OpenML](https://www.openml.org/api/v1/json/data/40945)
указана как `Public`. Скрипт скачивает исходный ARFF без токена в
`data/raw/titanic.arff` и проверяет SHA-256.

## Запуск

Нужны Git, Python 3.11 и [uv](https://docs.astral.sh/uv/getting-started/installation/).
Выполните команды из корня репозитория.

```bash
uv sync --locked
uv run pre-commit install
uv run pre-commit run --all-files
uv run python -m src.get_data
uv run python -m src.train
```

Обучаем логистическую регрессию. Используем стратифицированное разбиение
80/20 с `random_state=42`. Заполнение пропусков, масштабирование и
кодирование категориальных признаков настраиваются на обучающей выборке.
Поля `boat` и `body` раскрывают исход и исключены из признаков.

Модель вместе с предобработкой сохраняется в `artifacts/model.pkl`.
В `artifacts/metrics.json` сохраняются accuracy и ROC-AUC на тестовой
выборке, а также размеры обучающей и тестовой выборок.

[Отчёт по ДЗ №1](reports/hw-1.md)
