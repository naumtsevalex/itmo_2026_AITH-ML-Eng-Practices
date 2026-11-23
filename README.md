# Titanic — инженерные практики ML

Учебный проект по бинарной классификации. По классу билета, полу,
возрасту, тарифу и числу родственников на борту предсказываем, выжил ли пассажир.

Источник данных — [Titanic, OpenML №40945, версия 1](https://www.openml.org/d/40945).
Лицензия в [метаданных OpenML](https://www.openml.org/api/v1/json/data/40945)
указана как `Public`. Скрипт скачивает исходный ARFF без токена в
`data/raw/titanic.arff` и проверяет SHA-256.

Доклад к семинару №2 — [Evidently: drift и качество данных](seminar/evidently/).

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

## ДЗ №2. Версии данных в DVC

Создайте исходный ARFF и подготовленный CSV для текущего коммита.

```bash
uv sync --locked
uv run dvc repro
uv run dvc status
```

В `.dvc/config` настроено локальное хранилище `titanic-dvc-storage` рядом с
папкой репозитория. На другой машине его можно заполнить из публичного
источника. В корне свежего клона выполните команды ниже.

```bash
uv sync --locked
git switch --detach data-v1
uv run dvc repro
uv run dvc push
git switch --detach data-v2
uv run dvc repro
uv run dvc push
git switch main
```

Чтобы проверить восстановление на этой же машине, клонируйте репозиторий ещё
раз в соседнюю папку. Оба клона будут использовать одно локальное хранилище.
В новом клоне выполните следующие команды.

```bash
uv sync --locked
git switch --detach data-v1
uv run dvc pull
uv run dvc status
wc -l data/prepared/titanic.csv
git switch --detach data-v2
uv run dvc pull
uv run dvc status
wc -l data/prepared/titanic.csv
```

С учётом заголовка `wc -l` покажет 1 310 и 655 строк соответственно.
[Схема данных](data/README.md) · [Отчёт по ДЗ №2](reports/hw-2.md)

## ДЗ №3. Эксперименты в MLflow

Выполните команды из корня репозитория на версии `hw-3` или в `main`.

### Подготовить данные

Если локальный DVC remote уже заполнен в ДЗ №2, переходите к запуску
экспериментов. На новой машине заполните его из OpenML в соседнем клоне.

```bash
git clone --branch data-v1 . ../titanic-hw3-data-v1
(
  cd ../titanic-hw3-data-v1
  uv sync --locked
  uv run dvc repro
  uv run dvc push
)
```

### Записать три запуска

```bash
uv sync --locked
mkdir -p artifacts/hw-3
uv run dvc get . data/prepared/titanic.csv --rev data-v1 --out artifacts/hw-3/titanic.csv
uv run python -m src.experiments
```

DVC восстанавливает CSV версии `data-v1` в отдельную папку внутри
`artifacts/`. Скрипт проверяет SHA-256, один раз делит данные на train/test
и записывает три запуска с параметрами, метриками и моделями. Повторный запуск
добавляет ещё три записи. SQLite-база и модели хранятся в `artifacts/hw-3/`.

### Открыть результаты

```bash
uv run mlflow ui --backend-store-uri sqlite:///artifacts/hw-3/mlflow.db --host 127.0.0.1 --port 5000
```

Откройте [локальный MLflow](http://127.0.0.1:5000), выберите эксперимент
**Titanic HW3** и переключитесь в **Model training → Runs**. Для сравнения
отметьте три запуска и нажмите **Compare**.
В карточке каждого запуска модель находится в **Artifacts → model**.

[Отчёт по ДЗ №3](reports/hw-3.md)
