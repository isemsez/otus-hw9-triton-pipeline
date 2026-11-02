# OTUS HW9 — Оркестрация моделей в Triton Inference Server

ML-пайплайн «классификация → эмбеддинг → генерация» под управлением NVIDIA Triton
Inference Server с бизнес-логикой на уровне сервера (Business Logic Scripting, BLS).

## Сценарий пайплайна

```
            ┌──────────────────────────────────────────────────────┐
 TEXT ────► │ pipeline (Python backend, BLS)                       │
            │  1. text_embedder   (ONNX, all-MiniLM-L6-v2)         │
            │  2. text_classifier (ONNX, LogisticRegression)       │
            │  3. if «question»: text_generator (Qwen2.5-0.5B)     │ ───► LABEL / PROBS / ANSWER
            └──────────────────────────────────────────────────────┘
```

Пользователь присылает текст. BLS-скрипт токенизирует его, получает эмбеддинг
(all-MiniLM-L6-v2, ONNX → onnxruntime backend), по эмбеддингу классифицирует
«вопрос / не вопрос» (LogisticRegression, ONNX → onnxruntime backend) и только
для вопросов вызывает LLM (Qwen2.5-0.5B-Instruct, Python backend), возвращая
единый ответ: метку, вероятности и сгенерированный ответ.

## Структура репозитория

```
hw9/
├── model_repository/
│   ├── text_embedder/                 # эмбеддинги, backend: onnxruntime
│   │   ├── config.pbtxt
│   │   └── 1/model.onnx               # all-MiniLM-L6-v2 → ONNX (~90 МБ)
│   ├── text_classifier/               # классификатор, backend: onnxruntime
│   │   ├── config.pbtxt
│   │   └── 1/model.onnx               # LogisticRegression → skl2onnx
│   ├── text_generator/                # LLM, backend: python
│   │   ├── config.pbtxt
│   │   └── 1/
│   │       ├── model.py               # Qwen2.5-0.5B-Instruct, CPU
│   │       └── artifacts/qwen/        # веса (~1 ГБ, не в git — см. ниже)
│   └── pipeline/                      # BLS-оркестратор, backend: python
│       ├── config.pbtxt
│       └── 1/
│           ├── model.py               # BLS: embedder → classifier → generator
│           └── artifacts/tokenizer/   # токенизатор MiniLM
├── scripts/
│   └── run_perf.sh                    # нагрузочные тесты Perf Analyzer
├── results/                           # логи Perf Analyzer и метрики
├── client_http.py                     # проверка BLS-цепочки одним запросом (HTTP)
├── client_grpc.py                     # то же через gRPC (сравнение протоколов)
├── prepare_models.py                  # экспорт/обучение/скачивание моделей
├── Dockerfile                         # tritonserver + torch/transformers
├── docker-compose.yml                 # порты 8000/8001/8002, volume с моделями
└── README.md
```

## Модели и бэкенды

| Модель | Артефакт | Backend | Назначение |
|---|---|---|---|
| `text_embedder` | `model.onnx` (all-MiniLM-L6-v2, 384-dim) | `onnxruntime` | эмбеддинг текста |
| `text_classifier` | `model.onnx` (LogisticRegression) | `onnxruntime` | вопрос/утверждение по эмбеддингу |
| `text_generator` | `model.py` + Qwen2.5-0.5B-Instruct | `python` | генерация ответа |
| `pipeline` | `model.py` (BLS) | `python` | оркестрация всей цепочки |

> **Замечание о генеративной модели.** Задание предполагает vLLM/TensorRT-LLM
> backend, но оба требуют NVIDIA GPU. Работа выполняется на машине без NVIDIA GPU
> (Docker без `--gpus`), поэтому LLM обслуживается Python backend'ом с
> Qwen2.5-0.5B-Instruct на CPU — это допустимая для Triton схема, а конфигурация
> репозитория и BLS-логика не меняются при переезде на vLLM backend.

## Запуск

### 1. Подготовка моделей

```bash
python -m venv .venv
.venv/Scripts/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/pip install transformers skl2onnx scikit-learn onnxruntime onnx huggingface_hub
.venv/Scripts/python prepare_models.py
```

Скрипт экспортирует MiniLM в ONNX (`torch.onnx.export`), обучает LogisticRegression
на эмбеддингах (0 = statement, 1 = question) и экспортирует её через `skl2onnx`,
скачивает Qwen2.5-0.5B-Instruct (~1 ГБ) и токенизатор.

### 2. Сборка и запуск сервера

```bash
docker compose up -d --build
docker logs -f hw9-triton   # дождаться READY для всех 4 моделей
```

> **Если nvcr.io недоступен** (в некоторых сетях реестр NVIDIA отдаёт 403),
> образ можно получить через зеркало и перетегировать:
>
> ```bash
> docker pull docker.m.daocloud.io/nvcr.io/nvidia/tritonserver:24.12-py3
> docker tag  docker.m.daocloud.io/nvcr.io/nvidia/tritonserver:24.12-py3 \
>            nvcr.io/nvidia/tritonserver:24.12-py3
> # аналогично для :24.12-py3-sdk (Perf Analyzer)
> ```

### 3. Проверка цепочки

```bash
.venv/Scripts/pip install "tritonclient[http]" "tritonclient[grpc]"
.venv/Scripts/python client_http.py --text "What is the capital of France?"  # HTTP, порт 8000
.venv/Scripts/python client_grpc.py --text "What is the capital of France?"  # gRPC, порт 8001
.venv/Scripts/python client_http.py --text "The capital of France is Paris."
```

Первые две команды посылают вопрос (HTTP и gRPC — логика пайплайна одинакова,
меняется только транспорт), ответ генерирует LLM. Третья посылает утверждение:
генерация пропускается, `ANSWER` возвращается пустым.

### 4. Нагрузочное тестирование

```bash
bash scripts/run_perf.sh     # 4 прогона, логи в results/
```

## Результаты нагрузочного тестирования

Окружение: CPU-only (12 логических ядер, без NVIDIA GPU), Triton 24.12 (Triton
Server 2.53.0), Perf Analyzer из SDK-образа, HTTP-протокол. Полные логи — в
`results/perf_*.txt`.

### Итоговая таблица (concurrency = 1)

| Модель | Throughput, infer/s | p50, мс | p95, мс | Server avg, мс (queue / compute infer) |
|---|---:|---:|---:|---|
| `text_embedder` (ONNX, ORT CPU) | **81.2** | 11.6 | 17.8 | 5.14 (0.05 / **4.67**) |
| `text_classifier` (ONNX, ORT CPU) | **674** | 1.46 | 1.66 | 0.12 (0.05 / **0.012**) |
| `text_generator` (Qwen 0.5B, Python, CPU) | **0.79** | 1249 | 1284 | 1244 (0.04 / **1244**) |
| `pipeline` (полная BLS-цепочка) | **0.67** | 1485 | 1717 | 1488 (0.04 / **1488**) |

Разбивка задержки по стадиям сервера (Queue / Compute Input / Compute Infer /
Compute Output) видна в логах Perf Analyzer в секции `Server: Avg request latency`.
Например, для `text_embedder` при c=1: overhead 361 мкс + queue 51 мкс +
compute input 25 мкс + compute infer 4671 мкс + compute output ~30 мкс —
почти всё время занимает сам инференс.

### Масштабирование по concurrency

| Модель | c=1 | c=3/5/7 | Вывод |
|---|---|---|---|
| `text_classifier` | 674/s, 1.7 мс | 1669 / 2363 / 2899/s | почти линейное масштабирование (модель дешёвая, потоков хватает) |
| `text_embedder` | 81/s, 17.8 мс | 107 / 110 / 110/s, p95 до 83 мс | упор в ~110 infer/s: 1 instance ORT занял все ядра (CPU контейнера под нагрузкой — ~1030% из 1200%) |
| `text_generator` | 0.79/s | c=2: не стабилизировался в окне 10 с | 1 instance Python backend, генерация 48 токенов ~1.25 с |
| `pipeline` | 0.67/s, p95 1.72 с | c=2: 0.78/s, **p95 2.75 с** | throughput не растёт, задержка ×2 — классический признак единственного «узкого» инстанса |

### Использование ресурсов

- `docker stats hw9-triton` под нагрузкой `text_embedder`: **CPU 1028%**
  (≈10.3 из 12 ядер), память 2.48 GiB — ONNX Runtime сам распараллеливает
  инференс по ядрам.
- Метрики Triton (`:8002/metrics`, Prometheus): суммарно за прогоны —
  `nv_inference_request_success`: pipeline 117, text_generator 240,
  text_embedder 11263, text_classifier 234793 успешных запросов, ошибок нет.

### Узкое место

Задержку цепочки полностью определяет **`text_generator`**: 1244 мс из 1488 мс
серверной задержки pipeline (≈84%). Embedder добавляет ~5 мс, classifier ~0.1 мс,
оверхед BLS-оркестрации — единицы микросекунд (вызовы моделей идут внутри
сервера, без сетевых хопов). При concurrency=2 генератор ставит запросы в
очередь (единственный instance), p95 pipeline вырастает до 2.75 с при том же
throughput — то есть система упирается в CPU-генерацию LLM.

Специфика теста: для `text_embedder` использован `--input-data zero`
(случайные INT64 индексы невалидны для Gather токен-эмбеддингов; нулевой id —
это `[PAD]`), для классификатора задана полная форма `1,384`.

## Выводы

**Triton + BLS против самодельного FastAPI/Flask-сервиса.**

- *Единая точка входа при нескольких моделях.* BLS-вызовы (`pb_utils.InferenceRequest`)
  идут внутри сервера: нет сетевых хопов между сервисами, нет проблемы
  сериализации промежуточных тензоров, нет отдельного сервиса-«дирижёра»,
  который надо масштабировать и мониторить отдельно.
- *Готовая эксплуатационная обвязка.* HTTP/gRPC/metrics-эндпоинты, health-чеки
  моделей, версионирование репозитория (`1/`, откат = смена папки), hot-reload
  конфигов — в FastAPI всё это пришлось бы писать руками.
- *Наблюдаемость.* Разбивка задержки по стадиям (Queue / Compute Input / Infer /
  Output) для каждой модели и Perf Analyzer из коробки позволяют точно найти
  узкое место; для самописного сервиса потребовался бы собственный трейсинг.
- *Настройка параллелизма декларативно.* `instance_group`, dynamic batching,
  приоритеты моделей — правкой `config.pbtxt` без изменения кода.

Чего FastAPI проще: кастомная бизнес-логика произвольной сложности, отладка
обычным Python-стеком, отсутствие ограничений на окружение модели (в Python
backend надо жить в его conda/pip-окружении).

**Возможные оптимизации (если задержки высоки):**

1. **`text_generator` — главный кандидат.**
   - GPU + vLLM/TensorRT-LLM backend (в данной работе недоступен — нет NVIDIA GPU):
     генерация 48 токенов ушёл бы с ~1.25 с (CPU) к ~десяткам мс;
   - квантование (INT8/GGUF) и/или уменьшение `MAX_NEW_TOKENS`;
   - `instance_group [{ count: N }]` — несколько инстансов Python backend
     (на CPU даст ограниченный выигрыш: ядра уже заняты ORT-потоками);
   - dynamic batching для генератора (объединение запросов в батч).
2. **`text_embedder` (~110 infer/s на инстанс):** `instance_group count: 2-3`
   с ограничением ORT-потоков (`intra_op_thread_count`), либо dynamic batching
   (`max_queue_delay_us`) для сглаживания пиковой нагрузки.
3. **`text_classifier`** в оптимизациях не нуждается (0.67–2.9 тыс. infer/s,
   12 мкс compute).

