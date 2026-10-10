# DroPE: локальное воспроизведение

Учебный проект по статье Gelberg, Eguchi, Akiba, Cetin, **Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings**, arXiv:2512.12167v1.

## Что здесь воспроизводится

- Код авторов для NoPE attention используется без изменений (`vendor/custom_models/attention.py`).
- Маленький Transformer проверяет утверждение 3.2: одинаковые токены дают равномерное attention и одинаковые скрытые состояния без RoPE. Проверяется сохранение causal mask.
- На pretrained SmolLM-360M сравниваются RoPE, удаление RoPE без обучения и микрорекалибровка на WikiText-2. Это сокращенная абляция, а не полное воспроизведение обучения на миллиардах токенов.
- Для опубликованного Llama-2-7B-DroPE проверяется доступ к конфигурации и началу весов. Это не инференс 7B модели.
- Попытка запуска `run_authors_checkpoint.py` в фактической CPU-среде остановилась на проверке CUDA. Ошибка сохранена в `results/authors_launch_attempt.log`.
- В `results/authors_results.json` сохранены опубликованные числа из таблиц 1–3. Эти числа получены авторами, не локально.

## Быстрый запуск (Windows, Python 3.12)

Откройте PowerShell в этой папке. Установка идет в отдельное окружение:

```powershell
./setup_cpu.ps1
& ./.venv/Scripts/python.exe download_assets.py --cache cache
& ./.venv/Scripts/python.exe reproduce.py --cache cache --out results/local.json --steps 16 --seq-len 256 --device cpu
```

Только теоретические и структурные проверки, без скачивания pretrained модели:

```powershell
& ./.venv/Scripts/python.exe reproduce.py --structural-only --out results/structural.json
```

Веса SmolLM скачиваются отдельно и не включены в архив. Нужны интернет и около 2 ГБ места для модели, плюс место для PyTorch. `download_assets.py` фиксирует revisions модели и данных и создает SHA-256 manifest. Большие загрузки сохраняются через временный `.part` файл. Незавершенная загрузка начинается заново.

## Запуск готовой модели авторов (дополнительный сценарий)

Для RTX 5060 используйте отдельное окружение с CUDA 12.8 и поддержкой Blackwell:

```powershell
python -m venv .venv-gpu
& ./.venv-gpu/Scripts/python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
& ./.venv-gpu/Scripts/python.exe -m pip install -r requirements.txt
& ./.venv-gpu/Scripts/python.exe -m pip install accelerate==1.10.1 bitsandbytes==0.47.0 sentencepiece==0.2.1
& ./.venv-gpu/Scripts/python.exe run_authors_checkpoint.py
```

Команда скачивает около 13,5 ГБ весов и применяет NF4 для экономии VRAM. Этот сценарий подготовлен, но не проверен полным запуском в данном проекте. При нехватке памяти нужно больше VRAM или CPU offload. NF4 меняет численную точность, поэтому результаты нельзя напрямую приравнивать к BF16 результатам статьи. Даже если веса помещаются, длинный KV cache требует отдельной памяти. У Llama-2-7B BF16 KV cache на 8192 токенах занимает около 4 ГиБ.

## Метрика локальной абляции

Перплексия: `exp(NLL)`, где NLL вычисляет causal LM loss по следующему токену. Для каждой длины 256/2048/4096 берется один и тот же префикс test split. Это одна последовательность на длину, не полноценная оценка датасета. Между длинами меняется набор целевых токенов, поэтому длины нельзя трактовать как строго парные измерения качества. Сравнения режимов внутри одной длины используют одинаковые токены. Train и test разделены. Обучение: 16 шагов, batch 1, 256 токенов, 4096 предъявленных токенов, AdamW, LR 3e-5, seed 42, без QKNorm.

## Источники и версии

- Статья: https://arxiv.org/abs/2512.12167v1
- Официальный код: https://github.com/SakanaAI/DroPE
- Git commit: `260a8bb9f4fda0673266ddfee7668915aade8b33`
- Опубликованный checkpoint: https://huggingface.co/SakanaAI/Llama-2-7b-hf-DroPE
- Checkpoint revision: `02a73325b6ed83e2fb951d68b41e50b47a258785`
- Base SmolLM: https://huggingface.co/HuggingFaceTB/SmolLM-360M
- SmolLM revision: `59f7ef243ee09a72cbc14cb054393a3e3b771d41`
- Данные: https://huggingface.co/datasets/Salesforce/wikitext

Файлы в `vendor/` принадлежат авторам оригинального проекта. Они сохранены с атрибуцией для учебного воспроизведения. Исходный репозиторий в данном snapshot не содержит отдельного LICENSE; этот проект не назначает авторскому коду новую лицензию. У модели Llama-2 свои лицензионные условия. Ограничения и фактические результаты изложены в отчете.
