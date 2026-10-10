# DroPE: объяснение статьи и частичное воспроизведение

Материалы по статье **Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings** (Yoav Gelberg, Koshi Eguchi, Takuya Akiba, Edoardo Cetin, 2025).

## Материалы для сдачи

| Требуемый материал | Файл |
| --- | --- |
| Презентация с объяснением статьи, 8 кратких слайдов | [DroPE_presentation.pptx](DroPE_presentation.pptx) |
| Отчет о воспроизведении, 5 страниц | [DroPE_report.pdf](DroPE_report.pdf) |
| Код с инструкциями по запуску | [Папка code](code/) · [Инструкция](code/README.md) |
| Архив кода, конфигураций и логов | [DroPE_code.zip](DroPE_code.zip) |

Презентация и отчет написаны на русском. Источники иллюстраций и опубликованных результатов указаны в отчете и заметках слайдов.

## Что выполнено локально

Использован **исходный код attention авторов** из [SakanaAI/DroPE](https://github.com/SakanaAI/DroPE), без изменений. На pretrained **SmolLM-360M** сравнивались три состояния: исходная модель с RoPE, модель после удаления RoPE без обучения и модель после короткой рекалибровки. Запуск выполнен в Windows на CPU, Python 3.12.14, PyTorch 2.8.0+cpu, Transformers 4.56.2.

Проверки маленького Transformer подтвердили сохранение причинной маски и равномерное attention NoPE на одинаковых токенах. Рекалибровка SmolLM: 16 шагов, batch size 1, длина 256, всего 4096 предъявленных токенов WikiText-2. Обучающие и тестовые данные разделены.

### Фактические локальные результаты

Перплексия, меньше лучше. Для каждой длины оценена одна тестовая последовательность. Внутри одной длины все режимы используют одинаковый текст.

| Состояние | 256 токенов | 2048 токенов | 4096 токенов |
| --- | ---: | ---: | ---: |
| Исходная RoPE | 22,19 | 9,71 | 63,16 |
| Удаление RoPE без обучения | 270,09 | 491,66 | 541,53 |
| После 16 шагов рекалибровки | 134,97 | 258,59 | 296,24 |

Перплексия снизилась после короткой рекалибровки, но качество исходной модели не восстановилось. Такой бюджет несопоставим с миллиардами токенов в статье. Успешная обработка 4096 токенов сама по себе не доказывает эффективное использование длинного контекста.

**Это частичное воспроизведение.** Полные LongBench/RULER эксперименты и опубликованные benchmark scores локально не воспроизведены. Для готового Llama-2-7B-DroPE проверен доступ к конфигурации, индексу и началу весов. Попытка GPU-запуска в установленной CPU-среде остановилась на проверке CUDA. Полный инференс этой модели не выполнен. Дополнительный сценарий CUDA/NF4 приложен, но не проверен полным запуском.

### Где находятся доказательства запуска

- [Локальные измерения](code/results/local.json) и [лог эксперимента](code/results/run.log).
- [Структурные проверки](code/results/structural.json).
- [Проверка доступа к checkpoint авторов](code/results/authors_probe.json).
- [Лог попытки запуска 7B](code/results/authors_launch_attempt.log).
- [Точные версии окружения](code/results/environment.lock.txt).
- [SHA-256 скачанных ресурсов](code/results/assets_sha256.json) и [происхождение авторского кода](code/results/provenance.json).
- [Опубликованные числа из таблиц статьи](code/results/authors_results.json). Они получены **авторами статьи**, а не в локальном эксперименте.

## Как запустить

Установите Python 3.12, клонируйте репозиторий и откройте PowerShell в папке `code`:

```powershell
git clone https://github.com/nZiben/drope-reproduction.git
cd drope-reproduction/code
./setup_cpu.ps1
& ./.venv/Scripts/python.exe download_assets.py --cache cache
& ./.venv/Scripts/python.exe reproduce.py --cache cache --out results/local.json --steps 16 --seq-len 256 --device cpu
```

Только структурные проверки, без скачивания pretrained весов:

```powershell
& ./.venv/Scripts/python.exe reproduce.py --structural-only --out results/structural.json
```

Полное описание метрики, зависимостей, загрузок и дополнительного GPU-сценария: [code/README.md](code/README.md).

Веса моделей и виртуальные окружения не включены в Git. Скрипт скачивает модель и данные отдельно с фиксацией revisions и контрольных сумм.

## Источники и атрибуция

- [Статья, arXiv:2512.12167v1](https://arxiv.org/abs/2512.12167v1).
- [Код авторов, commit 260a8bb9f4fda0673266ddfee7668915aade8b33](https://github.com/SakanaAI/DroPE/tree/260a8bb9f4fda0673266ddfee7668915aade8b33).
- [Опубликованная модель Llama-2-7B-DroPE](https://huggingface.co/SakanaAI/Llama-2-7b-hf-DroPE).
- [Базовая SmolLM-360M](https://huggingface.co/HuggingFaceTB/SmolLM-360M).
- [Данные WikiText](https://huggingface.co/datasets/Salesforce/wikitext).
- [Объяснение и рисунки авторов](https://pub.sakana.ai/DroPE/).

Файлы `code/vendor/` принадлежат авторам оригинального проекта. Они сохранены с атрибуцией и исходным README. В использованном snapshot репозитория нет отдельного LICENSE; данная работа не назначает авторскому коду новую лицензию. Модели и данные сохраняют собственные условия использования.

```bibtex
@article{gelberg2025extending,
  title={Extending the Context of Pretrained LLMs by Dropping Their Positional Embeddings},
  author={Gelberg, Yoav and Eguchi, Koshi and Akiba, Takuya and Cetin, Edoardo},
  journal={arXiv preprint arXiv:2512.12167},
  year={2025}
}
```
