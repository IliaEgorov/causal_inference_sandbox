# Оценка эффекта изменения в трафике

`causal_traffic_impact.py` отвечает на вопрос: «Каким был инкрементальный эффект изменения на конкретный трафик?» Метод — Difference-in-Differences (DiD): изменение метрики treatment-группы минус одновременное изменение контрольной группы.

## Когда применять

Подходит для, например, увеличения бюджета Google Ads на 20%, если изменённые кампании можно выделить как `traffic_type=media, traffic_type_group=ppc`, а остальные когорты не затронуты. Укажите первый полный день, когда изменение было включено.

```bash
pip install -r requirements.txt causalpy matplotlib
python causal_traffic_impact.py cohort_daily.csv \
  --change-date 2026-09-12 \
  --treated traffic_type=media traffic_type_group=ppc \
  --outcome arpu --out-dir google_budget_impact
```

Для изменения welcome bonus, которое увидел только affiliate-трафик:

```bash
python causal_traffic_impact.py cohort_daily.csv \
  --change-date 2026-09-12 --treated traffic_type=affiliate \
  --outcome ftd_rate --out-dir welcome_bonus_impact
```

`--outcome` поддерживает `arpu`, `ftd_rate` и `adpu`.

## Результаты

- `did_effect.csv` — ATT, 95% интервал, p-value и эффект относительно baseline.
- `pretrend_check.csv` — тренд разрыва treatment/control до изменения.
- `treated_vs_control.png` — дневной график для визуальной проверки.
- `causalpy_did_plot.png` — нативная визуализация CausalPy, если библиотека установлена.

## Критически важное допущение

Контрольные когорты должны быть **не затронуты** изменением и иметь до изменения похожий тренд с treatment. Если welcome bonus изменили для всех источников, остальные источники не являются контролем: DiD здесь не доказывает причинность. Нужны внешний контроль (другая страна/бренд, не получившие бонус), эксперимент или существенно более осторожный interrupted time-series анализ.

Скрипт строит pre-trend проверку, но она не заменяет бизнес-проверку: высокий p-value не доказывает параллельность трендов.