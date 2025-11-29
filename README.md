# 🐉 Tiamat - AI-Powered Gold Trading System

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python)
![LightGBM](https://img.shields.io/badge/LightGBM-ML-green?style=for-the-badge)
![MetaTrader 5](https://img.shields.io/badge/MetaTrader-5-orange?style=for-the-badge)
![ASP.NET Core](https://img.shields.io/badge/ASP.NET-Core-purple?style=for-the-badge&logo=dotnet)
![License](https://img.shields.io/badge/License-Educational-red?style=for-the-badge)

**Автоматизирана система за алгоритмична търговия на злато (XAUUSD), използваща машинно обучение**

*🏆 Проект, участвал в няколко олимпиади по информатика и информационни технологии*

---

## 📖 Съдържание

- [За проекта](#-за-проекта)
- [Резултати от Backtesting](#-резултати-от-backtesting)
- [Архитектура](#-архитектура)
- [Технологии](#-технологии)
- [Модули](#-модули)
- [Бърз старт](#-бърз-старт)
- [Disclaimer](#️-disclaimer)

---

## 🎯 За проекта

**Tiamat** е цялостна система за автоматизирана търговия на злато (XAUUSD), която комбинира:

- **Машинно обучение** (LightGBM) за предсказване на ценови движения
- **Bayesian оптимизация** (Optuna) за намиране на оптимални хиперпараметри
- **Технически анализ** с над 80+ индикатора и характеристики
- **Real-time търговия** чрез MetaTrader 5
- **Уеб мониторинг** с ASP.NET Core

### Как работи?

1. **Обучение на модела** - Системата анализира исторически данни от 2018-2023 г. и обучава LightGBM класификатор да предсказва ценови движения в долари
2. **Генериране на сигнали** - На база на вероятностите от модела се генерират BUY/SELL сигнали с динамични Stop Loss и Take Profit нива
3. **Автоматично изпълнение** - Сигналите се изпращат към MetaTrader 5 за автоматично изпълнение на сделки
4. **Управление на риска** - Вградена система за управление на риска, избягване на новини и времеви филтри

---

## 📊 Резултати от Backtesting

### 2024 Test Period (Out-of-Sample)

| Метрика | Стойност |
|---------|----------|
| **Return** | 52.72% |
| **Sharpe Ratio** | 1.71 |
| **Max Drawdown** | -6.35% |
| **Win Rate** | 53.85% |
| **Profit Factor** | 1.70 |
| **Calmar Ratio** | 8.06 |
| **Sortino Ratio** | 7.37 |
| **Брой сделки** | 104 |

### 2023 Validation Period

| Метрика | Стойност |
|---------|----------|
| **Return** | 11.08% |
| **Sharpe Ratio** | 1.42 |
| **Max Drawdown** | -3.30% |
| **Win Rate** | 60.00% |
| **Profit Factor** | 2.42 |
| **Calmar Ratio** | 3.29 |
| **Sortino Ratio** | 4.08 |
| **Брой сделки** | 15 |

> ⚠️ **Важно:** Тези резултати са от backtesting и не гарантират бъдеща доходност. Реалната търговия носи риск от загуба на капитал.

---

## 🏗 Архитектура

```
┌─────────────────────────────────────────────────────────────────────┐
│                        TIAMAT СИСТЕМА                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────┐  │
│  │  TiamatOffline   │    │  TiamatOnline    │    │   MetaTrader │  │
│  │                  │    │                  │    │      5       │  │
│  │  • Обучение на   │    │  • Live сървър   │    │              │  │
│  │    модел         │───▶│  • Feature       │◀──▶│  • Sender    │  │
│  │  • Оптимизация   │    │    engineering   │    │    Script    │  │
│  │  • Backtesting   │    │  • Сигнали       │    │  • DLL       │  │
│  │                  │    │                  │    │    Executor  │  │
│  └──────────────────┘    └────────┬─────────┘    └──────────────┘  │
│                                   │                                 │
│                          ┌────────▼─────────┐                       │
│                          │   ASP.NET Core   │                       │
│                          │   Web Dashboard  │                       │
│                          │                  │                       │
│                          │  • Мониторинг    │                       │
│                          │  • Статистики    │                       │
│                          │  • История       │                       │
│                          └──────────────────┘                       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Комуникационни канали

| Порт | Протокол | Описание |
|------|----------|----------|
| 8020 | HTTP | Получаване на ценови данни от MetaTrader |
| 8000 | HTTP/API | REST API за команди и уеб интерфейс |
| 12345 | TCP Socket | Изпращане на търговски сигнали към DLL |

---

## 🛠 Технологии

### Machine Learning & Data Science

- **LightGBM** - Gradient boosting за класификация
- **Optuna** - Bayesian оптимизация на хиперпараметри
- **TA-Lib / ta** - Библиотека за технически индикатори
- **Pandas & NumPy** - Обработка на данни
- **Scikit-learn** - ML utilities
- **Backtesting.py** - Симулация на стратегии

### Backend & Infrastructure

- **QuestDB** - Time-series база данни за бързи заявки
- **Flask** - HTTP/API сървър
- **ASP.NET Core** - Уеб приложение за мониторинг
- **Python 3.12** - Основен runtime

### Trading Platform

- **MetaTrader 5** - Търговска платформа
- **MQL5** - Expert Advisors за MT5
- **C++ DLL** - Socket комуникация с Python сървъра

---

## 📁 Модули

### 📂 TiamatOffline

Модул за обучение и оптимизация на ML модела.

```
TiamatOffline/
├── hyperparameters.py           # Optuna оптимизация + backtesting
├── dataset_without_indicators.py # Импорт на данни в QuestDB
├── news_tz.py                   # Обработка на новинарски данни
├── requirements.txt             # Python зависимости
├── XAUUSD_M1_RAW.csv           # Исторически данни (1M timeframe)
└── market_news.csv              # Данни за пазарни новини
```

**Ключови характеристики:**

- 80+ технически индикатора (EMA, SMA, MACD, RSI, Bollinger Bands, Ichimoku, ADX, Donchian и др.)
- Автоматично подрязване на характеристики (feature pruning)
- Walk-forward validation (Train: 2018-2022, Val: 2023, Test: 2024)
- Динамично изчисляване на SL/TP на база вероятностни bins

### 📂 TiamatOnline

Модул за real-time търговия.

```
TiamatOnline/
├── livepipeline.py          # Основен Python сървър
├── model.pkl                # Обучен модел
├── MetaTrader MQL/
│   ├── XAUUSD_SENDER.mq5    # Изпраща данни към Python
│   └── XAUUSD_DLL.mq5       # Изпълнява сделки
├── SignalProvider/          # C++ DLL проект
└── Tiamat/                  # ASP.NET Core уеб приложение
    ├── Tiamat.WebApp/
    ├── Tiamat.Core/
    ├── Tiamat.DataAccess/
    └── Tiamat.Models/
```

**Ключови характеристики:**

- Филтриране на сделки около новинарски събития (±20 мин)
- Избягване на maintenance период (22:00-00:00)
- Динамично управление на размера на позицията
- Криптирана комуникация с XOR cipher

---

## 🚀 Бърз старт

### Предварителни изисквания

- Python 3.12+
- QuestDB
- MetaTrader 5 (за live търговия)
- Visual Studio (за компилация на DLL)

### 1. Инсталация на зависимости

```bash
cd TiamatOffline
pip install -r requirements.txt
```

### 2. Настройка на QuestDB

1. Изтеглете QuestDB от https://questdb.com/download/
2. Стартирайте `questdb.exe`
3. Достъпете http://localhost:9000

### 3. Импортиране на данни

```bash
python news_tz.py
python dataset_without_indicators.py
```

### 4. Обучение на модел

```bash
python hyperparameters.py
```

> ⏱️ **Времетраене:** ~20+ часа за 50 trials

### 5. Стартиране на live сървър

```bash
cd TiamatOnline
python livepipeline.py
```


---

## 📈 Feature Engineering

Системата използва богат набор от технически индикатори:

| Категория | Индикатори |
|-----------|------------|
| **Trend** | EMA (9, 21), SMA (50, 200), MACD, Ichimoku Cloud, Parabolic SAR, ADX |
| **Momentum** | RSI (7, 14), ROC (5, 10, 20), Fisher Transform |
| **Volatility** | Bollinger Bands, ATR, Donchian Channel (20, 55) |
| **Volume** | OBV, Volume Ratio, Volume-Price Correlation |
| **Session** | Asian, London, NY sessions, London-NY overlap |
| **Statistical** | Z-scores (10, 20, 50), Momentum (5d, 20d), Volatility ratios |

---

## ⚠️ Disclaimer

**ВАЖНО:** Този проект е създаден с **образователна цел** и за участие в олимпиади.

- 🚫 **Не използвайте** тази система за реална търговия без задълбочено разбиране на рисковете
- 📉 Търговията с финансови инструменти носи **значителен риск от загуба на капитал**
- 📊 Резултатите от backtesting **не гарантират** бъдеща доходност
- 🔬 Past performance is **not indicative** of future results
- 💡 Винаги търгувайте само с пари, които **можете да си позволите да загубите**

---

## 📜 Лиценз

Този проект е публикуван с образователна цел. Използването за комерсиални цели или реална търговия е изцяло на собствен риск.

---

## 👤 Автор

Създаден като проект за олимпиада по информационни технологии от Антон Донев.

---

⭐ **Ако проектът ви е полезен, не забравяйте да дадете звезда!** ⭐
