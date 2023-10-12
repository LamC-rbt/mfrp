| Название| Авторы | Основные идеи | Ссылки |
| :---: | :---: | :---: | :---: |
| A convex version of multivariate adaptive regression splines | DL Martinez, DT Shih, VCP Chen, SB Kim | Использование выпуклого расширения классического алгоритма для улучшения качества на выпуклых задачах | [text](https://www.sciencedirect.com/science/article/abs/pii/S0167947314002291)|
| Fast knot optimization for multivariate adaptive regression splines using hill climbing methods | X Ju, VCP Chen, JM Rosenberger, F Liu | Предложен подход с использованием классических методов Hill Climbing и Hill Climbing с априорным знанием по уменьшению MSE. Уменьшение времени обучения модели до 80%| [text](https://www.sciencedirect.com/science/article/abs/pii/S0957417421000063) |
| Multivariate adaptive regression splines | Friedman J. H. | Предложена основная модель MARS для решения задачи регрессии. Произведены эксперементы, показывающие ее эффективность | [text](https://projecteuclid.org/journals/annals-of-statistics/volume-19/issue-1/Multivariate-Adaptive-Regression-Splines/10.1214/aos/1176347963.full)|
| CMARS: a new contribution to nonparametric regression with multivariate adaptive regression splines supported by continuous optimization| GW Weber, I Batmaz, G Köksal, P Taylan | Модификация MARS, позволяющая использовать методы непрерывной оптимизации. Добавлена регуляризация Тихонова | [text](https://www.tandfonline.com/doi/full/10.1080/17415977.2011.624770) |
| RCMARS: Robustification of CMARS with different scenarios under polyhedral uncertainty set | A Özmen, GW Weber, İ Batmaz, E Kropat | Рассматривается модифицированная модель CMARS, которая лучше обрабатывает более шумные данные | [text](https://www.sciencedirect.com/science/article/abs/pii/S1007570411001912) |
| Solar radiation forecasting using MARS, CART, M5, and random forest model | R Srivastava, AN Tiwari, VK Giri | Сравнивается предсказательная способность различных моделей на реальных данных. Рассматриваются достоинства и недостатки каждого метода. MARS показал себя одним из лучших |[text](https://www.cell.com/heliyon/pdf/S2405-8440(19)36352-2.pdf)|
| Parallel MARS Algorithm Based on B-splines | Sergey Bakin, Markus Hegland, Michael R. Osborne  | Предложена модификация алгоритма MARS, основанная на B-splines. Так же предлагается способ параллелизации модели для ускорения обучения |[text](https://link.springer.com/article/10.1007/PL00022715)|
| Identification of gender differences in the factors influencing shoulders, neck and upper limb MSD by means of multivariate adaptive regression splines (MARS)| NB Serrano, AS Sánchez, FS Lasheras| Показана эффективность адгоритма MARS для выявления зависимостей и решения прикладной медицинской задачи | [text](https://www.sciencedirect.com/science/article/abs/pii/S0003687019301942) |
| Modification of Multivariate Adaptive Regression Spline (MARS)| SDP Yasmirullah, BW Otok | Расширение алгоритма для более эффективной работы с категориальными данными вида счетчиков. Добавление к классическому MARS пуассоновской регрессии | [text](https://iopscience.iop.org/article/10.1088/1742-6596/1863/1/012078/meta) |
| Model selection in multivariate adaptive regression splines (MARS) using information complexity as the fitness function | E Kartal Koc, H Bozdogan | В статье предложен новый информационный критерий (ICOMP), контролирующий сложность модели. Достигнута более гибкая настройка между сложностью модели и качеством пресказаний |[text](https://link.springer.com/article/10.1007/s10994-014-5440-5) |
| Analysis of macro nutrient related growth responses using multivariate adaptive regression splines | M Akin, SP Eyduran, E Eyduran, BM Reed | Показана высокая точность предсказаний в рассматриваемой области | [text](https://www.sciencedirect.com/science/article/abs/pii/S0022169418301434) |
| A comparison of regression trees, logistic regression, generalized additive models, and multivariate adaptive regression splines for predicting AMI mortality | Peter C. Austin | Сравнение классических алгоритмов машинного обучения. Показано превосходство MARS в реальных задачах | [text](https://onlinelibrary.wiley.com/doi/abs/10.1002/sim.2770) |
| Polynomial splines and their tensor products in extended linear modeling | CJ Stone, MH Hansen, C Kooperberg | Предложено новое построение итоговых базисных функций. Модель становится более интерпретируемой. В модели находятся меньше функций появившихся в результате переобучения | [text](https://projecteuclid.org/journals/annals-of-statistics/volume-25/issue-4/Polynomial-splines-and-their-tensor-products-in-extended-linear-modeling/10.1214/aos/1031594728.short) |
| Bayesian MARS | D. G. T. DENISON, B. K. MALLICK, A. F. M. SMITH  | Рассмотрен Баесовский подход к построению MARS. Возрастает интерпертируемость и точность предсказаний. Используются марковские цепи и метод монте-карло | [text](https://link.springer.com/article/10.1023/A:1008824606259) |
| MARS via LASSO | Dohyeong Ki, Billy Fang, Adityanand Guntuboyina  | Реализация LASSO в MARS. Используются выпуклые функции для приближения | [text](https://arxiv.org/abs/2111.11694) |

# Планирование экспериментов

* Цель: Улучшить существующие алгоритмы MARS, сравнить качество с ансамблевыми подходами и бэггингом со случайными поворотами признаков.

* Данные: Датасеты для задачи регрессии. Небольшие можно взять, например, из sklearn (типа diabets и т.п.)

* Проведение экспериментов: Обучение на предобработанном датасете различных моделей, подбор гиперпараметров. Сравнение их на отложенной и обучающей выборках. Рассмотреть эффективность поворотов признаков. Построить графики функции потерь (в том числе и для различных значений размеров подпространств в бэггинге).

* Планирование блока с экспериментами.
    1. Описание алгоритмов
    2. Описание наборов данных, на которых тестировался алгоритм.
    3. Результаты экспериментов с графиками.
    4. Обсуждения и выводы.

* Ожидаемые результаты: преимущество бэггинга со случайными поворотами признаков по функции потерь как на обучающей. так и на тестовой выборках.
