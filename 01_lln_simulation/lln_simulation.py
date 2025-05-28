import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binom, cauchy

# Параметры задачи
m_values = np.logspace(0, 6, num=50, dtype=int)  # Значения m от 1 до 10^6 (логарифмическая шкала)
num_runs = 10 # Количество запусков для усреднения

# Сохранение результатов
means_a = np.zeros((num_runs, len(m_values)))
means_b = np.zeros((num_runs, len(m_values)))

# Проведение экспериментов
for run in range(num_runs):
    for i, m in enumerate(m_values):
        # Случай (a): Биномиальное распределение
        Xi_a = binom.rvs(n=1, p=0.5, size=m)
        means_a[run, i] = np.mean(Xi_a)

        # Случай (b): Распределение Коши
        Xi_b = cauchy.rvs(size=m)
        means_b[run, i] = np.mean(Xi_b)

# Теоретическое среднее для случая (a)
theoretical_mean_a = 0.5

# Построение графиков
plt.figure(figsize=(12, 6))

# Случай (a)
for run in range(num_runs):
    plt.plot(m_values, means_a[run, :], color='blue', alpha=0.5)
plt.axhline(theoretical_mean_a, color='black', linestyle='--', label='Теоретическое среднее (a)')
plt.xscale('log')
plt.title('График выборочного среднего для случая (a): Биномиальное распределение')
plt.xlabel('m')
plt.ylabel('Выборочное среднее')
plt.legend()
plt.grid(True)
plt.show()

# Случай (b)
plt.figure(figsize=(12, 6))
for run in range(num_runs):
    plt.plot(m_values, means_b[run, :], color='red', alpha=0.5)
plt.xscale('log')
plt.title('График выборочного среднего для случая (b): Распределение Коши')
plt.xlabel('m')
plt.ylabel('Выборочное среднее')
plt.grid(True)
plt.show()
