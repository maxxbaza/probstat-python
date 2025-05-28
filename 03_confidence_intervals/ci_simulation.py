import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Заданные параметры
mu = 5.1         # Истинное среднее
sigma = 2.6      # Известное стандартное отклонение
n = 26           # Размер выборки
alpha = 0.05     # Уровень значимости
num_samples = 100

# z-критическое значение для 95% интервала
z_crit = stats.norm.ppf(1 - alpha / 2)

intervals = []
contains_mu = []

# Генерация выборок и построение доверительных интервалов
for _ in range(num_samples):
    sample = np.random.normal(mu, sigma, n)
    sample_mean = np.mean(sample)
    margin = z_crit * sigma / np.sqrt(n)
    ci_lower = sample_mean - margin
    ci_upper = sample_mean + margin
    intervals.append((ci_lower, ci_upper))
    contains_mu.append(ci_lower <= mu <= ci_upper)

# Подсчёт числа интервалов, содержащих 5.1
num_containing = sum(contains_mu)
print(f"Интервалов, содержащих 5.1: {num_containing} из {num_samples}")

# Рисуем график доверительных интервалов
plt.figure(figsize=(8, 6))
for i, ((low, high), hit) in enumerate(zip(intervals, contains_mu)):
    color = 'black' if hit else 'red'
    plt.plot([low, high], [i, i], color=color)
plt.axvline(mu, color='blue', linestyle='--', label='μ = 5.1')
plt.xlabel("Confidence interval")
plt.ylabel("Sample index")
plt.title("95% Confidence Intervals for Mean (σ known)")
plt.legend()
plt.grid(True)
plt.show()


n_simulations = 10000

margin = z_crit * sigma / np.sqrt(n)

# Счётчик попаданий
count = 0

for _ in range(n_simulations):
    sample = np.random.normal(mu, sigma, n)
    sample_mean = np.mean(sample)
    ci_lower = sample_mean - margin
    ci_upper = sample_mean + margin
    if ci_lower <= mu <= ci_upper:
        count += 1

# Результат
print(f"Число интервалов, содержащих 5.1: {count}")
print(f"Доля интервалов, содержащих 5.1: {count / n_simulations:.4f}")
