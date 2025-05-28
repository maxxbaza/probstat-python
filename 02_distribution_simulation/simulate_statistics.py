import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Параметры
mu = 5
sigma = 1
n = 20
num_samples = 10000

# Генерация выборок
samples = np.random.normal(mu, sigma, (num_samples, n))

# Вычисление статистик
sample_means = np.mean(samples, axis=1)
sample_vars = np.var(samples, axis=1, ddof=1)  # Выборочная дисперсия
sample_max = np.max(samples, axis=1)

# Теоретические плотности
x_mean = np.linspace(min(sample_means), max(sample_means), 100)
density_mean = stats.norm.pdf(x_mean, loc=mu, scale=sigma / np.sqrt(n))

x_var = np.linspace(min(sample_vars), max(sample_vars), 100)
density_var = stats.chi2.pdf(x_var * (n - 1) / sigma**2, df=n-1) * (n - 1) / sigma**2

x_max = np.linspace(min(sample_max), max(sample_max), 100)
F_x = stats.norm.cdf(x_max, loc=mu, scale=sigma)  # Функция распределения
f_x = stats.norm.pdf(x_max, loc=mu, scale=sigma)  # Плотность распределения
density_max = n * (F_x ** (n - 1)) * f_x  # Теоретическая плотность для максимального элемента

# Построение гистограмм и плотностей
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Гистограмма для выборочного среднего
axes[0].hist(sample_means, bins=50, density=True, alpha=0.6, color='b')
axes[0].plot(x_mean, density_mean, 'r-', label='Теоретическая плотность')
axes[0].set_title("Распределение выборочного среднего")
axes[0].legend()

# Гистограмма для выборочной дисперсии
axes[1].hist(sample_vars, bins=50, density=True, alpha=0.6, color='g')
axes[1].plot(x_var, density_var, 'r-', label='Теоретическая плотность')
axes[1].set_title("Распределение выборочной дисперсии")
axes[1].legend()

# Гистограмма для максимума выборки
axes[2].hist(sample_max, bins=50, density=True, alpha=0.6, color='m')
axes[2].plot(x_max, density_max, 'r-', label='Теоретическая плотность')
axes[2].set_title("Распределение максимального элемента")
axes[2].legend()

plt.show()
