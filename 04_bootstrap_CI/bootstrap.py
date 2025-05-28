import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

# Параметры задачи
mu = 5.1
sigma = 2.6
n = 26
alpha = 0.05
n_bootstrap = 10000

# 1. Одна выборка
np.random.seed(0)
original_sample = np.random.normal(mu, sigma, n)
original_mean = np.mean(original_sample)

# 2. Bootstrap: ресемплируем выборку 10_000 раз с возвращением
bootstrap_means = []
for _ in range(n_bootstrap):
    resample = np.random.choice(original_sample, size=n, replace=True)
    bootstrap_means.append(np.mean(resample))

# 3. 95% бутстрэп-доверительный интервал (percentile method)
ci_lower = np.percentile(bootstrap_means, 100 * alpha / 2)
ci_upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
print(f"Bootstrap 95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")

# 4. Аналитический ДИ (с известной σ)
z_crit = stats.norm.ppf(1 - alpha / 2)
margin = z_crit * sigma / np.sqrt(n)
analytical_ci = (original_mean - margin, original_mean + margin)
print(f"Аналитический 95% CI: ({analytical_ci[0]:.3f}, {analytical_ci[1]:.3f})")

# 5. Визуализация
plt.hist(bootstrap_means, bins=40, density=True, color='lightgray', edgecolor='black')
plt.axvline(ci_lower, color='red', linestyle='--', label='Bootstrap CI lower')
plt.axvline(ci_upper, color='red', linestyle='--', label='Bootstrap CI upper')
plt.axvline(original_mean, color='blue', linestyle='-', label='Sample mean')
plt.axvline(analytical_ci[0], color='green', linestyle=':', label='Analytic CI lower')
plt.axvline(analytical_ci[1], color='green', linestyle=':', label='Analytic CI upper')
plt.title("Bootstrap Distribution of the Sample Mean")
plt.legend()
plt.grid(True)
plt.show()
