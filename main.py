import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Дані для акцій A1, A2 та A3
expected_returns = np.array([0.10, 0.30, 0.45])  # Сподівані норми прибутку
std_devs = np.array([0.00, 0.10, 0.15])  # Стандартні відхилення
correlations = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, -0.8],
    [0.0, -0.8, 1.0]
])

# Обчислення коваріаційної матриці
cov_matrix = np.outer(std_devs, std_devs) * correlations

# Функція для обчислення ризику і прибутку портфеля
def portfolio_performance(weights, returns, cov_matrix):
    expected_return = np.dot(weights, returns)
    risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return expected_return, risk

# Побудова множини ефективних ПЦП (пункт а)
def generate_efficient_frontier(returns, cov_matrix, num_portfolios=100):
    num_assets = len(returns)
    results = {'returns': [], 'risks': [], 'weights': []}
    
    for _ in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(num_assets), size=1)[0]
        expected_return, risk = portfolio_performance(weights, returns, cov_matrix)
        results['returns'].append(expected_return)
        results['risks'].append(risk)
        results['weights'].append(weights)
    
    return results

# Побудова ефективної множини (пункт а)
frontier = generate_efficient_frontier(expected_returns, cov_matrix)

# Відображення ефективної множини
plt.scatter(frontier['risks'], frontier['returns'], c=frontier['returns'], cmap='viridis')
plt.colorbar(label='Очікуваний прибуток')
plt.xlabel('Ризик (стандартне відхилення)')
plt.ylabel('Очікуваний прибуток')
plt.title('Множина ефективних ПЦП')
plt.show()

# Функція для знаходження портфеля з заданим очікуваним прибутком
def find_portfolio_for_return(target_return):
    num_assets = len(expected_returns)
    
    # Обмеження: сума ваг повинна бути рівною 1, і очікуваний прибуток повинен бути заданим
    constraints = (
        {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        {'type': 'eq', 'fun': lambda weights: np.dot(weights, expected_returns) - target_return}
    )
    bounds = [(0, 1) for _ in range(num_assets)]
    
    # Мінімізуємо ризик при заданому очікуваному прибутку
    result = minimize(lambda weights: portfolio_performance(weights, expected_returns, cov_matrix)[1],
                      x0=np.ones(num_assets) / num_assets,
                      constraints=constraints, bounds=bounds)
    
    if result.success:
        weights = result.x
        expected_return, risk = portfolio_performance(weights, expected_returns, cov_matrix)
        return weights, expected_return, risk
    else:
        return None

# Пошук портфеля з очікуваним прибутком 25% (пункт б)
portfolio_25 = find_portfolio_for_return(0.25)
if portfolio_25:
    weights_25, expected_return_25, risk_25 = portfolio_25
    print(f"(б) Портфель з очікуваним прибутком 25%:\nВаги: {weights_25}\nОчікуваний прибуток: {expected_return_25:.2%}\nРизик: {risk_25:.2%}")
else:
    print("(б) Не вдалося знайти портфель з очікуваним прибутком 25%.")

# Пошук портфеля з очікуваним прибутком 40% (пункт в)
portfolio_40 = find_portfolio_for_return(0.40)
if portfolio_40:
    weights_40, expected_return_40, risk_40 = portfolio_40
    print(f"(в) Портфель з очікуваним прибутком 40%:\nВаги: {weights_40}\nОчікуваний прибуток: {expected_return_40:.2%}\nРизик: {risk_40:.2%}")
else:
    print("(в) Не вдалося знайти портфель з очікуваним прибутком 40%.")

# Виведення рівнів ризику для цих ПЦП (пункт г)
print(f"(г) Рівні ризиків:")
print(f"Ризик портфеля з очікуваним прибутком 25%: {risk_25:.2%}")
if portfolio_40:
    print(f"Ризик портфеля з очікуваним прибутком 40%: {risk_40:.2%}")
