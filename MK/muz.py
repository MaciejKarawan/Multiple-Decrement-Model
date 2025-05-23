import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def setup_plotting_style():
    plt.rc("figure",
           autolayout=True,
           figsize=(11.69, 8.27),
           titlesize=10,
           titleweight='bold')
    plt.rc("axes",
           labelweight="bold",
           labelsize="large",
           titleweight="bold",
           titlesize=16,
           titlepad=10,
           titlelocation='center')

    plt.rcParams['font.family'] = 'DejaVu Sans'
    sns.set_theme(style='whitegrid', context='paper', palette='dark')


### Model A
def model_A_survival(t, mu):
    return np.exp(-mu * t)


### Model B
def model_B_survival(t, x, params):
    A, B, c = params
    return np.exp(-A*t - (B/np.log(c))*(c**(x+t) - c**x))


### Funkcja celu dla modelu A (do minimalizacji)
def objective_A(mu, ages, survival_values):
    predicted = np.array([model_A_survival(t, mu) for t in ages])
    return np.sum((predicted - survival_values)**2)


# Funkcja celu dla modelu B (do minimalizacji)
def objective_B(params, x0, ages, survival_values):
    predicted = np.array([model_B_survival(t, x0, params) for t in ages])
    return np.sum((predicted - survival_values)**2)


# Kalibracja modeli
def calibrate_models(data, sex):
    ages = data['age'].values
    l_values = data['l'].values
    survival_values = l_values / l_values[0]

    # Kalibracja modelu A
    result_A = minimize(objective_A, x0=[0.01], args=(ages, survival_values),
                        bounds=[(0.0001, 0.1)], method='L-BFGS-B')
    mu_optimal = result_A.x[0]

    # Kalibracja modelu B

    # Początkowe wartości parametrów [A, B, c]
    initial_params = [0.0001, 0.00001, 1.1]
    bounds = [(0.00001, 0.01), (0.000001, 0.001), (1.01, 1.2)]

    result_B = minimize(objective_B, x0=initial_params,
                        args=(0, ages, survival_values),
                        bounds=bounds, method='L-BFGS-B')
    A_optimal, B_optimal, c_optimal = result_B.x

    print(f"\nWyniki kalibracji dla {sex}:")
    print(f"Model A: μ = {mu_optimal:.8f}")
    print(f"Model B: A = {A_optimal:.8f}, B = {B_optimal:.8f}, c = {c_optimal:.8f}")

    return {
        'A': {'mu': mu_optimal},
        'B': {'A': A_optimal, 'B': B_optimal, 'c': c_optimal}
    }


# Wizualizacja dopasowania modeli
def visualize_models(data, params, sex, save_path=None):
    ages = data['age'].values
    l_values = data['l'].values
    survival_values = l_values / l_values[0]

    # Przewidywania modelu A
    mu = params['A']['mu']
    predicted_A = np.array([model_A_survival(t, mu) for t in ages])

    # Przewidywania modelu B
    A, B, c = params['B']['A'], params['B']['B'], params['B']['c']
    predicted_B = np.array([model_B_survival(t, 0, [A, B, c]) for t in ages])

    # Wizualizacja
    plt.figure(figsize=(10, 6))
    plt.plot(ages, survival_values, 'o', markersize=3, label='TTŻ 2023')
    plt.plot(ages, predicted_A, '-', linewidth=2, label=f'Model A (μ = {mu:.6f})')
    plt.plot(ages, predicted_B, '--', linewidth=2, label=f'Model B (A = {A:.6f}, B = {B:.6f}, c = {c:.4f})')

    plt.title(f'Dopasowanie modeli śmiertelności do TTŻ 2023 - {sex}')
    plt.xlabel('Wiek')
    plt.ylabel('Funkcja przeżycia S(x)')
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Wykres zapisany do: {save_path}")

    plt.show()


######### Funkcje pomocnicze do obliczania składki

# Parametry ubezpieczenia
i = 0.05  # efektywna roczna stopa procentowa
v = 1 / (1 + i)  # czynnik dyskontujący
x = 25  # wiek początkowy
n = 30  # okres ubezpieczenia
benefits = {
    'n': 100000,  # wypłata w przypadku śmierci naturalnej
    'w': 200000,  # wypłata w przypadku śmierci spowodowanej wypadkiem
    'c': 500000   # wypłata w przypadku śmierci spowodowanej chorobą
}
proportions = {
    'n': 0.85,  # proporcja natężenia śmiertelności - śmierć naturalna
    'w': 0.10,  # proporcja natężenia śmiertelności - wypadek
    'c': 0.05   # proporcja natężenia śmiertelności - choroba
}


# Funkcje przeżycia dla modeli A, B i C
def survival_A(t, mu):
    return np.exp(-mu * t)


def survival_B(t, x, params):
    A, B, c = params['A'], params['B'], params['c']
    return np.exp(-A*t - (B/np.log(c))*(c**(x+t) - c**x))


def survival_C(t, x, data):
    if x + t >= len(data):
        return 0.0
    return data.loc[x + t, 'l'] / data.loc[x, 'l']


# Prawdopodobieństwo śmierci w j-tym roku dla modelu A
def death_prob_A_year(j, mu, cause_prop):
    p_j = survival_A(j, mu)
    p_j_plus_1 = survival_A(j+1, mu)
    q_j = p_j - p_j_plus_1

    return q_j * cause_prop


# Prawdopodobieństwo śmierci w j-tym roku dla modelu B
def death_prob_B_year(j, x, params, cause_prop):
    p_j = survival_B(j, x, params)
    p_j_plus_1 = survival_B(j+1, x, params)
    q_j = p_j - p_j_plus_1

    return q_j * cause_prop


# Prawdopodobieństwo śmierci w j-tym roku dla modelu C
def death_prob_C_year(j, x, data, cause_prop):

    if x + j >= len(data) or x + j + 1 >= len(data):
        return 0.0

    p_j = survival_C(j, x, data)
    p_j_plus_1 = survival_C(j+1, x, data)
    q_j = p_j - p_j_plus_1

    return q_j * cause_prop


# Obliczenie wartości A (oczekiwanej wartości wypłaty)
def calculate_A(model, sex, x, n, benefits, proportions, params, ttz_data=None):
    """
    Obliczenie wartości A (oczekiwanej wartości wypłaty)
    model - 'A', 'B' lub 'C'
    sex - 'M' lub 'F'
    x - wiek początkowy
    n - okres ubezpieczenia
    benefits - słownik z wypłatami dla różnych przyczyn śmierci
    proportions - słownik z proporcjami natężenia śmiertelności
    params - parametry modeli
    ttz_data - dane TTŻ 2023 (tylko dla modelu C)
    """
    A_value = 0

    for j in range(n):
        if model == 'A':
            mu = params[sex]['A']['mu']
            # Prawdopodobieństwo śmierci w j-tym roku z poszczególnych przyczyn
            q_n = death_prob_A_year(j, mu, proportions['n'])
            q_w = death_prob_A_year(j, mu, proportions['w'])
            q_c = death_prob_A_year(j, mu, proportions['c'])

        elif model == 'B':
            model_params = params[sex]['B']
            # Prawdopodobieństwo śmierci w j-tym roku z poszczególnych przyczyn
            q_n = death_prob_B_year(j, x, model_params, proportions['n'])
            q_w = death_prob_B_year(j, x, model_params, proportions['w'])
            q_c = death_prob_B_year(j, x, model_params, proportions['c'])

        elif model == 'C':
            # Prawdopodobieństwo śmierci w j-tym roku z poszczególnych przyczyn
            q_n = death_prob_C_year(j, x, ttz_data, proportions['n'])
            q_w = death_prob_C_year(j, x, ttz_data, proportions['w'])
            q_c = death_prob_C_year(j, x, ttz_data, proportions['c'])

        # Składnik sumy dla j-tego roku
        A_value += v**(j+1) * (benefits['n'] * q_n + benefits['w'] * q_w + benefits['c'] * q_c)

    return A_value


# Obliczenie wartości a (wartości obecnej przyszłych składek)
def calculate_a(model, sex, x, n, params, ttz_data=None):
    """
    Obliczenie wartości a (wartości obecnej przyszłych składek)
    model - 'A', 'B' lub 'C'
    sex - 'M' lub 'F'
    x - wiek początkowy
    n - okres ubezpieczenia
    params - parametry modeli
    ttz_data - dane TTŻ 2023 (tylko dla modelu C)
    """
    a_value = 0

    for j in range(n):
        if model == 'A':
            mu = params[sex]['A']['mu']
            # Prawdopodobieństwo przeżycia j lat
            p_j = survival_A(j, mu)
        elif model == 'B':
            model_params = params[sex]['B']
            # Prawdopodobieństwo przeżycia j lat
            p_j = survival_B(j, x, model_params)
        elif model == 'C':
            # Prawdopodobieństwo przeżycia j lat
            p_j = survival_C(j, x, ttz_data)

        # Składnik sumy dla j-tego roku
        a_value += v**j * p_j

    return a_value


# Obliczenie wysokości składki
def calculate_premium(model, sex, x, n, benefits, proportions, params, ttz_data=None):
    """
    Obliczenie wysokości składki
    model - 'A', 'B' lub 'C'
    sex - 'M' lub 'F'
    x - wiek początkowy
    n - okres ubezpieczenia
    benefits - słownik z wypłatami dla różnych przyczyn śmierci
    proportions - słownik z proporcjami natężenia śmiertelności
    params - parametry modeli
    ttz_data - dane TTŻ 2023 (tylko dla modelu C)
    """
    # Obliczenie A (oczekiwanej wartości wypłaty)
    A_value = calculate_A(model, sex, x, n, benefits, proportions, params, ttz_data)

    # Obliczenie a (wartości obecnej przyszłych składek)
    a_value = calculate_a(model, sex, x, n, params, ttz_data)

    # Obliczenie wysokości składki
    premium = A_value / a_value

    return premium, A_value, a_value