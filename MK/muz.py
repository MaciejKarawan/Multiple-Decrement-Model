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


### Poprawka, dla modelu A estymowanego na konkretnym przedziale

# Funkcja do estymacji parametru mu na wybranym przedziale wiekowym
def estimate_mu_on_interval(data, age_range, sex_label):
    """
    Estymacja parametru mu modelu A na wybranym przedziale wiekowym

    Parametry:
    data - DataFrame z danymi TTŻ
    age_range - krotka (min_age, max_age) określająca przedział wiekowy
    sex_label - etykieta płci do wyświetlania (np. "mężczyźni", "kobiety")

    Zwraca:
    mu - estymowany parametr mu
    """
    min_age, max_age = age_range

    # Filtrowanie danych do wybranego przedziału
    mask = (data['age'] >= min_age) & (data['age'] <= max_age)
    filtered_data = data[mask].copy()

    if len(filtered_data) == 0:
        raise ValueError(f"Brak danych w przedziale wiekowym {min_age}-{max_age}")

    # Przygotowanie danych do kalibracji
    ages = filtered_data['age'].values - min_age  # Przesunięcie wieku, aby zaczynał się od 0
    l_values = filtered_data['l'].values

    # Normalizacja l_values do funkcji przeżycia (l_x / l_0)
    survival_values = l_values / l_values[0]

    # Kalibracja modelu A
    result_A = minimize(objective_A, x0=[0.01], args=(ages, survival_values),
                        bounds=[(0.0001, 0.1)], method='L-BFGS-B')
    mu_optimal = result_A.x[0]

    print(f"Estymacja parametru μ dla {sex_label} w przedziale {min_age}-{max_age} lat:")
    print(f"μ = {mu_optimal:.8f}")

    return mu_optimal

# Funkcja do wizualizacji wielu fitów modelu A na różnych przedziałach
def visualize_multi_fits(data, age_ranges, sex_label, save_path=None):
    """
    Wizualizacja dopasowania modelu A na wielu przedziałach wiekowych

    Parametry:
    data - DataFrame z danymi TTŻ
    age_ranges - lista krotek (min_age, max_age, label) określających przedziały wiekowe i ich etykiety
    sex_label - etykieta płci do wyświetlania (np. "mężczyźni", "kobiety")
    save_path - ścieżka do zapisania wykresu (opcjonalnie)
    """
    plt.figure(figsize=(10, 6))

    # Pełne dane do porównania
    full_ages = data['age'].values
    full_l_values = data['l'].values
    full_survival = full_l_values / full_l_values[0]

    # Rysowanie oryginalnych danych
    plt.plot(full_ages, full_survival, 'o', markersize=3, alpha=0.5, label='TTŻ 2023')

    # Kolory dla różnych fitów
    colors = plt.cm.tab10.colors

    # Rysowanie fitów dla każdego przedziału
    for i, (min_age, max_age, label) in enumerate(age_ranges):
        # Filtrowanie danych do wybranego przedziału
        mask = (data['age'] >= min_age) & (data['age'] <= max_age)
        filtered_data = data[mask].copy()

        if len(filtered_data) == 0:
            print(f"Ostrzeżenie: Brak danych w przedziale {min_age}-{max_age}")
            continue

        # Przygotowanie danych do kalibracji
        ages = filtered_data['age'].values - min_age  # Przesunięcie wieku, aby zaczynał się od 0
        l_values = filtered_data['l'].values
        survival_values = l_values / l_values[0]

        # Estymacja parametru mu
        mu = estimate_mu_on_interval(data, (min_age, max_age), sex_label)

        # Generowanie przewidywań modelu A dla całego zakresu wieku
        predicted_ages = np.arange(max_age - min_age + 1)
        predicted_survival = np.array([model_A_survival(t, mu) for t in predicted_ages])

        # Rysowanie dopasowania
        plt.plot(predicted_ages + min_age, predicted_survival, '-',
                 color=colors[i % len(colors)], linewidth=2,
                 label=f'{label}: μ = {mu:.6f}')

        # Zaznaczenie przedziału na wykresie
        plt.axvspan(min_age, max_age, alpha=0.1, color=colors[i % len(colors)])

    plt.title(f'Dopasowanie modelu A (stałe natężenie) dla różnych przedziałów wiekowych - {sex_label}')
    plt.xlabel('Wiek')
    plt.ylabel('Funkcja przeżycia S(x)')
    plt.legend(loc='best')
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


def calculate_reserves(model, sex, x, n, benefits, premium, proportions, params, ttz_data=None):
    """
    ...
    """
    print(model)
    print(premium)
    Reserves = np.zeros(n+1)
    for j in range(n-1, -1, -1):
        if model == 'A':
            mu = params[sex]['A']['mu']
            # Prawdopodobieństwo śmierci x+j latka w ciągu roku z poszczególnych przyczyn
            q_n = proportions['n'] * (1 - np.exp(-mu))
            q_w = proportions['w'] * (1 - np.exp(-mu))
            q_c = proportions['c'] * (1 - np.exp(-mu))

            # prawdopodobieństwo przeżycia roku przez x+j latka
            p = 1 - (q_n + q_w + q_c)

        elif model == 'B':
            model_params = params[sex]['B']
            # Prawdopodobieństwo śmierci x+j latka w ciągu roku z poszczególnych przyczyn
            q_n = death_prob_B_year(0, x+j, model_params, proportions['n'])
            q_w = death_prob_B_year(0, x+j, model_params, proportions['w'])
            q_c = death_prob_B_year(0, x+j, model_params, proportions['c'])
            # prawdopodobieństwo przeżycia roku przez x+j latka
            p = 1 - (q_n + q_w + q_c)
        elif model == 'C':
            # Prawdopodobieństwo śmierci x+j latka w ciągu roku z poszczególnych przyczyn
            q_n = death_prob_C_year(0, x+j, ttz_data, proportions['n'])
            q_w = death_prob_C_year(0, x+j, ttz_data, proportions['w'])
            q_c = death_prob_C_year(0, x+j, ttz_data, proportions['c'])
            # prawdopodobieństwo przeżycia roku przez x+j latka
            p = 1 - (q_n + q_w + q_c)
        # Wartość rezerwy w następnym roku trwania ubezpieczenia

        V_prev = v*(benefits['n'] * q_n + benefits['w'] * q_w + benefits['c'] * q_c + p * Reserves[j+1]) - premium
        Reserves[j] = V_prev

    return Reserves


def visualize_reserves_split(x, n, reserves_dict, save_path_prefix=None):
    """
    Rysuje panel 1×3 wykresów rezerw dla modeli A, B i C,
    każdy wykres porównuje rezerwy Mężczyzn i Kobiet.

    Parametry:
      x               - wiek początkowy (wykorzystywane na osi)
      n               - liczba lat trwania (rozmiar rezerw to n+1)
      reserves_dict   - słownik rezerw: {'A': {'M': arr, 'F': arr}, ...}
      save_path_prefix- (opcjonalnie) prefiks ścieżki do zapisu
    """

    models = ['A', 'B', 'C']
    sexes = ['M', 'F']
    styles = {'M': '-', 'F': '--'}
    labels = {'M': 'Mężczyźni', 'F': 'Kobiety'}

    years = np.arange(n + 1)  # od 0 do n

    # Stwórz panel 1×3
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, model in enumerate(models):
        ax = axes[i]
        for s in sexes:
            res = reserves_dict[model][s]
            ax.plot(years, res, styles[s], lw=2, label=labels[s])
        ax.set_title(f'Model {model} – rezerwy')
        ax.set_xlabel('Lata od rozpoczęcia')
        if i == 0:
            ax.set_ylabel('Wartość rezerwy')
        ax.grid(True)
        ax.legend()
    plt.tight_layout()

    if save_path_prefix:
        path = f"{save_path_prefix}_reserves_panel.png"
        fig.savefig(path, dpi=300, bbox_inches='tight')
        print(f'Zapisano panel rezerw do: {path}')
    plt.show()

def calculate_risk_premium(model, sex, x, n, benefits, proportions, params, reserves, ttz_data=None):
    risk_premium_values = []
    for j in range(n):
        if model == 'A':
            mu = params[sex]['A']['mu']
            # Prawdopodobieństwo śmierci x+j latka w ciągu roku z poszczególnych przyczyn
            q_n = death_prob_A_year(0, mu, proportions['n'])
            q_w = death_prob_A_year(0, mu, proportions['w'])
            q_c = death_prob_A_year(0, mu, proportions['c'])

        elif model == 'B':
            model_params = params[sex]['B']
            # Prawdopodobieństwo śmierci x+j latka w ciągu roku z poszczególnych przyczyn
            q_n = death_prob_B_year(0, x+j, model_params, proportions['n'])
            q_w = death_prob_B_year(0, x+j, model_params, proportions['w'])
            q_c = death_prob_B_year(0, x+j, model_params, proportions['c'])


        elif model == 'C':
            # Prawdopodobieństwo śmierci x+j latka w ciągu roku z poszczególnych przyczyn
            q_n = death_prob_C_year(0, x+j, ttz_data, proportions['n'])
            q_w = death_prob_C_year(0, x+j, ttz_data, proportions['w'])
            q_c = death_prob_C_year(0, x+j, ttz_data, proportions['c'])

        risk_premium_values.append(v*(benefits['n'] * q_n + benefits['w'] * q_w + benefits['c'] * q_c - reserves[j+1] * (q_n+q_w+q_c) ))
    return risk_premium_values

def calculate_savings_premium(premium, risk_premium_value):
    return premium - risk_premium_value

def visualize_premiums_split(x, n, benefits, proportions, params, reserves_dict, ttz_data=None, save_path_prefix=None):
    """
    Rysuje dwa panele 1×3:
      1) Roczne wartości składki ryzyka (risk premium) dla modeli A, B, C
         w panelu rosnącym j (okres od 0 do n-1)
      2) Roczne wartości składki oszczędności (savings premium)

    Parametry:
      x               - wiek początkowy
      n               - liczba lat trwania
      benefits        - słownik wypłat {'n','w','c'}
      proportions     - słownik proporcji śmiertelności {'n','w','c'}
      params          - słownik parametrów modeli z kluczami 'M' i 'F'
      reserves_dict   - słownik rezerw: {'A': reserves_A, 'B': reserves_B, 'C': reserves_C}
                        gdzie reserves_* to tablica długości n+1
      ttz_data        - (opcjonalnie) dane TTŻ dla modelu C
      save_path_prefix- prefiks ścieżki do zapisu wykresów
    """
    models = ['A', 'B', 'C']
    sexes = ['M', 'F']
    styles = {'M': '-', 'F': '--'}
    labels = {'M': 'Mężczyźni', 'F': 'Kobiety'}

    years = np.arange(n)

    # oblicz risk i savings
    risk = {model: {s: [] for s in sexes} for model in models}
    save = {model: {s: [] for s in sexes} for model in models}

    for model in models:
        for s in sexes:
            # podajemy DataFrame tylko do modelu C:
            df_ttż = ttz_data[s] if (model == 'C' and ttz_data is not None) else None

            # obliczamy ryzyko raz (to zwraca listę długości n)
            risk_vals = calculate_risk_premium(
                model, s, x, n, benefits, proportions, params, reserves_dict[model][s], df_ttż
            )
            risk[model][s] = risk_vals

            # obliczamy składkę jednorazowo:
            prem = calculate_premium(
                model, s, x, n, benefits, proportions, params, df_ttż
            )[0]
            # i składkę oszczędności jako różnicę:
            save[model][s] = calculate_savings_premium(prem, risk_vals)

    # Panel 1: risk premium
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, model in enumerate(models):
        ax = axes1[i]
        for s in sexes:
            ax.plot(years, risk[model][s], styles[s], lw=2, label=labels[s])
        ax.set_title(f'Model {model} – składka ryzyka')
        ax.set_xlabel('Lata od rozpoczęcia')
        if i == 0:
            ax.set_ylabel('Risk Premium')
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    if save_path_prefix:
        p1 = f"{save_path_prefix}_risk_panel.png"
        fig1.savefig(p1, dpi=300, bbox_inches='tight')
        print(f'Zapisano panel składek ryzyka do: {p1}')

    # Panel 2: savings premium
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, model in enumerate(models):
        ax = axes2[i]
        for s in sexes:
            ax.plot(years, save[model][s], styles[s], lw=2, label=labels[s])
        ax.set_title(f'Model {model} – składka oszczędności')
        ax.set_xlabel('Lata od rozpoczęcia')
        if i == 0:
            ax.set_ylabel('Savings Premium')
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    if save_path_prefix:
        p2 = f"{save_path_prefix}_savings_panel.png"
        fig2.savefig(p2, dpi=300, bbox_inches='tight')
        print(f'Zapisano panel składek oszczędności do: {p2}')

    plt.show()

def build_reserves_dict(x, n, benefits, proportions, params, ttz_data=None):
    """
    Zwraca słownik rezerw:
      {'A': {'M': reserves_AM, 'F': reserves_AF},
       'B': {'M': reserves_BM, 'F': reserves_BF},
       'C': {'M': reserves_CM, 'F': reserves_CF}}
    gdzie dla modelu C wykorzystujemy osobne DataFrame'y z ttz_data['M'] i ttz_data['F'].
    """
    reserves_dict = {'A': {}, 'B': {}, 'C': {}}
    sexes = ['M', 'F']

    for model in ['A', 'B', 'C']:
        for s in sexes:
            # wybierz dane TTŻ tylko dla modelu C:
            df_ttz = ttz_data[s] if (model == 'C' and ttz_data is not None) else None

            # oblicz składkę (potrzebna do rezerw)
            premium, _, _ = calculate_premium(
                model, s, x, n, benefits, proportions, params, df_ttz
            )

            # oblicz rezerwy
            reserves = calculate_reserves(
                model, s, x, n, benefits, premium, proportions, params, df_ttz
            )

            reserves_dict[model][s] = reserves

    return reserves_dict

def visualize_death_probs(x, n, ttz_data, params, proportions, save_path=None):
    """
    Rysuje dwa pliki:
      1) Panel 1×3 rocznych prawdopodobieństw q_j dla modeli A, B, C
      2) Panel 1×3 skumulowanych prawdopodobieństw Q_j
    Każdy panel porównuje płcie.

    Parametry:
      x               - wiek początkowy
      n               - liczba lat do przeanalizowania
      ttz_data        - słownik TTŻ 2023: {'M': df_m, 'F': df_f}
      params          - słownik parametrów modeli z kluczami 'M' i 'F'
      proportions     - słownik proporcji śmiertelności {'n','w','c'}
      save_path_prefix- (opcjonalnie) prefiks ścieżki do zapisu (bez rozszerzenia)
    """

    models = ['A', 'B', 'C']
    sexes = ['M', 'F']
    styles = {'M': '-', 'F': '--'}
    labels = {'M': 'Mężczyźni', 'F': 'Kobiety'}

    years = np.arange(n)
    # oblicz q i Q
    q = {model: {s: [] for s in sexes} for model in models}
    Q = {model: {s: None for s in sexes} for model in models}
    for model in models:
        for s in sexes:
            df = ttz_data[s]
            arr = []
            for j in years:
                if model == 'A':
                    mu = params[s]['A']['mu']
                    val = sum(death_prob_A_year(j, mu, proportions[c]) for c in proportions)
                elif model == 'B':
                    pB = params[s]['B']
                    val = sum(death_prob_B_year(j, x, pB, proportions[c]) for c in proportions)
                else:
                    val = sum(death_prob_C_year(j, x, df, proportions[c]) for c in proportions)
                arr.append(val)
            q[model][s] = np.array(arr)
            Q[model][s] = np.cumsum(arr)
    # Panel 1: roczne prawdopodobienstwa
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, model in enumerate(models):
        ax = axes1[i]
        for s in sexes:
            ax.plot(years, q[model][s], styles[s], lw=2, label=labels[s])
        ax.set_title(f'Model {model} – q_j')
        ax.set_xlabel('Lata od rozpoczęcia')
        if i == 0:
            ax.set_ylabel('q_j')
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    if save_path:
        path1 = f"{save_path}_q_panel.png"
        fig1.savefig(path1, dpi=300, bbox_inches='tight')
        print(f'Zapisano panel q_j do: {path1}')

    # Panel 2: skumulowane prawdopodobienstwa
    fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    for i, model in enumerate(models):
        ax = axes2[i]
        for s in sexes:
            ax.plot(years, Q[model][s], styles[s], lw=2, label=labels[s])
        ax.set_title(f'Model {model} – Q_j')
        ax.set_xlabel('Lata od rozpoczęcia')
        if i == 0:
            ax.set_ylabel('Q_j')
        ax.grid(True)
        ax.legend()
    plt.tight_layout()
    if save_path:
        path2 = f"{save_path}_cumulative_q_panel.png"
        fig2.savefig(path2, dpi=300, bbox_inches='tight')
        print(f'Zapisano panel Q_j do: {path2}')

    plt.show()

