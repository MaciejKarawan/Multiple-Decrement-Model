from decrement_model.paths import DATA, generated_path
from decrement_model.model import *

setup_plotting_style()


def part1():
    # Wczytanie danych dla mężczyzn
    global data_m
    data_m = pd.read_excel(DATA / "ttz2023.xlsx", sheet_name=0, skiprows=6, usecols="B:G", header=None)
    data_m.columns = ['q', 'l', 'd', 'L', 'T', 'e']
    data_m['sex'] = 'M'
    data_m['age'] = range(len(data_m))

    # Wczytanie danych dla kobiet
    global data_f
    data_f = pd.read_excel(DATA / "ttz2023.xlsx", sheet_name=1, skiprows=6, usecols="B:G", header=None)
    data_f.columns = ['q', 'l', 'd', 'L', 'T', 'e']
    data_f['sex'] = 'F'
    data_f['age'] = range(len(data_f))

    print("Dane TTŻ 2023 wczytane pomyślnie.")
    print(f"Liczba wierszy dla mężczyzn: {len(data_m)}")
    print(f"Liczba wierszy dla kobiet: {len(data_f)}")

    # Kalibracja modeli dla mężczyzn
    params_m = calibrate_models(data_m, "mężczyzn")

    # Kalibracja modeli dla kobiet
    params_f = calibrate_models(data_f, "kobiet")

    # Wizualizacja dopasowania modeli
    #visualize_models(data_m, params_m, "mężczyźni", save_path="dopasowanie_modeli_mezczyzni.png")
    #visualize_models(data_f, params_f, "kobiety", save_path="dopasowanie_modeli_kobiety.png")

    # Przykładowe przedziały wiekowe dla mężczyzn
    male_age_ranges = [
        (25, 55, "25 - 55"),
        (20, 60, "20 - 60"),
        (15, 65, "15 - 65"),
    ]

    # Estymacja parametru mu dla wybranego przedziału
    mu_m = estimate_mu_on_interval(data_m, (25, 55), "mężczyzn")
    mu_f = estimate_mu_on_interval(data_m, (25, 55), "kobiety")

    # Wizualizacja wielu fitów
    visualize_multi_fits(data_m, male_age_ranges, "mężczyźni", save_path="multi_fits_model_A_mezczyzni.png")
    visualize_multi_fits(data_f, male_age_ranges, "kobiety", save_path="multi_fits_model_A_kobiety.png")

    # Parametry modeli śmiertelności skalibrowane na podstawie TTŻ 2023
    global params

    params = {
        'M': {
            'A': {'mu': mu_m},
            'B': {'A': params_m['B']['A'], 'B': params_m['B']['B'], 'c': params_m['B']['c']}
        },
        'F': {
            'A': {'mu': mu_f},
            'B': {'A': params_f['B']['A'], 'B': params_f['B']['B'], 'c': params_f['B']['c']}
        }
    }


def part2():

    """# Parametry ubezpieczenia
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
    }"""

    # Słownik z danymi TTŻ 2023
    ttz_data = {
        'M': data_m,
        'F': data_f
    }

    # Obliczenie składek dla wszystkich modeli i płci
    results = {}

    for sex in ['M', 'F']:
        results[sex] = {}
        for model in ['A', 'B', 'C']:
            if model == 'C':
                premium, A_value, a_value = calculate_premium(
                    model, sex, x, n, benefits, proportions, params, ttz_data[sex]
                )
            else:
                premium, A_value, a_value = calculate_premium(
                    model, sex, x, n, benefits, proportions, params
                )

            results[sex][model] = {
                'premium': premium,
                'A_value': A_value,
                'a_value': a_value
            }

    # Wyświetlenie wyników
    print("\nWysokość składki dla ubezpieczenia na życie dla 25-latków na okres 30 lat")
    print("=" * 70)
    print(f"Stopa procentowa: {i*100}%")
    print(f"Wypłaty: śmierć naturalna - {benefits['n']} zł, wypadek - {benefits['w']} zł, choroba - {benefits['c']} zł")
    print("=" * 70)

    for sex, sex_name in [('M', 'Mężczyzna'), ('F', 'Kobieta')]:
        print(f"\n{sex_name}:")
        for model in ['A', 'B', 'C']:
            if model == 'A':
                model_name = "Model A (stałe natężenie)"
            elif model == 'B':
                model_name = "Model B (Makeham)"
            else:
                model_name = "Model C (TTŻ 2023)"

            premium = results[sex][model]['premium']
            A_value = results[sex][model]['A_value']
            a_value = results[sex][model]['a_value']

            print(f"  {model_name}:")
            print(f"    Składka roczna: {premium:.2f} zł")
            print(f"    Wartość A: {A_value:.2f} zł")
            print(f"    Wartość a: {a_value:.4f}")

    # Zapisanie wyników do pliku
    with open(generated_path("wyniki_skladek_ABC.txt"), 'w', encoding='utf-8') as f:
        f.write("Wysokość składki dla ubezpieczenia na życie dla 25-latków na okres 30 lat\n")
        f.write("=" * 70 + "\n")
        f.write(f"Stopa procentowa: {i*100}%\n")
        f.write(f"Wypłaty: śmierć naturalna - {benefits['n']} zł, wypadek - {benefits['w']} zł, choroba - {benefits['c']} zł\n")
        f.write("=" * 70 + "\n\n")

        for sex, sex_name in [('M', 'Mężczyzna'), ('F', 'Kobieta')]:
            f.write(f"{sex_name}:\n")
            for model in ['A', 'B', 'C']:
                if model == 'A':
                    model_name = "Model A (stałe natężenie)"
                elif model == 'B':
                    model_name = "Model B (Makeham)"
                else:
                    model_name = "Model C (TTŻ 2023)"

                premium = results[sex][model]['premium']
                A_value = results[sex][model]['A_value']
                a_value = results[sex][model]['a_value']

                f.write(f"  {model_name}:\n")
                f.write(f"    Składka roczna: {premium:.2f} zł\n")
                f.write(f"    Wartość A: {A_value:.2f} zł\n")
                f.write(f"    Wartość a: {a_value:.4f}\n\n")

    print("\nWyniki zapisano do pliku 'wyniki_skladek_ABC.txt'")




if __name__ == '__main__':
    part1()
    part2()
    visualize_death_probs(
        x=25,
        n=30,
        ttz_data={'M' : data_m, 'F' : data_f},
        params=params,
        proportions=proportions,
        save_path='MK'
    )
    reserves_dict = build_reserves_dict(
        x=25,
        n=30,
        benefits=benefits,
        proportions=proportions,
        params=params,
        ttz_data={'M': data_m, 'F': data_f}
    )

    visualize_reserves_split(
        x=25,
        n=30,
        reserves_dict=reserves_dict,
        save_path_prefix='MK'
    )

    visualize_premiums_split(
        x=25,
        n=30,
        benefits=benefits,
        proportions=proportions,
        params=params,
        reserves_dict=reserves_dict,
        ttz_data={'M': data_m, 'F': data_f},
        save_path_prefix='MK'
    )

