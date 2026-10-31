from scipy.stats import ks_2samp


# Проверяем одинаковые данные
def test_no_drift():
    reference = [20, 25, 30, 35, 40]
    current = [20, 25, 30, 35, 40]

    statistic, p_value = ks_2samp(reference, current)

    assert p_value >= 0.05


# Проверяем разные распределения
def test_drift():
    reference = list(range(20, 120))
    current = list(range(200, 300))

    statistic, p_value = ks_2samp(reference, current)

    assert p_value < 0.05

