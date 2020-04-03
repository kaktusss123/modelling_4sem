import numpy as np
from numpy.linalg import matrix_power as mp


def read_matrix(filename: str):
    """
    Читает матрицу из файла
    """
    matrix = np.zeros((15, 15))
    with open(filename, encoding='utf-8') as f:
        for i, line in enumerate(f):
            matrix[i, :] = list(map(float, line.replace(',', '.').split()))
    return matrix


def task1(p: np.array, i: int, j: int, k: int) -> int:
    """
    Вероятность того, что за k шагов система перейдет из состояния i в состояние j
    """
    p_k = mp(p, k)
    return p_k[i, j]


def task2(p: np.array, a_0: np.array, k: int) -> np.array:
    """
    Вероятности состояний системы спустя k шагов, если в начальный момент вероятность состояний были a_0
    """
    return np.dot(mp(p, k), a_0)


def task3(p: np.array, i_: int, j_: int, k: int) -> float:
    """
    Вероятность первого перехода за k шагов из состояния i в состояние j
    """
    p_ = p
    for _ in range(2, k+1):  # 2, 3, 4, ..., k-1, k
        # сделаем матрицу для рассчета вероятности первого прехода для конкретного k
        new = np.zeros(p.shape)
        # заполняем матрицу
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                s = 0
                for m in range(p.shape[0]):
                    s += p[i, m] * p_[m, j] if m != j else 0
                new[i, j] = s
        # делаем полученную матрицу p_(k-1)
        p_ = new
    return p_[i_, j_]


def task4(p: np.array, i: int, j: int, k: int) -> float:
    """
    Вероятность перехода из i в j не позднее, чем за k шагов
    """
    s = 0
    for t in range(1, k+1):
        s += task3(p, i, j, t)
    return s


def task5(p: np.array, i: int, j: int) -> float:
    """
    Среднее количество шагов для перехода из состояния i в состояние j
    """
    t = 1
    t_ = 0
    for t in range(1, 100):
        t_ += t * task3(p, i, j, t)
    return t_


def task6(p: np.array, j: int, k: int) -> float:
    """
    Вероятность первого возвращения в состояние j за k шагов
    """
    s = np.zeros(p.shape)
    for m in range(1, k):
        s += task6(p, j, m) * mp(p, k-m)
    return (mp(p, k) - s)[j, j]


def task7(p: np.array, j: int, k: int) -> float:
    """
    Вероятность возвращения в состояние j не позднее чем за k шагов
    """
    s = 0
    for t in range(1, k+1):
        s += task6(p, j, t)
    return s


def task8(p: np.array, j: int) -> float:
    """
    Среднее время возвращения в состояние j
    """
    s = 0
    for t in range(1, 100):  # большое число
        s += t * task6(p, j, t)
    return s


def task9(p):
    """
    Установившиеся вероятности
    """
    m = p.T - np.eye(p.shape[0])
    m[-1, :] = 1
    b = np.array([0] * (p.shape[0] - 1) + [1])
    x = np.dot(np.linalg.inv(m), b)
    return x


if __name__ == "__main__":
    matrix = read_matrix('matrix.csv')

    # Task1
    print(
        f'1) Вероятность того, что за 8 шагов система перейдет из состояния 11 в состояние 2:
        {task1(matrix, 10, 1, 8): .3f}')

    # Task2
    t2=task2(matrix, np.array((0.08, 0.11, 0.09, 0.11, 0, 0.01,
                                 0.05, 0.01, 0.12, 0.05, 0.07, 0.11, 0.06, 0.05, 0.08)), 10)
    print(f'2) Вероятности состояний системы спустя 10 шагов, если в начальный момент вероятность состояний \
    были следующими: \nA=(0,08;0,11;0,09;0,11;0;0,01;0,05;0,01;0,12;0,05;0,07;0,11;0,06;0,05;0,08)\n')
    print(t2)

    # Task3
    print(f'3) Вероятность первого перехода за 8 шагов из состояния 14 в состояние 3')
    print(task3(matrix, 13, 2, 8))

    # Task4
    print(f'4) Вероятность перехода из состояния 3 в состояние 13 не позднее чем за 10 шагов')
    print(task4(matrix, 2, 12, 10))

    # Task5
    print(f'5) Среднее количество шагов для перехода из состояния 11 в состояние 15')
    print(task5(matrix, 10, 14))

    # Task6
    print(f'6) Вероятность первого возвращения в состояние 5 за 7 шагов')
    print(task6(matrix, 4, 7))

    # Task7
    print(f'7) Вероятность возвращения в состояние 5 не позднее чем за 6 шагов')
    print(task7(matrix, 4, 6))

    # Task8
    print(f'8) Среднее время возвращения в состояние 12')
    print(task8(matrix, 11))

    # Task9
    print(f'Установившиеся вероятности')
    print(task9(matrix))
