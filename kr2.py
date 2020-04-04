import numpy as np


class SMO:

    def __create_matrix(self):
        self.__matrix = np.zeros((self.m + self.n + 1, self.m + self.n + 1))
        for i in range(self.m + self.n):
            self.__matrix[i, i+1] = self.lbd
            self.__matrix[i+1, i] = self.mu * \
                (i + 1) if i < self.m else self.mu * self.m

    def __get_stable_prob(self):
        """Установивишиеся вероятности"""
        D = np.diag([self.__matrix[i, :].sum()
                     for i in range(self.__matrix.shape[0])])
        M = self.__matrix.T - D
        M_ = M
        M_[-1, :] = 1
        B = np.zeros(M_.shape[0])
        B[-1] = 1
        self.__stable_prob = np.linalg.inv(M_).dot(B)

    def __init__(self, lbd, m, mu, n):
        self.lbd = lbd
        self.m = m
        self.mu = mu
        self.n = n
        self.__create_matrix()
        self.__get_stable_prob()

    # Ответы
    matrix = property(lambda self: self.__matrix)
    stable_prob = property(lambda self: self.__stable_prob)
    p_otk = property(lambda self: self.__stable_prob[-1])
    otn_prop_sp = property(lambda self: 1 - self.__stable_prob[-1])
    abs_prop_sp = property(lambda self: (
        1 - self.__stable_prob[-1]) * self.lbd)
    L_o4 = property(lambda self: sum(
        [i * self.__stable_prob[self.m + i] for i in range(1, self.n + 1)]))
    T_o4 = property(lambda self: sum(
        [(i + 1) / (self.m * self.mu) * self.__stable_prob[self.m + i] for i in range(self.n)]))
    N_kan = property(lambda self: sum(
        [i * self.__stable_prob[i] for i in range(1, self.m + 1)]) +
        sum([self.m * self.__stable_prob[i] for i in range(self.m + 1, self.m + self.n + 1)]))
    p_no_o4 = property(lambda self: sum(self.__stable_prob[:self.m]))
    sr_t_prost = property(lambda self: 1 / self.mu)
    sr_t_no_o4 = property(lambda self: sum(
        [i + 1 / (self.m * self.lbd) * self.__stable_prob[i] for i in range(self.m)]))


if __name__ == '__main__':
    smo = SMO(29, 5, 6, 14)
    # Task a - установившиеся вероятности
    print('a) Установившиеся вероятности:', smo.stable_prob)
    # Task b - вероятность отказа в обслуживании
    print('b) Вероятность отказа в обслуживании:', smo.p_otk)
    # Task c - Относительная и абсолютная интенсивность обслуживания
    print('c) Относительная пропускная способность:', smo.otn_prop_sp)
    print('c) Абсолютная пропускная способность:', smo.abs_prop_sp)
    # Task d - Средняя длина в очереди
    print('d) Средняя длина очереди:', smo.L_o4)
    # Task e - Среднее время в очереди
    print('e) Среднее время в очереди:', smo.T_o4)
    # Task f - Среднее число занятых каналов
    print('f) Среднее число занятых каналов:', smo.N_kan)
    # Task g - вероятность того, что поступающая заявка не будет ждать в очереди
    print('g) Вероятность того, что поступающая заявка не будет ждать в очереди:', smo.p_no_o4)
    # Task h - Найти среднее время простоя системы массового обслужива
    print('h) Среднее время простоя смо:', smo.sr_t_prost)
    # Task i - Найти среднее время, когда в системе нет очереди.
    print('i) Среднее время, когда в системе нет очереди:', smo.sr_t_no_o4)
    print('Матрица интенсивностей:')
    print(smo.matrix)
