import torch

# 1.1 - Создание тензоров
# Создайте следующие тензоры:
firstTensor = torch.rand((3, 4)) # - Тензор размером 3x4, заполненный случайными числами от 0 до 1
secondTensor = torch.zeros((2, 3, 4)) # - Тензор размером 2x3x4, заполненный нулями
thirdTensor = torch.ones((5, 5)) # - Тензор размером 5x5, заполненный единицами
fourthTensor = torch.arange(0, 16).reshape((4, 4)) # - Тензор размером 4x4 с числами от 0 до 15 (используйте reshape)


# 1.2 - Операция с тензорами
# Дано: тензор A размером 3x4 и тензор B размером 4x3
A = torch.rand((3, 4))
B = torch.rand((4, 3))
# Выполните:
firstOp = A.T # - Транспонирование тензора A
secondOp = A @ B # - Матричное умножение A и B
thirdOp = A * B.T # - Поэлементное умножение A и транспонированного B
fourthOp = A.sum() # - Вычислите сумму всех элементов тензора A


# 1.3 - Индексация и срезы
# Создайте тензор размером 5x5x5
tensor = torch.rand((5, 5, 5))
# Извлеките:
firstSlice = tensor[0, 0, :] # - Первую строку
secondSlice = tensor[:, :, -1] # - Последний столбец
thirdSlice = tensor[2, 1:3, 1:3] # - Подматрицу размером 2x2 из центра тензора
fourthSlice = tensor[::2, ::2, ::2] # - Все элементы с четными индексами

# 1.4 - Работа с формами
# Создайте тензор размером 24 элемента
tensor24 = torch.arange(24)
# Преобразуйте его в следующие формы:
firstReshape = tensor24.reshape(2, 12) # - 2x12
secondReshape = tensor24.reshape(3, 8) # - 3x8
thirdReshape = tensor24.reshape(4, 6) # - 4x6
fourthReshape = tensor24.reshape(2, 3, 4) # - 2x3x4
fifthReshape = tensor24.reshape(2, 2, 2, 3) # - 2x2x2x3

