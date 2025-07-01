import torch
import time
import torch.cuda

# 3.1 - Подготовка данных
# 3.2 - Функция измерения времени
# 3.3 - Сравнение операций
# Создайте большие матрицы следующих размеров:
# Заполните их случайными числами
device_cpu = torch.device('cpu')
device_gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

x = torch.rand((128, 1024, 1024)) # - 64 x 1024 x 1024
y = torch.rand((128, 1024, 1024)) # - 64 x 1024 x 1024
z = torch.rand((128, 1024, 1024)) # - 64 x 1024 x 1024

# Сравните время выполнения следующих операций на CPU и CUDA:
# - Матричное умножение (torch.matmul)
# - Поэлементное сложение
# - Поэлементное умножение
# - Транспонирование
# - Вычисление суммы всех элементов
def matMul(x, y, z):
    """
    Получаем матричное произведение тензоров
    :param x: Первый тензор
    :param y: Второй тензор
    :param z: Третий тензор
    :return: Матричное произведение первого и второго тензора
    """
    return torch.matmul(x, y)

def elSum(x, y, z):
    """
    Получаем поэлементную сумму трех тензоров
    :param x: Первый тензор
    :param y: Второй тензор
    :param z: Третий тензор
    :return: Поэлементная сумма первого, второго и третьего тензора
    """
    return x + y + z

def elMul(x, y, z):
    """
    Получаем поэлементное произведение трех тензоров
    :param x: Первый тензор
    :param y: Второй тензор
    :param z: Третий тензор
    :return: Поэлементное произведение первого, второго и третьего тензора
    """
    return x * y * z

def Transp(x, y, z):
    """
    Получаем транспонированные тензоры
    :param x: Первый тензор
    :param y: Второй тензор
    :param z: Третий тензор
    :return: Список их первого, второго и третьего транспонированного тензора
    """
    xT = x.mT
    yT = y.mT
    zT = z.mT
    return [xT, yT, zT]

def matSum(x, y, z):
    """
    Получаем общую сумму всех трех тензоров
    :param x: Первый тензор
    :param y: Второй тензор
    :param z: Третий тензор
    :return: Сумма первого, второго и третьего тензора
    """
    return x.sum() + y.sum() + z.sum()

operations = {"matMul": matMul, "elSumm": elSum, "elMull": elMul, "Transp": Transp, "matSum": matSum}

# Создайте функцию для измерения времени выполнения операций
# Используйте torch.cuda.Event() для точного измерения на графическом процессоре
# Используйте time.time() для измерения на центральном процессоре
def cpuTime():
    """
    Засекает время выполнения функции и создает сравнительную таблицу
    :return:
    """
    print(" Names | CPU (мс) | GPU (мс) | Ускорение")
    for operation in operations.keys():
        # Переносим все тензоры на CPU
        x.to(device=device_cpu)
        y.to(device=device_cpu)
        z.to(device=device_cpu)
        # Получаем время выполнения для каждой функции при использовании CPU
        start = time.time()
        res = operations[operation](x, y, z)
        end = time.time()
        cpuTime = (end - start) * 1000
        # Переносим все тензоры на GPU
        x.to(device=device_gpu)
        y.to(device=device_gpu)
        z.to(device=device_gpu)
        # Получаем время выполнения для каждой функции при использовании GPU
        startEvent = torch.cuda.Event(enable_timing=True)
        endEvent = torch.cuda.Event(enable_timing=True)
        startEvent.record()
        res = operations[operation](x, y, z)
        endEvent.record()
        torch.cuda.synchronize()
        gpuTime = startEvent.elapsed_time(endEvent)
        # Выводим строку с названием операции, временем и их разницей
        print(f"{operation} |  {cpuTime:.2f}  |  {gpuTime:.8f}  | {(cpuTime / gpuTime):.2f}x")


cpuTime()

# 3.4 - Анализ результатов
# - Быстрее всего на GPU выполняет матричное умножение
# - Некоторые несложные операции выгодно выполнять на CPU, так как в таком случае GPU работает медленнее из-за перемещения данных
# - Чем больше размер матриц, тем выше ускорение на GPU
#





