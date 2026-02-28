import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ
# ============================================================


def create_vector():
    """
    Создает одномерный массив (вектор) от 0 до 9 включительно.
    """
    # np.arange(n) создает массив от 0 до n-1
    return np.arange(10)


def create_matrix():
    """
    Создает двумерный массив (матрицу) размером 5x5 со случайными числами от 0 до 1.
    """
    # rand(rows, cols) генерирует числа из равномерного распределения [0, 1)
    return np.random.rand(5, 5)


def reshape_vector(vec):
    """
    Меняет форму массива из 10 элементов в таблицу 2 строки на 5 столбцов.
    """
    # общее количество элементов (2*5=10) должно совпадать с исходным
    return vec.reshape(2, 5)


def transpose_matrix(mat):
    """
    Транспонирует матрицу (зеркально отражает её относительно главной диагонали).
    """
    # .T — это самый быстрый и удобный способ транспонирования в NumPy
    return mat.T

# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a, b):
    """
    Сложение векторов одинаковой длины без использования циклов.
    """
    return a + b


def scalar_multiply(vec, scalar):
    """
    Умножение каждого элемента вектора на число.
    """
    return vec * scalar


def elementwise_multiply(a, b):
    """
    Поэлементное умножение двух массивов.
    """
    return a * b


def dot_product(a, b):
    """
    Скалярное произведение векторов.
    Результат — одно число (сумма произведений элементов).
    """
    return np.dot(a, b)

# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a, b):
    """
    Классическое умножение матриц.
    """
    return a @ b


def matrix_determinant(a):
    """
    Вычисление определителя квадратной матрицы.
    """
    return np.linalg.det(a)


def matrix_inverse(a):
    """
    Нахождение обратной матрицы.
    """
    return np.linalg.inv(a)


def solve_linear_system(a, b):
    """
    Решение системы линейных уравнений Ax = b.
    """
    return np.linalg.solve(a, b)

# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path="numpy_lab/data/students_scores.csv"):
    """Загружает CSV и возвращает NumPy массив чисел, пропуская заголовки."""
    # Pandas по умолчанию считает первую строку заголовком (header=0)
    df = pd.read_csv(path)
    # Теперь в df.values только цифры, и их можно смело превращать во float
    return df.values.astype(float)

def statistical_analysis(data):
    """Вычисляет основные статистические показатели."""
    # Принудительно превращаем входные данные в массив float
    data = np.asanyarray(data).astype(float) 
    
    return {
        "mean": np.mean(data),
        "median": np.median(data),
        "std": np.std(data),
        "min": np.min(data),
        "max": np.max(data),
        "q1": np.percentile(data, 25),
        "q3": np.percentile(data, 75)
    }

def normalize_data(data):
    """Min-Max нормализация данных."""
    d_min = np.min(data)
    d_max = np.max(data)
    return (data - d_min) / (d_max - d_min)

# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_histogram(data):
    """Строит и сохраняет гистограмму."""
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=5, color='skyblue', edgecolor='black')
    plt.title("Распределение оценок по математике")
    plt.xlabel("Баллы")
    plt.ylabel("Количество студентов")
    
    os.makedirs("plots", exist_ok=True) # Создаем папку, если её нет
    plt.savefig("plots/histogram.png")
    plt.close()

def plot_heatmap(matrix):
    """Строит тепловую карту корреляции."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Корреляция между предметами")
    plt.savefig("plots/heatmap.png")
    plt.close()

def plot_line(x, y):
    """Строит график зависимости оценки от номера студента."""
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o', linestyle='-', color='green')
    plt.title("Успеваемость: Студент -> Оценка (Math)")
    plt.xlabel("ID Студента")
    plt.ylabel("Балл")
    plt.grid(True)
    plt.savefig("plots/line_plot.png")
    plt.close()

if __name__ == "__main__":
    # --- ТЕСТЫ 1 БЛОКА ---
    print("--- Результаты 1-го блока ---")
    v = create_vector()
    print(f"Вектор от 0 до 9: {v}")
    
    m = create_matrix()
    print(f"Случайная матрица 5x5 (первая строка):\n{m[0]}") 
    
    rv = reshape_vector(v)
    print(f"Массив после reshape (2,5):\n{rv}")
    
    tm = transpose_matrix(rv)
    print(f"Транспонированный массив (5,2):\n{tm}")

    # --- ТЕСТЫ 2 БЛОКА ---
    print("--- Результаты 2-го блока ---")
    vec_a = np.array([1, 2, 3])
    vec_b = np.array([4, 5, 6])
    
    print(f"Сложение: {vector_add(vec_a, vec_b)}")          
    print(f"Умножение на 2: {scalar_multiply(vec_a, 2)}")   
    print(f"Поэлементное *: {elementwise_multiply(vec_a, vec_b)}") 
    print(f"Скалярное произведение: {dot_product(vec_a, vec_b)}")

    print("--- Результаты 3-го блока ---")
    # --- ТЕСТЫ 3 БЛОКА ---
    # Создадим простую квадратную матрицу 2x2
    A = np.array([[2, 1], 
                  [1, 3]])
    B = np.array([5, 10]) # Вектор b для системы Ax = b

    # 1. Умножение (A * A)
    print(f"A * A:\n{matrix_multiply(A, A)}")

    # 2. Определитель
    det = matrix_determinant(A)
    print(f"Определитель матрицы A: {det:.2f}")

    # 3. Обратная матрица
    inv_A = matrix_inverse(A)
    print(f"Обратная матрица A:\n{inv_A}")

    # 4. Решение системы
    x = solve_linear_system(A, B)
    print(f"Решение системы Ax=b: {x}") 
    # Проверка: 2*1 + 1*3 = 5; 1*1 + 3*3 = 10 -> [1, 3]

    # --- РЕЗУЛЬТАТЫ 4 И 5 БЛОКОВ ---
    print("\n--- Результаты 4-го и 5-го блоков ---")
    
    # Список возможных путей к файлу
    paths_to_try = [
        "numpy_lab/data/students_scores.csv", 
        "data/students_scores.csv"
    ]
    
    data_path = None
    for p in paths_to_try:
        if os.path.exists(p):
            data_path = p
            break

    if data_path:
        # 1. Загрузка
        data = load_dataset(data_path)
        
        # Срезаем только первый столбец и гарантируем тип float
        math_scores = data[:, 0].astype(float)
        
        # 2. Анализ
        stats = statistical_analysis(math_scores)
        print(f"Статистика по математике: {stats}")
        # 3. Визуализация
        print(f"Использую файл: {data_path}")
        print("Генерация графиков в папку 'plots'...")
        
        plot_histogram(math_scores)
        student_ids = np.arange(len(math_scores))
        plot_line(student_ids, math_scores)
        
        # Для корреляции транспонируем, чтобы предметы стали строками
        corr_matrix = np.corrcoef(data.T)
        plot_heatmap(corr_matrix)
        
        print("Готово! Проверьте папку 'plots'.")
    else:
        print("Ошибка: Файл students_scores.csv не найден!")
        print(f"Текущая папка: {os.getcwd()}") # Поможет понять, где мы находимся