import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ
# ============================================================

def create_vector():
    """Создает одномерный массив (вектор) от 0 до 9 включительно."""
    return np.arange(10)


def create_matrix():
    """Создает матрицу 5x5 со случайными числами от 0 до 1."""
    return np.random.rand(5, 5)


def reshape_vector(vec):
    """Меняет форму массива из 10 элементов в таблицу 2x5."""
    return vec.reshape(2, 5)


def transpose_matrix(mat):
    """Транспонирует матрицу."""
    return mat.T

# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a, b):
    """Сложение векторов одинаковой длины без циклов."""
    return a + b


def scalar_multiply(vec, scalar):
    """Умножение каждого элемента вектора на число."""
    return vec * scalar


def elementwise_multiply(a, b):
    """Поэлементное умножение двух массивов."""
    return a * b


def dot_product(a, b):
    """Скалярное произведение векторов."""
    return np.dot(a, b)

# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a, b):
    """Классическое умножение матриц."""
    return a @ b


def matrix_determinant(a):
    """Вычисление определителя квадратной матрицы."""
    return np.linalg.det(a)


def matrix_inverse(a):
    """Нахождение обратной матрицы."""
    return np.linalg.inv(a)


def solve_linear_system(a, b):
    """Решение системы линейных уравнений Ax = b."""
    return np.linalg.solve(a, b)

# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path="data/students_scores.csv"):
    """Загружает CSV и возвращает NumPy массив чисел."""
    df = pd.read_csv(path)
    # Используем современный метод .to_numpy() вместо .values
    return df.to_numpy(dtype=float)


def statistical_analysis(data):
    """Вычисляет основные статистические показатели."""
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
    # Создаем папку plots, если она не существует
    os.makedirs("plots", exist_ok=True) 
    
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=5, color='skyblue', edgecolor='black')
    plt.title("Распределение оценок по математике")
    plt.xlabel("Баллы")
    plt.ylabel("Количество студентов")
    
    plt.savefig("plots/histogram.png")
    plt.close()

def plot_heatmap(matrix):
    """Строит тепловую карту корреляции."""
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Корреляция между предметами")
    plt.savefig("plots/heatmap.png")
    plt.close()

def plot_line(x, y):
    """Строит график зависимости оценки от номера студента."""
    os.makedirs("plots", exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, marker='o', linestyle='-', color='green')
    plt.title("Успеваемость: Студент -> Оценка (Math)")
    plt.xlabel("ID Студента")
    plt.ylabel("Балл")
    plt.grid(True)
    plt.savefig("plots/line_plot.png")
    plt.close()

if __name__ == "__main__":
    print("Запустите python3 -m pytest tests.py -v для проверки лабораторной работы.")

