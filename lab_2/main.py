import os
from typing import Dict, Union
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use('Agg')


# ============================================================
# 1. СОЗДАНИЕ И ОБРАБОТКА МАССИВОВ
# ============================================================

def create_vector() -> np.ndarray:
    """Создает одномерный массив (вектор) от 0 до 9 включительно."""
    return np.arange(10)


def create_matrix() -> np.ndarray:
    """Создает матрицу 5x5 со случайными числами от 0 до 1."""
    return np.random.rand(5, 5)


def reshape_vector(vec: np.ndarray) -> np.ndarray:
    """
    Меняет форму массива из 10 элементов в таблицу 2x5.
    
    Args:
        vec: Входной массив из 10 элементов.
    Returns:
        Массив формы (2, 5).
    """
    return vec.reshape(2, 5)


def transpose_matrix(mat: np.ndarray) -> np.ndarray:
    """Транспонирует матрицу."""
    return mat.T


# ============================================================
# 2. ВЕКТОРНЫЕ ОПЕРАЦИИ
# ============================================================

def vector_add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Сложение векторов одинаковой длины без циклов."""
    return a + b


def scalar_multiply(vec: np.ndarray, scalar: float) -> np.ndarray:
    """Умножение каждого элемента вектора на число."""
    return vec * scalar


def elementwise_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Поэлементное умножение двух массивов."""
    return a * b


def dot_product(a: np.ndarray, b: np.ndarray) -> Union[float, np.int64]:
    """Скалярное произведение векторов."""
    return np.dot(a, b)


# ============================================================
# 3. МАТРИЧНЫЕ ОПЕРАЦИИ
# ============================================================

def matrix_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Классическое умножение матриц."""
    return a @ b


def matrix_determinant(a: np.ndarray) -> float:
    """Вычисление определителя квадратной матрицы."""
    return float(np.linalg.det(a))


def matrix_inverse(a: np.ndarray) -> np.ndarray:
    """Нахождение обратной матрицы."""
    return np.linalg.inv(a)


def solve_linear_system(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Решение системы линейных уравнений Ax = b."""
    return np.linalg.solve(a, b)


# ============================================================
# 4. СТАТИСТИЧЕСКИЙ АНАЛИЗ
# ============================================================

def load_dataset(path: str = "data/students_scores.csv") -> np.ndarray:
    """Загружает CSV и возвращает NumPy массив чисел."""
    df = pd.read_csv(path)
    return df.to_numpy(dtype=float)


def statistical_analysis(data: np.ndarray) -> Dict[str, float]:
    """
    Вычисляет основные статистические показатели.
    
    Returns:
        Словарь со средним, медианой, ст. отклонением и др.
    """
    return {
        "mean": float(np.mean(data)),
        "median": float(np.median(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "q1": float(np.percentile(data, 25)),
        "q3": float(np.percentile(data, 75))
    }


def normalize_data(data: np.ndarray) -> np.ndarray:
    """Выполняет Min-Max нормализацию данных в диапазон [0, 1]."""
    d_min = np.min(data)
    d_max = np.max(data)
    if d_max == d_min:
        return np.zeros_like(data)
    return (data - d_min) / (d_max - d_min)


# ============================================================
# 5. ВИЗУАЛИЗАЦИЯ
# ============================================================

def plot_histogram(data: np.ndarray):
    """Строит и сохраняет гистограмму распределения баллов."""
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.hist(data, bins=5, color='skyblue', edgecolor='black')
    plt.title("Распределение оценок")
    plt.xlabel("Баллы")
    plt.ylabel("Количество")
    plt.savefig("plots/histogram.png")
    plt.close()


def plot_heatmap(matrix: np.ndarray):
    """Строит тепловую карту корреляции между предметами."""
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Корреляция предметов")
    plt.savefig("plots/heatmap.png")
    plt.close()

def plot_line(x: np.ndarray, y: np.ndarray):
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
    print("Выполните запуск тестов: python -m pytest tests.py -v")