# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 20:37:30 2026

@author: Ольга
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (нужно для 3D-проекции)
from matplotlib.lines import Line2D


from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def generate_samples_1(NumSamples=100):
    '''Функция генерации выборки 
    a ~ U(-1, 1),  b ~ U(-1, 1),  ε ~ U(-1, 1),
    точки вида (a, b − 0.5, aε).
    '''
    a = np.random.uniform(low=-1, high=1, size=(NumSamples, 1))
    b = np.random.uniform(low=-1, high=1, size=(NumSamples, 1))
    epsilon = np.random.uniform(low=-1, high=1, size=(NumSamples, 1))
    x = a
    y = b - 0.5
    z = a * epsilon
    return (x, y, z)


def generate_samples_2(NumSamples=100):
    '''Функция генерации выборки
    a ~ N(0, 0.5²),  b ~ N(0, 0.5²),  ε ~ N(0, 0.5²),
    точки вида (a, b + ε, 0.5 + a + ε).
    '''
    a = np.random.normal(loc=0, scale=0.5, size=(NumSamples, 1))
    b = np.random.normal(loc=0, scale=0.5, size=(NumSamples, 1))
    epsilon = np.random.normal(loc=0, scale=0.5, size=(NumSamples, 1))
    x = a
    y = b + epsilon
    z = 0.5 + a + epsilon
    return (x, y, z)


def generate_samples_3(NumSamples=100):
    '''Функция генерации выборки
    a ~ N(0, 0.5²),  b ~ N(0, 0.5²),  ε ~ N(0, 0.5²),
    точки вида (0.1a − 1, 3b, ε − 1).
    '''
    a = np.random.normal(loc=0, scale=0.5, size=(NumSamples, 1))
    b = np.random.normal(loc=0, scale=0.5, size=(NumSamples, 1))
    epsilon = np.random.normal(loc=0, scale=0.5, size=(NumSamples, 1))
    x = 0.1 * a - 1
    y = 3 * b
    z = epsilon - 1
    return (x, y, z)




def generate_dataset(NumSamples=100, seed=None):
    '''Возвращает (X, y) с тремя группами точек (метки 0, 1, 2).'''
    if seed is not None:
        np.random.seed(seed)
    x1, y1, z1 = generate_samples_1(NumSamples)
    x2, y2, z2 = generate_samples_2(NumSamples)
    x3, y3, z3 = generate_samples_3(NumSamples)

    X = np.vstack((
        np.hstack((x1, y1, z1)),
        np.hstack((x2, y2, z2)),
        np.hstack((x3, y3, z3)),
    ))
    y = np.concatenate((
        np.zeros(NumSamples, dtype=int),
        np.ones(NumSamples, dtype=int),
        np.full(NumSamples, 2, dtype=int),
    ))
    return X, y


def split_into_two_groups(X, y):
    """
    Разбивает выборку на 2 группы:
        класс 1 (метка 1) — точки вида 3;
        класс 0 (метка 0) — точки видов 1 и 2.
    Возвращает (X_binary, y_binary).
    """
    mask_group1 = (y == 2)          # точки вида 3
    mask_group2 = (y == 0) | (y == 1)  # точки видов 1 и 2

    X_group1 = X[mask_group1]
    X_group2 = X[mask_group2]

    X_binary = np.vstack((X_group1, X_group2))
    y_binary = np.concatenate((
        np.ones(len(X_group1), dtype=int),   # вид 3 → класс 1
        np.zeros(len(X_group2), dtype=int),  # виды 1 и 2 → класс 0
    ))
    return X_binary, y_binary


#Классификаторы

def build_linear_classifier():
    """Линейная пороговая классификация."""
    return LogisticRegression(
        C=1.0,
        max_iter=5000,
        solver='lbfgs',
    )


class GMMClassifier:
    """
    Классификатор на основе гауссовой смеси:
    для каждого класса строится своя GMM, предсказание ---
    argmax по логарифму правдоподобия.
    """

    def __init__(self, n_components=2, reg_covar=1e-3, seed=0):
        self.n_components = n_components
        self.reg_covar = reg_covar
        self.seed = seed
        self.models_ = {}
        self.classes_ = None

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        for c in self.classes_:
            Xc = X[y == c]
            n_comp = min(self.n_components, max(1, len(Xc)))
            gm = GaussianMixture(
                n_components=n_comp,
                covariance_type='full',
                reg_covar=self.reg_covar,
                random_state=self.seed,
                max_iter=1000,
            )
            gm.fit(Xc)
            self.models_[c] = gm
        return self

    def predict(self, X):
        scores = np.column_stack([
            self.models_[c].score_samples(X) for c in self.classes_
        ])
        return self.classes_[np.argmax(scores, axis=1)]


def build_gmm_classifier():
    return GMMClassifier(n_components=3, reg_covar=1e-3, seed=0)


def build_gpc_classifier():
    """Гауссов процесс с ядром в виде радиальных базисных функций."""
    kernel = C(1.0, (1e-2, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e3))
    return GaussianProcessClassifier(
        kernel=kernel,
        optimizer='fmin_l_bfgs_b',
        n_restarts_optimizer=3,
        random_state=0,
        max_iter_predict=300,
    )


#Оценка

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    classes = np.unique(y_test)
    precisions = precision_score(y_test, y_pred, labels=classes,
                                 average=None, zero_division=0)
    recalls = recall_score(y_test, y_pred, labels=classes,
                           average=None, zero_division=0)



    return {
        'name': name,
        'precision': dict(zip(classes.tolist(), precisions.tolist())),
        'recall': dict(zip(classes.tolist(), recalls.tolist())),
        'y_pred': y_pred,
    }


def run_experiment(X_bin, y_bin, title, classifiers, test_size=0.3, seed=42):
    """Делим на train/test и обучаем все классификаторы."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bin)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_bin, test_size=test_size, random_state=seed, stratify=y_bin,
    )


    results = []
    for name, factory in classifiers:
        res = evaluate_model(name, factory(), X_train, y_train, X_test, y_test)
        res['title'] = title
        results.append(res)
    return results



#Визуализация

CLASS_COLORS = {
    0: '#1f77b4',   # синий
    1: '#d62728',   # красный
}


def plot_3d_scatter(ax, X, y, title, marker_size=18, alpha=0.75):
    """3D-облако истинных классов (для бинарной задачи два цвета)."""
    for c in np.unique(y):
        mask = (y == c)
        ax.scatter(
            X[mask, 0], X[mask, 1], X[mask, 2],
            c=CLASS_COLORS[c], s=marker_size, alpha=alpha,
            edgecolors='k', linewidths=0.3,
            label=f'класс {c}',
        )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=11)
    ax.legend(loc='upper left', fontsize=9)


def plot_predictions_3d(ax, X, y_true, y_pred, title, marker_size=20):
    """
    Предсказания модели в 3D:
    - заливка точки — истинный класс (цвет);
    - чёрная обводка — модель угадала;
    - красная толстая обводка — ошибка.
    """
    correct = (y_true == y_pred)
    for c in np.unique(y_true):
        mask = (y_true == c)
        ax.scatter(
            X[mask & correct, 0], X[mask & correct, 1], X[mask & correct, 2],
            c=CLASS_COLORS[c], s=marker_size, alpha=0.75,
            edgecolors='black', linewidths=0.4,
        )
        ax.scatter(
            X[mask & ~correct, 0], X[mask & ~correct, 1], X[mask & ~correct, 2],
            c=CLASS_COLORS[c], s=marker_size * 1.5, alpha=1.0,
            edgecolors='red', linewidths=1.6,
        )
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=11)

    handles = [
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor=CLASS_COLORS[c], markeredgecolor='black',
               markersize=9, label=f'класс {c} — верно')
        for c in sorted(np.unique(y_true))
    ]
    handles.append(
        Line2D([0], [0], marker='o', color='w',
               markerfacecolor='gray', markeredgecolor='red',
               markeredgewidth=1.6, markersize=10, label='ошибка')
    )
    ax.legend(handles=handles, loc='upper left', fontsize=8)
 
#Основной сценарий

if __name__ == "__main__":
    np.random.seed(42)
    X, y = generate_dataset(NumSamples=300, seed=42)

    # Объединяем точки видов 2 и 3 в один класс
    X_bin, y_bin = split_into_two_groups(X, y)
    print("Распределение классов:", dict(zip(*np.unique(y_bin, return_counts=True))))

    # Масштабирование признаков — полезно для GPC и GMM
    scaler = StandardScaler()
    X_bin_scaled = scaler.fit_transform(X_bin)

    X_train, X_test, y_train, y_test = train_test_split(
        X_bin_scaled, y_bin, test_size=0.3, random_state=42, stratify=y_bin,
    )

    classifiers = [
        ("Линейная пороговая классификация", build_linear_classifier),
        ("Гауссова смесь (GMM, 3 компоненты)", build_gmm_classifier),
        ("Гауссов процесс (RBF-ядро)", build_gpc_classifier),
    ]
    
    results = []
    for name, factory in classifiers:
        model = factory()
        res = evaluate_model(name, model, X_train, y_train, X_test, y_test)
        # accuracy на тесте — доля правильных предсказаний
        res['accuracy'] = (res['y_pred'] == y_test).mean()
        results.append(res)

    # ---------------------- Сводная таблица ----------------------
    print("\n" + "=" * 78)
    print("Сводная таблица (precision / recall по классам)")
    print("=" * 78)
    header = header = f"{'Метод':<38} | {'P(кл.0)':>8} | {'R(кл.0)':>8} | {'P(кл.1)':>8} | {'R(кл.1)':>8}"
    print(header)
    print("-" * len(header))
    for r in results:
        p0 = r['precision'].get(0, 0.0)
        r0 = r['recall'].get(0, 0.0)
        p1 = r['precision'].get(1, 0.0)
        r1 = r['recall'].get(1, 0.0)
        print(f"{r['name']:<38} | {p0:>8.3f} | {r0:>8.3f} | {p1:>8.3f} | {r1:>8.3f}")

    # ============================================================
    # РИСУНОК 1. Истинные классы — 3D-облако
    # ============================================================
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    plot_3d_scatter(
        ax, X_bin_scaled, y_bin,
        'Истинное распределение классов (масштабированные признаки)',
    )
    plt.tight_layout()
    plt.show()

    # ============================================================
    # РИСУНОК 2. Предсказания трёх моделей на тестовой выборке — 3D
    # ============================================================
    fig = plt.figure(figsize=(20, 7))
    for k, r in enumerate(results, start=1):
        ax = fig.add_subplot(1, 3, k, projection='3d')
        plot_predictions_3d(
            ax, X_test, y_test, r['y_pred'],
            f"{r['name']}\naccuracy = {r['accuracy']:.3f}",
        )
    plt.suptitle(
        'Предсказания моделей на тестовой выборке (красная обводка — ошибка)',
        fontsize=14,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()

