# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 17:44:03 2026

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
from sklearn.metrics import precision_score, recall_score, accuracy_score
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
        класс 1 (метка 1) — точки вида 1 (a, b − 0.5, aε);
        класс 0 (метка 0) — точки видов 2 и 3.
    Возвращает (X_binary, y_binary).
    """
    mask_group1 = (y == 0)          # точки вида 1
    mask_group2 = (y == 1) | (y == 2)  # точки видов 2 и 3

    X_group1 = X[mask_group1]
    X_group2 = X[mask_group2]

    X_binary = np.vstack((X_group1, X_group2))
    y_binary = np.concatenate((
        np.ones(len(X_group1), dtype=int),   # вид 1 → класс 1
        np.zeros(len(X_group2), dtype=int),  # виды 2 и 3 → класс 0
    ))
    return X_binary, y_binary


#Классификаторы

def build_linear_classifier():
    """Линейная пороговая классификация"""
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
    """Гауссов процесс с ядром в виде радиальных базисных функций)."""
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
        'accuracy': accuracy_score(y_test, y_pred),
    }


def run_experiment(X_bin, y_bin, title, classifiers, test_size=0.3, seed=42):
    """Делим выборку на train/test и обучаем все классификаторы."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_bin)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_bin, test_size=test_size, random_state=seed, stratify=y_bin,
    )

    # print("\n" + "=" * 72)
    # print(f"ЭКСПЕРИМЕНТ: {title}")
    # print(f"Размер выборки: {len(X_bin)}  (train={len(X_train)}, test={len(X_test)})")
    # print("=" * 72)

    results = []
    for name, factory in classifiers:
        res = evaluate_model(name, factory(), X_train, y_train, X_test, y_test)
        res['title'] = title
        res['X_test'] = X_test
        res['y_test'] = y_test
        results.append(res)
    return results



def print_comparison(results_old, results_new, metric_name):
    print("\n" + "=" * 78)
    print(f"Сравнение {metric_name.upper()}: прежняя выборка vs новая выборка")
    print("=" * 78)
    header = (f"{'Метод':<38} | {'класс':>5} | "
              f"{'старая':>8} | {'новая':>8} | {'Δ':>7}")
    print(header)
    print("-" * len(header))

    for r_old, r_new in zip(results_old, results_new):
        name = r_old['name']
        for cls in sorted(set(list(r_old[metric_name].keys()) +
                              list(r_new[metric_name].keys()))):
            v_old = r_old[metric_name].get(cls, 0.0)
            v_new = r_new[metric_name].get(cls, 0.0)
            delta = v_new - v_old
            print(f"{name:<38} | {cls:>5} | {v_old:>8.3f} | {v_new:>8.3f} | {delta:>+7.3f}")
        print("-" * len(header))


def make_conclusion(results_old, results_new, metric_name):
    print("\n" + "=" * 78)
    print("Выводы о сохранении точностных характеристик")
    print("=" * 78)
    for r_old, r_new in zip(results_old, results_new):
        name = r_old['name']
        diffs = []
        for cls in sorted(set(list(r_old[metric_name].keys()) +
                              list(r_new[metric_name].keys()))):
            v_old = r_old[metric_name].get(cls, 0.0)
            v_new = r_new[metric_name].get(cls, 0.0)
            diffs.append(abs(v_new - v_old))
        max_diff = max(diffs) if diffs else 0.0
        if max_diff < 0.03:
            verdict = "характеристики СОХРАНЯЮТСЯ (отклонение < 0.03)"
        elif max_diff < 0.07:
            verdict = "характеристики СОХРАНЯЮТСЯ УДОВЛЕТВОРИТЕЛЬНО (отклонение 0.03–0.07)"
        else:
            verdict = "характеристики ЗАМЕТНО МЕНЯЮТСЯ (отклонение > 0.07)"
        print(f"\n• {name}")
        print(f"  максимальное отклонение {metric_name}: {max_diff:.3f}")
        print(f"  → {verdict}")


# Визуализация


CLASS_COLORS = {
    0: '#1f77b4',   # синий  — класс 0 (виды 2 и 3)
    1: '#d62728',   # красный — класс 1 (вид 1)
}


def plot_3d_scatter(ax, X, y, title, marker_size=18, alpha=0.75):
    """3D-облако истинных классов (для бинарной задачи — два цвета)."""
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
    - заливка — истинный класс;
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



if __name__ == "__main__":
    classifiers = [
        ("Линейная пороговая классификация", build_linear_classifier),
        ("Гауссова смесь (3 компоненты)", build_gmm_classifier),
        ("Гауссов процесс с ядром", build_gpc_classifier),
    ]

    # Прежняя выборка
    X_old, y_old = generate_dataset(NumSamples=300, seed=42)
    X_bin_old, y_bin_old = split_into_two_groups(X_old, y_old)
    results_old = run_experiment(
        X_bin_old, y_bin_old,
        "ПРЕЖНЯЯ выборка (seed=42)",
        classifiers, test_size=0.3, seed=42,
    )

    # Новая выборка
    # Другой seed → новые случайные значения a, b, ε при тех же распределениях.
    X_new, y_new = generate_dataset(NumSamples=300, seed=2025)
    X_bin_new, y_bin_new = split_into_two_groups(X_new, y_new)
    results_new = run_experiment(
        X_bin_new, y_bin_new,
        "НОВАЯ выборка (seed=2025)",
        classifiers, test_size=0.3, seed=42,
    )

    # Сравнение
    print_comparison(results_old, results_new, 'precision')
    print_comparison(results_old, results_new, 'recall')

    # Вывод
    make_conclusion(results_old, results_new, 'precision')
    make_conclusion(results_old, results_new, 'recall')
    
    
    # ============================================================
    # РИСУНОК 1. Истинные классы — 3D-облако (прежняя выборка)
    # ============================================================
    # Масштабируем ещё раз только для визуализации (модели уже обучились
    # на своей копии, но график должен быть в тех же координатах).
    scaler_vis = StandardScaler()
    X_bin_old_vis = scaler_vis.fit_transform(X_bin_old)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    plot_3d_scatter(
        ax, X_bin_old_vis, y_bin_old,
        'Истинное распределение классов (прежняя выборка, seed=42)',
    )
    plt.tight_layout()
    plt.show()

    # ============================================================
    # РИСУНОК 2. Предсказания трёх моделей на тестовой выборке — 3D
    # (прежняя выборка)
    # ============================================================
    fig = plt.figure(figsize=(20, 7))
    for k, r in enumerate(results_old, start=1):
        ax = fig.add_subplot(1, 3, k, projection='3d')
        plot_predictions_3d(
            ax, r['X_test'], r['y_test'], r['y_pred'],
            f"{r['name']}\naccuracy = {r['accuracy']:.3f}",
        )
    plt.suptitle(
        'Предсказания моделей на тестовой выборке — ПРЕЖНЯЯ выборка '
        '(красная обводка — ошибка)',
        fontsize=14,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()

    # ============================================================
    # РИСУНОК 3. Истинные классы — 3D-облако (новая выборка)
    # ============================================================
    scaler_vis_new = StandardScaler()
    X_bin_new_vis = scaler_vis_new.fit_transform(X_bin_new)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')
    plot_3d_scatter(
        ax, X_bin_new_vis, y_bin_new,
        'Истинное распределение классов (новая выборка, seed=2025)',
    )
    plt.tight_layout()
    plt.show()

    # ============================================================
    # РИСУНОК 4. Предсказания трёх моделей на тестовой выборке — 3D
    # (новая выборка)
    # ============================================================
    fig = plt.figure(figsize=(20, 7))
    for k, r in enumerate(results_new, start=1):
        ax = fig.add_subplot(1, 3, k, projection='3d')
        plot_predictions_3d(
            ax, r['X_test'], r['y_test'], r['y_pred'],
            f"{r['name']}\naccuracy = {r['accuracy']:.3f}",
        )
    plt.suptitle(
        'Предсказания моделей на тестовой выборке — НОВАЯ выборка '
        '(красная обводка — ошибка)',
        fontsize=14,
    )
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()    

