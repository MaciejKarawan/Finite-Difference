import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from FD import *


def setup_plotting_style():
    """Konfiguruje styl wykresów dla całego projektu."""
    plt.rc("figure",
           autolayout=True,
           figsize=(11.69, 8.27),
           titlesize=10,
           titleweight='bold')
    plt.rc("axes",
           labelweight="bold",
           labelsize="large",
           titleweight="bold",
           titlesize=16,
           titlepad=10,
           titlelocation='center')

    sns.set_theme(style='whitegrid', context='paper', palette='dark')


def plot_fd_comparison(
        results_wo_memory,  # Dane dla metody bez pamięci
        results_with_memory,  # Dane dla metody z pamięcią
        grid_sizes,  # Lista rozmiarów siatki (NAS)
        title,  # Tytuł wykresu
        labels=None,  # Etykiety metod (domyślnie ['FD bez pamięci', 'FD z pamięcią'])
        show_time=True  # Czy pokazywać czas wykonania
):
    """
    Funkcja wizualizująca porównanie dwóch metod FD
    """
    if labels is None:
        labels = ['FD bez pamięci', 'FD z pamięcią']

    plt.figure(figsize=(12, 6))

    # Wykres różnic
    plt.plot(grid_sizes, results_wo_memory['diff'], 'b-o', label=f'{labels[0]} (różnica)')
    plt.plot(grid_sizes, results_with_memory['diff'], 'r-o', label=f'{labels[1]} (różnica)')
    plt.xlabel('Liczba kroków przestrzennych (NAS)')
    plt.ylabel('Różnica od ceny teoretycznej')
    plt.grid(True)

    if show_time:
        ax2 = plt.gca().twinx()
        ax2.plot(grid_sizes, results_wo_memory['time'], 'b--', marker='s', label=f'{labels[0]} (czas)')
        ax2.plot(grid_sizes, results_with_memory['time'], 'r--', marker='s', label=f'{labels[1]} (czas)')
        ax2.set_ylabel('Czas wykonania [s]')

    plt.title(title)

    # Łączenie legend
    lines1, labels1 = plt.gca().get_legend_handles_labels()
    if show_time:
        lines2, labels2 = ax2.get_legend_handles_labels()
        plt.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        plt.legend(lines1, labels1)

    plt.tight_layout()
    plt.show()