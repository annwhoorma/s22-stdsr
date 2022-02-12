from typing import Optional
import click
from numpy import dtype, mean

from new_algorithm import NewAlgorithm
from mrl98 import Buffer, MRL98, Element
import datagen
import numpy_solver

from scipy.stats import ttest_ind

import matplotlib.pyplot as plt
from time import time

P_VALUE_THRESH = .05

possible_e = [0.001, 0.005, 0.01, 0.05, 0.1]
possible_N = [10**5, 10**6, 10**7, 10**9]
possible_d = {
    'random': datagen.generate_random,
    'bimodal': datagen.generate_bimodal_normal,
    'poisson': datagen.generate_poisson,
    'normal': datagen.generate_normal
}
# b = number of buffers, k = number of elements per buffer
parameters = {
    # --- e=0.001
    (0.001, 10**5): {'b': 3, 'k': 2778},
    (0.001, 10**6): {'b': 5, 'k': 3031},
    (0.001, 10**7): {'b': 5, 'k': 5495},
    # --- e=0.005
    (0.005, 10**5): {'b': 3, 'k': 953},
    (0.005, 10**6): {'b': 8, 'k': 583},
    (0.005, 10**7): {'b': 8, 'k': 875},
    # --- e=0.01
    (0.01, 10**5): {'b': 7, 'k': 217},
    (0.01, 10**6): {'b': 12, 'k': 229},
    (0.01, 10**7): {'b': 9, 'k': 412},
    # --- e=0.05
    (0.05, 10**5): {'b': 6, 'k': 78},
    (0.05, 10**6): {'b': 6, 'k': 117},
    (0.05, 10**7): {'b': 8, 'k': 129},
    # --- e=0.1
    (0.1, 10**5): {'b': 5, 'k': 55},
    (0.1, 10**6): {'b': 7, 'k': 54},
    (0.1, 10**7): {'b': 10, 'k': 60},
    (0.1, 10**8): {'b': 15, 'k': 51},
}

def get_distribution(name: str, N: int, poisson_lambda: Optional[float], random_start: Optional[int], random_finish: Optional[int]):
    gen_func = possible_d[name]
    if name == 'poisson':
        return gen_func(N, poisson_lambda)
    if name == 'random':
        return gen_func(N, random_start, random_finish)
    return gen_func(N)

def run_several_times(mrl_type: str, runs: int, d, n, b, k, phi):
    mrl98_values = []
    numpy_values = []
    for _ in range(runs):
        data = possible_d[d](n).tolist()
        nalg = NewAlgorithm(mrl_type, data.copy(), b, k, phi)
        value_at_phi = nalg.run()
        numpy_value_at_phi = numpy_solver.calculate_quantile(data, phi)
        mrl98_values.append(value_at_phi)
        numpy_values.append(numpy_value_at_phi)
    return mrl98_values, numpy_values

# @profile
def track_memory_usage(mrl_type: str, d, n, b, k, phi):
    data = possible_d[d](n).tolist()
    nalg = NewAlgorithm(mrl_type, data, b, k, phi)
    # nalg.run()

def plot_graph(x, y, title, xlabel, ylabel):
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot()
    ax.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


@click.command()
@click.argument('mrl_year', type=str, default=98)
@click.argument('phi', type=float, default=0.2)
@click.argument('runs', type=int, default=1)
@click.argument('e', type=float, default=possible_e[4])
@click.argument('n', type=int, default=possible_N[2])
@click.argument('d', type=str, default='normal')
@click.argument('poisson_lambda', type=int, default=5)
@click.argument('random_start', type=int, default=0)
@click.argument('random_finish', type=int, default=10)
def main(mrl_year, phi, runs, e, n, d, poisson_lambda, random_start, random_finish):
    assert '98' in mrl_year or '99' in mrl_year, 'MRL year should contain 98 or 99'
    assert 0 <= phi <= 1, 'phi must be between 0 and 1'
    assert e in possible_e, f'e must be from {possible_e}'
    assert n in possible_N, f'n must be from {possible_N}'
    assert d in possible_d.keys(), f'd must be from {list(possible_d.keys())}'
    params = parameters[(e, n)]
    b, k = params['b'], params['k']
    print(f'The Null Hypothesis (H0): MRL98 is as good as np.quantile() with a p-value threshold of {P_VALUE_THRESH}')
    print(f'Allowed error rate: {e}')
    print(f'Number of runs: {runs}\n')
    # track_memory_usage(mrl_year, d, n, b, k, phi)
    mrl_values, numpy_values = run_several_times(mrl_year, runs, d, n, b, k, phi)

    print('Done. Running 2-sample t-test...\n')
    print(f'Mean of MRL results of {runs} runs: {mean(mrl_values)}')
    print(f'Mean of numpy results of {runs} runs: {mean(numpy_values)}')
    _, p_value = ttest_ind(a=mrl_values, b=numpy_values, equal_var=False)
    print(f'p_value of 2-sample t-test: {p_value}')
    print(f'Rejecting H0: {p_value > P_VALUE_THRESH}')
    times = run_for_dataset_size_and_error(mrl_year, d, phi) # {(error, size): time_taken}
    print(times)
    values_per_error = {error: ([], []) for error in possible_e}
    for error_size, time_taken in times.items():
        values_per_error[error_size[0]][0].append(error_size[1])
        values_per_error[error_size[0]][1].append(time_taken)
    print('\nPlotting graphs...\n')
    for error in possible_e:
        plot_graph(values_per_error[error][0], values_per_error[error][1], f'At error {error}', 'dataset size', 'time taken (s)')


def run_for_dataset_size_and_error(mrl_type: str, d, phi):
    times = {}
    for error_size, params in parameters.items():
        error, size = error_size
        print(f'Running for size: {size} and error: {error}')
        b, k = params['b'], params['k']
        data = possible_d[d](size).tolist()
        start = time()
        nalg = NewAlgorithm(mrl_type, data.copy(), b, k, phi)
        value_at_phi = nalg.run()
        end = time()
        times[error_size] = end - start
        print(f'Value at {phi}: {value_at_phi}/n')
        print(times)
    return times


if __name__ == '__main__':
    main()