from email.policy import default
from typing import Optional
import click
from numpy import mean

from new_algorithm import NewAlgorithm
from mrl98 import Buffer, MRL98, Element
import datagen
import numpy_solver

from scipy.stats import ttest_ind


possible_e = [0.001, 0.005, 0.01, 0.05, 0.1]
possible_N = [10**5, 10**7, 10**9]
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
    (0.001, 10**7): {'b': 5, 'k': 5495},
    (0.001, 10**9): {'b': 10, 'k': 5954},
    # --- e=0.005
    (0.005, 10**5): {'b': 3, 'k': 953},
    (0.005, 10**7): {'b': 8, 'k': 875},
    (0.005, 10**9): {'b': 7, 'k': 2106},
    # --- e=0.01
    (0.01, 10**5): {'b': 7, 'k': 217},
    (0.01, 10**7): {'b': 9, 'k': 412},
    (0.01, 10**9): {'b': 10, 'k': 765},
    # --- e=0.05
    (0.05, 10**5): {'b': 6, 'k': 78},
    (0.05, 10**7): {'b': 8, 'k': 129},
    (0.05, 10**9): {'b': 8, 'k': 235},
    # --- e=0.1
    (0.1, 10**5): {'b': 5, 'k': 55},
    (0.1, 10**7): {'b': 10, 'k': 60},
    (0.1, 10**9): {'b': 12, 'k': 77},
}

def get_distribution(name: str, N: int, poisson_lambda: Optional[float], random_start: Optional[int], random_finish: Optional[int]):
    gen_func = possible_d[name]
    if name == 'poisson':
        return gen_func(N, poisson_lambda)
    if name == 'random':
        return gen_func(N, random_start, random_finish)
    return gen_func(N)


def run_several_times(runs: int, d, n, b, k, phi):
    mrl98_values = []
    numpy_values = []
    for _ in range(runs):
        data = possible_d[d](n).tolist()
        nalg = NewAlgorithm(data.copy(), b, k, phi)
        value_at_phi = nalg.run()
        numpy_value_at_phi = numpy_solver.calculate_quantile(data, phi)
        mrl98_values.append(value_at_phi)
        numpy_values.append(numpy_value_at_phi)
    return mrl98_values, numpy_values


@click.command()
@click.argument('phi', type=float)
@click.argument('runs', type=int, default=1)
@click.argument('e', type=float, default=possible_e[0])
@click.argument('n', type=int, default=possible_N[0])
@click.argument('d', type=str, default='normal')
@click.argument('poisson_lambda', type=int, default=5)
@click.argument('random_start', type=int, default=0)
@click.argument('random_finish', type=int, default=10)
def main(phi, runs, e, n, d, poisson_lambda, random_start, random_finish):
    assert 0 <= phi <= 1, 'phi must be between 0 and 1'
    assert e in possible_e, f'e must be from {possible_e}'
    assert n in possible_N, f'n must be from {possible_N}'
    assert d in possible_d.keys(), f'd must be from {list(possible_d.keys())}'
    params = parameters[(e, n)]
    b, k = params['b'], params['k']
    print('The Null Hypothesis (H0): MRL98 is as good as np.quantile()')
    print(f'Allowed error rate: {e}')
    print(f'Number of runs: {runs}\n')
    mrl98_values, numpy_values = run_several_times(runs, d, n, b, k, phi)
    print('Done. Running 2-sample t-test...\n')
    _, p_value = ttest_ind(a=mrl98_values, b=numpy_values, equal_var=False)
    print(f'Mean of MRL98 results: {mean(mrl98_values)}')
    print(f'Mean of numpy results: {mean(numpy_values)}')
    print(f'p_value of 2-sample t-test: {p_value}')
    print(f'Rejecting H0: {p_value > e}')

if __name__ == '__main__':
    main()