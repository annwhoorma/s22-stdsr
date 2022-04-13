#%%
import matplotlib.pyplot as plt

times = {(0.001, 100000): 1.0642704963684082, (0.001, 1000000): 10.898213386535645, (0.001, 10000000): 123.12200450897217, (0.005, 100000): 1.2010586261749268, (0.005, 1000000): 11.381516933441162, (0.005, 10000000): 169.79749155044556, (0.01, 100000): 1.104417085647583, (0.01, 1000000): 13.069535732269287, (0.01, 10000000): 232.23767590522766}
possible_e = [0.001, 0.005, 0.01, 0.05, 0.1]
values_per_error = {error: ([], []) for error in possible_e}

def plot_graph(x, y, title, xlabel, ylabel):
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot()
    ax.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

for error_size, time_taken in times.items():
    values_per_error[error_size[0]][0].append(error_size[1])
    values_per_error[error_size[0]][1].append(time_taken)

for error in possible_e:
    print(error, values_per_error[error][0], values_per_error[error][1])
    plot_graph(values_per_error[error][0], values_per_error[error][1], f'At error {error}', 'dataset size', 'time taken (s)')