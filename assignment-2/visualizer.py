# ref: Alfiya Musabekova
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

fig, ax = plt.figure(figsize=(20, 20)), plt.axes()
plt.xlabel('longitude')
plt.ylabel('latitude')
ln, = plt.plot([], [], lw=1)

def init():
    ax.set_xlim(35, 155)
    ax.set_ylim(40, 60)
    ln.set_data([], [])
    return ln,

def animate_me(cities, all_states, cities_coords, dirname, filename):
    def update(i):
        xdata = []
        ydata = []
        state = all_states[i]
        path = state['cities']
        for p in path:
            ydata.append(cities_coords[p].lat)
            xdata.append(cities_coords[p].lon)
        ax.title.set_text(f"T = {state['temp']}, optimal_dist = {state['distance']}")
        ln.set_data(xdata, ydata)
        return ln,

    anim = FuncAnimation(fig, update, frames=len(all_states), init_func=init, blit=True)
    for city in cities:
        ax.annotate(city, (cities_coords[city].lon, cities_coords[city].lat))
    anim.save(f'{dirname}/{filename}.gif', fps=5)
