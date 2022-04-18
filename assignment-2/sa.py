from pathlib import Path
import pandas as pd
import numpy as np
from math import sin, cos, pi, sqrt, exp
from enum import Enum
import random
import plotly.graph_objects as go
from geopy.distance import geodesic
from tqdm import tqdm
from visualizer import animate_me

PATH = Path('russian_cities.csv')
EARTHR = 6371

# ------ PREPROCESSING -------

class GeoCoordinate:
    def __init__(self, coord):
        self.lat, self.lon = coord

def read_csv(path: Path, sort_by='population', take_sorted_n=30) -> pd.DataFrame:
    usecols = ['address', 'population', 'city_type', 'geo_lat', 'geo_lon']
    df = pd.read_csv(path, header=0, usecols=usecols, keep_default_na=False)
    df = df[df['city_type'] == 'г']
    df = df.sort_values(by=sort_by, ascending=False)
    df = df[:take_sorted_n]
    df = df.drop(['population', 'city_type'], axis=1)
    df.index = range(len(df))
    return df

def create_distance_matrix(cities_df: pd.DataFrame) -> 'tuple[list, dict[str, dict[str, float]]]':
    cities = cities_df['address'].to_list()
    num_cities = len(cities_df)
    distance_dict = {city: {city_: 0 for city_ in cities} for city in cities}
    coordinates_dict = {}
    for i in range(num_cities):
        coord1 = cities_df.iloc[[i]][['geo_lat', 'geo_lon']].values[0]
        for j in range(num_cities):
            coord2 = cities_df.iloc[[j]][['geo_lat', 'geo_lon']].values[0]
            dist = geodesic(coord1, coord2).km
            distance_dict[cities[i]][cities[j]] = dist
        coordinates_dict[cities[i]] = GeoCoordinate(coord1)
    return cities, distance_dict, coordinates_dict

# -------------------------------------------------------------------------
# Annealing Rate
class Cooling(Enum):
    Fast = 0.79
    Mild = 0.89
    Slow = 0.99

class SimulatedAnnealing:
    T = 10000
    def __init__(self, cities_distances, cities_coords, cooling_type: Cooling, decr_T_every_n_iters: int, T_lower: int):
        self.state = {}
        self.all_states = []

        self.cities_distances = cities_distances
        self.cities_coords = cities_coords
        self.annealing_rate = cooling_type.value
        self.T_lower = T_lower
        self.decr_T_every_n_iters = decr_T_every_n_iters
        self._generate_initial_state()

    def _p_star(self, distance):
        return exp(- distance / self.T)

    def _alpha(self, new_state_distance):
        print(self.state['distance'], self._p_star(self.state['distance']))
        return self._p_star(new_state_distance) / self._p_star(self.state['distance'])

    def _accept_new_state(self, new_state):
        alpha = self._alpha(new_state['distance'])
        u = random.uniform(0, 1)
        return True if u <= alpha else False

    def _generate_initial_state(self):
        cities = random.sample(self.cities_distances.keys(), k=len(self.cities_distances))
        distance = self._calculate_distance(cities)
        self.state = {'cities': cities, 'distance': distance}

    def _update_state(self):
        i, j = random.sample(range(len(self.state['cities'])), k=2)
        new_cities = self.state['cities'].copy()
        new_cities[i], new_cities[j] = new_cities[j], new_cities[i]
        new_distance = self._calculate_distance(new_cities)
        new_state = {'cities': new_cities, 'distance': new_distance}
        # if the new solution is better, update the state
        if self._accept_new_state(new_state):
            self.state = new_state

    def _calculate_distance(self, cities):
        distance = 0
        for i in range(len(cities[:-1])):
            a = cities[i]
            b = cities[i + 1]
            try:
                distance += self.cities_distances[a][b]
            except:
                distance += self.cities_distances[b][a]
        return distance

    def _continue_running(self):
        return self.T_lower < self.T

    def _update_T(self):
        if self.current_iteration % self.decr_T_every_n_iters == 0:
            print(f'T={self.T}')
            self.T = self.T * self.annealing_rate

    def run(self):
        self.current_iteration = 0
        while self._continue_running():
            self._update_state()
            self._update_T()
            self._save_state()

    def _save_state(self):
        cities = self.state['cities']
        distance = self.state['distance']
        title = f'Iteration: {self.current_iteration}; Distance: {distance} km'
        self.all_states.append(
            {
                'title': title,
                'cities': cities,
                'distance': round(distance, 2),
                'temp': round(self.T, 2),
            }
        )


if __name__ == '__main__':
    dirname = 'gifs'
    Path(dirname).mkdir(exist_ok=True)
    coolings = [Cooling.Slow, Cooling.Mild, Cooling.Fast]
    decr_T_every_n_iters = 1
    for cooling in coolings:
        # try:
        cities_df = read_csv(PATH, take_sorted_n=30)
        cities, distance_dict, coordinates_dict = create_distance_matrix(cities_df)
        sa = SimulatedAnnealing(distance_dict, coordinates_dict, Cooling.Slow, T_lower=60, decr_T_every_n_iters=decr_T_every_n_iters)
        sa.run()
        all_states = sa.all_states
        last_dist = all_states[-1]['distance']
        name = f'cool={cooling.value}_dist={last_dist}_decr={decr_T_every_n_iters}'
        print(last_dist, sa.T)
        animate_me(cities, all_states, coordinates_dict, dirname, name)
        # except:
        #     print('didnt work:', cooling.value)