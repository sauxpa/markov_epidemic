import numpy as np
import scipy
import networkx as nx
from functools import lru_cache
from copy import copy

class MarkovSIS():
    """Class to simulate Markov epidemics
    in the Susceptible-Infected-Susceptible model.
    """
    def __init__(self,
                 infection_rate: float,
                 recovery_rate: float,
                 G: nx.Graph,
                 simulation_method: str='fastest',
                ) -> None:
        self._infection_rate = infection_rate
        self._recovery_rate = recovery_rate
        self._G = G
        self._simulation_method = simulation_method

        self.X = np.empty(0)
        self.transition_times = np.empty(0)

    @property
    def infection_rate(self) -> float:
        return self._infection_rate
    @infection_rate.setter
    def infection_rate(self, new_infection_rate: float) -> None:
        self._infection_rate = new_infection_rate

    @property
    def recovery_rate(self) -> float:
        return self._recovery_rate
    @recovery_rate.setter
    def recovery_rate(self, new_recovery_rate: float) -> None:
        self._recovery_rate = new_recovery_rate

    @property
    def effective_diffusion_rate(self) -> float:
        """Ratio of infection to recovery rate.
        """
        return self.infection_rate / self.recovery_rate

    @property
    def G(self) -> nx.Graph:
        return self._G
    @G.setter
    def G(self, new_G: nx.Graph) -> None:
        self.flush_graph()
        self._G = new_G

    @property
    def simulation_method(self) -> str:
        return self._simulation_method
    @simulation_method.setter
    def simulation_method(self, new_simulation_method: str) -> None:
        self._simulation_method = new_simulation_method

    @property
    def N(self) -> int:
        return self.G.number_of_nodes()

    @property
    @lru_cache(maxsize=None)
    def nodes_list(self) -> list:
        return list(self.G.nodes)

    @property
    @lru_cache(maxsize=None)
    def A(self) -> scipy.sparse.csr.csr_matrix:
        """Adjacency matrix of G.
        """
        return nx.adjacency_matrix(self.G)

    @property
    @lru_cache(maxsize=None)
    def spectrum(self) -> np.ndarray:
        """Calculate and cache adjacency spectrum
        (sorted in decreasing order).
        """
        _spectrum = nx.adjacency_spectrum(self.G)
        idx = _spectrum.argsort()[::-1]
        return np.real(_spectrum[idx])

    @property
    @lru_cache(maxsize=None)
    def spectral_radius(self) -> np.ndarray:
        return np.max(np.abs(self.spectrum))

    def flush_graph(self) -> None:
        """Clear LRU cache of graph related properties.
        """
        type(self).nodes_list.fget.cache_clear()
        type(self).A.fget.cache_clear()
        type(self).spectrum.fget.cache_clear()
        type(self).spectral_radius.fget.cache_clear()

    @property
    def number_of_infected(self):
        """Returns the number of infected individuals at
        each transition time of the simulated epidemic.
        """
        return np.sum(self.X, axis=1)

    def simulate(self, T:float, x0: np.ndarray=np.empty(0)) -> None:
        """Simulate diffusion of Markov SIS epidemic up to time T.
        """
        t = 0.0

        # By default, start with one infected node drawn uniformly at random.
        if len(x0) == 0:
            node = np.random.choice(self.G.nodes)
            Xt = np.zeros(self.N)
            Xt[node] = 1
        else:
            Xt = x0

        # List of state vectors of unknown size
        # (each random transition before T adds a row of size N)
        X = [Xt]

        # List of transition times
        transition_times = [0.0]

        while t < T:
            infected_neighbors = self.A.dot(Xt)

            # If the epidemic died sooner than T
            if np.sum(infected_neighbors) == 0:
                break

            # At each step, rates[i] contains the infection/curing rate of node i
            rates = np.array(
                [self.infection_rate * infected_neighbors[i] if Xt[i] == 0 else self.recovery_rate for i in self.G.nodes]
            )

            if self.simulation_method == 'fastest':
                # At each step, holding_times[i] contains the holding time of node i
                # Unsurprisingly, passing the array rates as the scale argument instead of
                # repeating it for every single rate is faster, most likely due to
                # neat optimization in numpy.
                # However, quite surprisingly, this also beats the "fast" method that
                # only requires a single exponential simulation.
                # Why? My money is on the quite slow random.choice below (but that's not that
                # easy to profile due to the nature of the epidemic simulation -- just know that
                # both the fast and fastest method are... well... fast enough.)
                holding_times = np.random.exponential(scale=1/rates)
                # The smallest holding time is the actual transition time
                i = np.argmin(holding_times)
                dt = holding_times[i]
                node = self.nodes_list[i]
            elif self.simulation_method == 'fast':
                # Instead of simulating N independant exponential distributions,
                # simulate a single one with parameter equal to the sum of
                # the individual parameters.
                total_rate = np.sum(rates)
                node = np.random.choice(self.G.nodes, p=rates/total_rate)
                dt = np.random.exponential(scale=1/total_rate)
            else:
                # At each step, holding_times[i] contains the holding time of node i
                holding_times = [np.random.exponential(scale=1/rate) for rate in rates]
                # The smallest holding time is the actual transition time
                i = np.argmin(holding_times)
                dt = holding_times[i]
                node = self.nodes_list[i]

            # Move forward
            t += dt
            transition_times.append(t)
            # Flip state of transitioned node
            Xnew = copy(Xt)
            Xnew[node] = 1 - Xnew[node]
            Xt = Xnew

            X.append(Xt)

        self.X = np.array(X)
        self.transition_times = np.array(transition_times)
