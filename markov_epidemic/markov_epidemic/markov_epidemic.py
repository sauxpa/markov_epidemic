import abc
import numpy as np
import scipy
import networkx as nx
from functools import lru_cache
from copy import copy

class MarkovEpidemic(abc.ABC):
    """Generic class to simulate Markov epidemics.
    """
    def __init__(self,
                 G: nx.Graph,
                 simulation_method: str='fastest',
                ) -> None:
        self._G = G
        self._simulation_method = simulation_method

        self.X = np.empty(0)
        self.transition_times = np.empty(0)
        self.nodes_infected_at_least_once = set()

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

    def random_seed_nodes(self, k):
        """Select k nodes uniformly at random to be infectedself.
        Returns a vector x0 such that x0[i] = 1 if i is in the seed group,
        0 otherwise.
        """
        x0 = np.zeros(self.N)
        seed_patients = np.random.choice(self.G.nodes, size=k, replace=False)
        x0[seed_patients] = 1
        return x0

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

    @abc.abstractmethod
    def transition_rates(self, Xt: np.ndarray) -> np.ndarray:
        """Markov transition rates, depends on the type of epidemic
        model (SIS, SIR, other...)
        """
        pass

    def custom_init(self) -> None:
        """If needed in for subclasses (e.g in SIR).
        """
        pass

    def custom_postprocessing(self) -> None:
        """If needed in for subclasses (e.g in SIR).
        """
        pass

    def simulate(self, T:float, x0: np.ndarray=np.empty(0)) -> None:
        """Simulate diffusion of Markov SIS epidemic up to time T.
        """
        self.nodes_infected_at_least_once = set()

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

        self.is_epidemic_live = True

        while t < T:
            # If the epidemic died sooner than T
            if np.sum(Xt) == 0:
                break

            # At each step, rates[i] contains the infection/curing rate of node i
            rates = self.transition_rates(Xt)

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

            if Xt[node] == 0:
                self.nodes_infected_at_least_once.add(node)

            # Flip state of transitioned node
            Xnew = copy(Xt)
            Xnew[node] = 1 - Xnew[node]
            Xt = Xnew

            X.append(Xt)

        self.X = np.array(X)
        self.transition_times = np.array(transition_times)
        self.T = self.X.shape[0]
