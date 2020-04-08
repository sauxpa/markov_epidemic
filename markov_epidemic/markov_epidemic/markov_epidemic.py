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
                 simulation_method: str = 'fastest',
                 ) -> None:
        self._G = G
        self._simulation_method = simulation_method

        self.X = np.empty(0)
        self.transition_times = np.empty(0)
        self.nodes_infected_at_least_once = set()

    @property
    def susceptible(self) -> int:
        """Susceptible is state 0.
        """
        return 0

    @property
    def infected(self) -> int:
        """Infected is state 1.
        """
        return 1

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
        x0[seed_patients] = self.infected
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
    def spectral_radius(self) -> float:
        return np.max(np.abs(self.spectrum))

    @property
    @lru_cache(maxsize=None)
    def spectral_gap(self) -> float:
        return self.spectrum[0] - self.spectrum[1]

    @property
    @lru_cache(maxsize=None)
    def cheeger_lower_bound(self) -> float:
        """Lower bound for the isoperimetric constant
        of the graph G given by its adjacency spectral gap.
        """
        return self.spectral_gap / 2

    @property
    @lru_cache(maxsize=None)
    def cheeger_upper_bound(self) -> float:
        """Upper bound for the isoperimetric constant
        of the graph G given by its adjacency spectral gap.
        """
        # Maximum degree
        dmax = np.max(self.A.dot(np.ones(self.N)))
        return np.sqrt(2 * dmax * self.spectral_gap)

    @property
    def cheeger_halfway_approx(self) -> float:
        """Approximate the isoperimetric constant of G
        by the average of its spectral upper and lower bounds.
        """
        return 0.5 * (self.cheeger_lower_bound + self.cheeger_upper_bound)

    def flush_graph(self) -> None:
        """Clear LRU cache of graph related properties.
        """
        type(self).nodes_list.fget.cache_clear()
        type(self).A.fget.cache_clear()
        type(self).spectrum.fget.cache_clear()
        type(self).spectral_radius.fget.cache_clear()
        type(self).spectral_gap.fget.cache_clear()
        type(self).cheeger_lower_bound.fget.cache_clear()
        type(self).cheeger_upper_bound.fget.cache_clear()

    @property
    def number_of_susceptible(self) -> int:
        """Returns the number of susceptible individuals at
        each transition time of the simulated epidemic.
        """
        return np.sum(self.X == self.susceptible, axis=1)

    @property
    def number_of_infected(self) -> int:
        """Returns the number of infected individuals at
        each transition time of the simulated epidemic.
        """
        return np.sum(self.X == self.infected, axis=1)

    def number_infected_neighbors(self, Xt: np.ndarray) -> np.ndarray:
        """Returns the vector of number of infected neighbors given a
        state vector Xt.
        """
        return self.A.dot(Xt == self.infected)

    def deterministic_baseline(self,
                               T: float,
                               initial_infected: int,
                               k: int,
                               n_t_eval: int = 100,
                               ) -> tuple:
        """Solves the deterministic baseline ODEs if they are provided in
        the child class.
        Corresponds to the mean-field approximation of the epidemic under
        the assumption of a k-regular graph.
        """
        self.k_deterministic = k
        solver = scipy.integrate.solve_ivp(
            self.deterministic_baseline_ODEs,
            (0.0, T),
            self.deterministic_baseline_init(initial_infected),
            t_eval=np.linspace(0.0, T, n_t_eval),
        )
        assert solver.success, 'Integration of deterministic ODEs failed.'
        return solver.t, solver.y

    def deterministic_baseline_ODEs(self,
                                    t: float,
                                    y: np.ndarray
                                    ) -> np.ndarray:
        raise NotImplementedError

    def deterministic_baseline_init(self, initial_infected: int) -> np.ndarray:
        raise NotImplementedError

    @abc.abstractmethod
    def transition_rates(self, Xt: np.ndarray) -> np.ndarray:
        """Markov transition rates, depends on the type of epidemic
        model (SIS, SIR, other...)
        """
        pass

    @abc.abstractmethod
    def next_state(self, state: int) -> int:
        """To be implemented in a child class.
        Advance node to its new state. Depending on the model, the number
        and type of states may differ (e.g SIS only allows S->I and I->S
        whereas SIR allows S->I and I->R).
        """
        pass

    @abc.abstractmethod
    def is_epidemic_over(self, Xt: np.ndarray) -> bool:
        """Returns True if all nodes are in a terminal state.
        This depends on the model, e.g in SIS and SIR it is enough
        to have no infected nodes, but in the SEIR model the epidemic
        might continue even if no node is infected, as long as some nodes
        are exposed.
        """
        pass

    def simulate(self, T: float, x0: np.ndarray = np.empty(0)) -> None:
        """Simulate diffusion of Markov epidemic up to time T.
        """
        t = 0.0

        # By default, start with one infected node drawn uniformly at random.
        if len(x0) == 0:
            node = np.random.choice(self.G.nodes)
            Xt = np.zeros(self.N, dtype='int')
            Xt[node] = self.infected
        else:
            Xt = x0.astype('int')

        # List of state vectors of unknown size
        # (each random transition before T adds a row of size N)
        X = [Xt]

        # List of transition times
        transition_times = [0.0]

        while t < T:
            # If the epidemic died sooner than T
            if self.is_epidemic_over(Xt):
                break

            # At each step, rates[i] contains
            # the infection/curing rate of node i
            rates = self.transition_rates(Xt)

            if self.simulation_method == 'fastest':
                # At each step, holding_times[i] contains
                # the holding time of node i.
                # Unsurprisingly, passing the array rates as the scale argument
                # instead of repeating it for every single rate is faster, most
                # likely due to neat optimization in numpy.
                # However, quite surprisingly, this also beats the "fast"
                # method that only requires a single exponential simulation.
                # Why? My money is on the quite slow random.choice below (but
                # that's not that easy to profile due to the nature of the
                # epidemic simulation -- just know that both the fast and
                # fastest method are... well... fast enough.)
                holding_times = np.random.exponential(scale=1/rates)
                # The smallest holding time is the actual transition time
                i = np.argmin(holding_times)
                dt = holding_times[i]
                node = self.nodes_list[i]
            elif self.simulation_method == 'fast':
                # Instead of simulating N independant exponential
                # distributions, simulate a single one with parameter equal to
                # the sum of the individual parameters.
                total_rate = np.sum(rates)
                node = np.random.choice(self.G.nodes, p=rates/total_rate)
                dt = np.random.exponential(scale=1/total_rate)
            else:
                # At each step, holding_times[i] contains
                # the holding time of node i
                holding_times = [
                    np.random.exponential(scale=1/rate) for rate in rates
                    ]
                # The smallest holding time is the actual transition time
                i = np.argmin(holding_times)
                dt = holding_times[i]
                node = self.nodes_list[i]

            # Move forward
            t += dt
            transition_times.append(t)

            # Change state of transitioned node
            Xnew = copy(Xt)
            Xnew[node] = self.next_state(Xt[node])
            Xt = Xnew

            X.append(Xt)

        self.X = np.array(X)
        self.transition_times = np.array(transition_times)
        self.T = self.X.shape[0]
