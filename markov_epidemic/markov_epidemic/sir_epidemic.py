import numpy as np
import networkx as nx
from .markov_epidemic import MarkovEpidemic


class MarkovSIR(MarkovEpidemic):
    """Class to simulate Markov epidemics
    in the Susceptible-Infected-Recovered model.
    """
    def __init__(self,
                 infection_rate: float,
                 recovery_rate: float,
                 G: nx.Graph,
                 simulation_method: str = 'fastest',
                 ) -> None:
        self._infection_rate = infection_rate
        self._recovery_rate = recovery_rate

        super().__init__(G, simulation_method=simulation_method)

        self.X = np.empty(0)
        self.transition_times = np.empty(0)

    @property
    def recovered(self) -> int:
        """Recovered is state 2.
        """
        return 2

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
    def number_of_recovered(self):
        """Returns the number of recovered individuals at
        each transition time of the simulated epidemic.
        """
        return np.sum(self.X == self.recovered, axis=1)

    @property
    def effective_diffusion_rate(self) -> float:
        """Ratio of infection to recovery rate.
        """
        return self.infection_rate / self.recovery_rate

    def next_state(self, x: int) -> int:
        if x == self.susceptible:
            return self.infected
        elif x == self.infected:
            return self.recovered
        elif x == self.recovered:
            raise ValueError(
                'Should not have transition starting from recovered state'
                )
        else:
            raise ValueError('Unknown state')

    def is_epidemic_over(self, Xt: np.ndarray) -> bool:
        return np.sum(Xt == self.infected) == 0

    def transition_rates(self, Xt: np.ndarray) -> np.ndarray:
        num_infected_neighbors = self.number_infected_neighbors(Xt)

        return np.array(
            [
                0 if Xt[node] == self.recovered
                else (self.infection_rate * num_infected_neighbors[node]
                      if Xt[node] == self.susceptible
                      else self.recovery_rate)
                for node in self.G.nodes
            ]
        )

    def deterministic_baseline_ODEs(self,
                                    t: float,
                                    y: np.ndarray
                                    ) -> np.ndarray:
        """ y = (S, I, R)
        """
        return np.array(
            [
                -self.infection_rate * self.k_deterministic
                * y[0] * y[1] / self.N,
                self.infection_rate * self.k_deterministic
                * y[0] * y[1] / self.N - self.recovery_rate * y[1],
                self.recovery_rate * y[1],
            ]
        )

    def deterministic_baseline_init(self, initial_infected: int) -> np.ndarray:
        return np.array([self.N-initial_infected, initial_infected, 0])
