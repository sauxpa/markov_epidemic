import numpy as np
import networkx as nx
from .markov_epidemic import MarkovEpidemic

class MarkovSIR(MarkovEpidemic):
    """Class to simulate Markov epidemics
    in the Susceptible-Infected-Removed model.
    """
    def __init__(self,
                 infection_rate: float,
                 recovery_rate: float,
                 G: nx.Graph,
                 simulation_method: str='fastest',
                ) -> None:
        self._infection_rate = infection_rate
        self._recovery_rate = recovery_rate

        super().__init__(G, simulation_method=simulation_method)

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

    def transition_rates(self, Xt: np.ndarray) -> np.ndarray:
        infected_neighbors = self.A.dot(Xt)

        return np.array(
            [
                0 if node in self.nodes_infected_at_least_once and Xt[node] == 0\
                else (self.infection_rate * infected_neighbors[node] if Xt[node] == 0 else self.recovery_rate)\
                for node in self.G.nodes
            ]
        )
        # rates = np.empty(self.N)
        # for node in self.G.nodes:
        #     if node in self.nodes_infected_at_least_once and Xt[node] == 0:
        #         rates[node] = 0
        #     elif Xt[node]:
        #         rates[node] = self.infection_rate * infected_neighbors[node]
        #     else:
        #         rates[node] = self.recovery_rate
        #
        # return rates
