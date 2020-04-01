def profile_simulation(n_sim: int=10) -> None:
    """Elementary speedup check due to more efficient Markov simulation.
    """
    import time
    import numpy as np
    import networkx as nx
    from .markov_epidemic import MarkovSIS
    from tqdm import tqdm

    infection_rate = 1.0
    recovery_rate = 1.0
    T = 1.0

    fastest_times = np.empty(n_sim)
    fast_times = np.empty(n_sim)
    slow_times = np.empty(n_sim)

    with tqdm(total=n_sim) as pbar:
        N = 200
        G = nx.random_regular_graph(15, N)

        for i in range(n_sim):
            SIS = MarkovSIS(infection_rate, recovery_rate, G, simulation_method = 'slow')
            start_time = time.time()
            SIS.simulate(T)
            slow_times[i] = time.time() - start_time

            SIS = MarkovSIS(infection_rate, recovery_rate, G, simulation_method = 'fast')
            start_time = time.time()
            SIS.simulate(T)
            fast_times[i] = time.time() - start_time

            SIS = MarkovSIS(infection_rate, recovery_rate, G, simulation_method = 'fastest')
            start_time = time.time()
            SIS.simulate(T)
            fastest_times[i] = time.time() - start_time

            pbar.update(1)

    slow_times_mean = np.mean(slow_times) * 1e3
    slow_times_std = np.std(slow_times) * 1e3

    fast_times_mean = np.mean(fast_times) * 1e3
    fast_times_std = np.std(fast_times) * 1e3

    fastest_times_mean = np.mean(fastest_times) * 1e3
    fastest_times_std = np.std(fastest_times) * 1e3

    print('Slow simulation: {:.0f}ms +/- {:.0f}ms'.format(slow_times_mean, slow_times_std))
    print('Fast simulation: {:.0f}ms +/. {:.0f}ms'.format(fast_times_mean, fast_times_std))
    print('Fastest simulation: {:.0f}ms +/. {:.0f}ms'.format(fastest_times_mean, fastest_times_std))
