import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import find_peaks
from scipy.signal import savgol_filter


def calculate_xcorr(transition_times: np.ndarray,
                    signal: np.ndarray,
                    interp_kind: str = 'previous',
                    sampling_step: float = 0.01,
                    ):
    """Calculate autocorrelogram.
    """
    # Interpolate to create even sampling
    num_infected = interp1d(transition_times,
                            signal,
                            fill_value='extrapolate',
                            kind=interp_kind,
                            )
    tt = np.arange(0, np.max(transition_times), sampling_step)
    signal_resampled = num_infected(tt)

    # Calculate autocorrelogram
    xcorr = np.correlate(signal_resampled, signal_resampled, 'full')
    xcorr_tt = np.concatenate([-tt[::-1], tt[1:]])

    # Normalize between -1 and 1
    xcorr /= np.max(xcorr)
    return xcorr, xcorr_tt


def period_estimator(transition_times: np.ndarray,
                     number_of_infected: np.ndarray,
                     interp_kind: str = 'previous',
                     sampling_step: float = 0.01,
                     savgol_window: int = 501,
                     savgol_polyorder: int = 3,
                     width_around_peak: int = 0.2,
                     distance_between_peaks: int = 100,
                     ) -> dict:
    """Estimate period pulsations in the number of infected,
    by the means of the smoothed autocorrelogram.
    """
    # Calculte the autocorrelogram of the increments rather than the signal
    # to better capture periodicity (the signal itself is higly non-stationary)
    xcorr, xcorr_tt = calculate_xcorr(transition_times[:-1],
                                      np.diff(number_of_infected),
                                      interp_kind=interp_kind,
                                      sampling_step=sampling_step,
                                      )

    # Smooth autocorrelogram to detect real peaks
    xcorr_smooth = savgol_filter(xcorr, savgol_window, savgol_polyorder)

    # Tails of the autocorrelogram are too noisy, keep only a fraction
    # of width around the central peak
    mid_xcorr = len(xcorr) // 2
    width_xcorr = int(len(xcorr) * width_around_peak)
    peaks, _ = find_peaks(
        xcorr_smooth[mid_xcorr-width_xcorr:mid_xcorr+width_xcorr],
        distance=distance_between_peaks
        )

    # To avoid boundary effects due to the arbitrary width cut above,
    # remove boundary peaks
    peaks = peaks[1:-1] + mid_xcorr-width_xcorr
    peaks = xcorr_tt[peaks]

    # Estimate the period as the mean distance between peaks
    period = np.mean(np.diff(peaks))

    return {
        'period': period,
        'xcorr': xcorr,
        'xcorr_smooth': xcorr_smooth,
        'xcorr_tt': xcorr_tt,
        'mid_xcorr': mid_xcorr,
        'width_xcorr': width_xcorr,
        'peaks': peaks,
        }


def profile_simulation(n_sim: int = 10) -> None:
    """Elementary speedup check due to more efficient Markov simulation.
    """
    import time
    import numpy as np
    import networkx as nx
    from .sis_epidemic import MarkovSIS
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
            SIS = MarkovSIS(infection_rate,
                            recovery_rate,
                            G,
                            simulation_method='slow'
                            )
            start_time = time.time()
            SIS.simulate(T)
            slow_times[i] = time.time() - start_time

            SIS = MarkovSIS(infection_rate,
                            recovery_rate,
                            G,
                            simulation_method='fast'
                            )
            start_time = time.time()
            SIS.simulate(T)
            fast_times[i] = time.time() - start_time

            SIS = MarkovSIS(infection_rate,
                            recovery_rate,
                            G,
                            simulation_method='fastest'
                            )
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

    print(
        'Slow simulation: {:.0f}ms +/- {:.0f}ms'.format(slow_times_mean,
                                                        slow_times_std)
        )
    print(
        'Fast simulation: {:.0f}ms +/. {:.0f}ms'.format(fast_times_mean,
                                                        fast_times_std)
        )
    print(
        'Fastest simulation: {:.0f}ms +/. {:.0f}ms'.format(fastest_times_mean,
                                                           fastest_times_std)
        )
