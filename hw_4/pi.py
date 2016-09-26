import numpy as np
from time import time
import multiprocessing
import threading


def timer(fn):
    """Timing decorator"""
    def timed(*args, **kwargs):
        start = time()
        result = fn(*args, **kwargs)
        end = time()
        print(fn.__name__, *args, end-start)
        return result, end-start
    return timed


def dart(*args, **kwargs):
    """Throw a dart; if in the circle, return 1"""
    x, y = np.random.rand(), np.random.rand()
    if np.sqrt((x-0.5)**2 + (y-0.5)**2) <= 0.5:
        return 1
    else:
        return 0


@timer
def pi_serial(n_tot, n_workers=1):
    """Throw n_tot darts serially"""
    n_in = sum([dart() for i in range(n_tot)])
    pi_approx = 4 * n_in / n_tot
    return pi_approx


class thread_dart(threading.Thread):
    """
    Extension of the thread base class. Keeps track of dart locations in
    a single thread.
    """
    def __init__(self, n):
        threading.Thread.__init__(self)
        self.n = n
        self.n_in = 0

    def run(self):
        self.n_in = sum([dart() for i in range(self.n)])


@timer
def pi_threading(n_tot, n_threads):
    """
    Throw n_tot darts and distribute the work among n_threads
    """
    # Set up the number of darts thrown by each thread
    if n_tot % n_threads != 0:
        darts_per_thread = n_tot // n_threads + 1
        remainder = n_tot - ((n_threads - 1) * darts_per_thread)
    else:
        darts_per_thread = n_tot // n_threads
        remainder = 0

    threads = []
    for i in range(n_threads):
        if i == n_threads-1 and remainder != 0:
            t = thread_dart(remainder)
        else:
            t = thread_dart(darts_per_thread)
        threads.append(t)
        t.start()
        t.join()

    n_tot = sum([t.n for t in threads])
    n_in = sum([t.n_in for t in threads])
    pi_approx = 4 * n_in / n_tot
    return pi_approx


@timer
def pi_proc(n_tot, n_procs):
    """
    Throw n_tot darts and distribute the work among n_procs
    """
    pool = multiprocessing.Pool(processes=n_procs)
    darts = pool.map(dart, range(n_tot))
    n_in = sum(darts)
    pi_approx = 4 * n_in / n_tot
    pool.terminate()
    del pool
    return pi_approx


def do_experiment(fn, n_tot_range, n_workers):
    times = []
    for i in range(1):
        times.append([fn(n_tot, n_workers)[1] for n_tot in n_tot_range])
    times = np.array(times)
    return np.mean(times, axis=0), np.std(times, axis=0)


if __name__ == '__main__':
    n_tot_range = [int(i) for i in np.logspace(2, 7, 20)]

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('white')
    sns.set_color_codes()
    print('Serial')
    mean, std = do_experiment(pi_serial, n_tot_range, 1)
    plt.plot(n_tot_range, mean, 'b-')
    plt.fill_between(n_tot_range, mean-std, mean+std, color='b', alpha=0.5)
    print('4 threads')
    mean, std = do_experiment(pi_threading, n_tot_range, 4)
    plt.plot(n_tot_range, mean, 'g-')
    plt.fill_between(n_tot_range, mean-std, mean+std, color='g', alpha=0.5)
    print('2 processes')
    mean, std = do_experiment(pi_proc, n_tot_range, 2)
    plt.plot(n_tot_range, mean, 'r-')
    plt.fill_between(n_tot_range, mean-std, mean+std, color='r', alpha=0.5)
    print('4 processes')
    mean, std = do_experiment(pi_proc, n_tot_range, 4)
    plt.plot(n_tot_range, mean, 'y-')
    plt.fill_between(n_tot_range, mean-std, mean+std, color='yellow', alpha=0.5)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()


