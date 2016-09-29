import sys
import multiprocessing
import threading
import ipyparallel
import numpy as np
from time import time
import pickle
import argparse


def timer(fn):
    """Timing decorator"""
    def timed(*args, **kwargs):
        start = time()
        result = fn(*args, **kwargs)
        end = time()
        print(fn.__name__, args, end-start)
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
def pi_serial(n_tot):
    """Throw n_tot darts serially"""
    n_in = sum([dart() for i in range(n_tot)])
    pi_approx = 4 * n_in / n_tot
    return pi_approx


@timer
def pi_proc(n_tot, n_procs=4):
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


@timer
def pi_cluster(n_tot):
    darts = c[:].map(dart, range(n_tot))
    n_in = sum(darts)
    pi_approx = 4 * n_in / n_tot
    return pi_approx


def do_experiment(fn, n_tot_range, reps=3, **kw):
    """Repeat a function 5 times and report the mean and std dev of the exec times"""
    times = []
    for i in range(reps):
        times.append([fn(n_tot, **kw)[1] for n_tot in n_tot_range])
    times = np.array(times)
    return np.mean(times, axis=0), np.std(times, axis=0)


def do_all(nmax=7, save='./experiment_times.pkl', reps=3):
    """Do all of the experiments for the plot and save the data to file"""
    if nmax == None:
        nmax = 7
    if reps == None:
        reps = 3
    n_tot_range = [int(i) for i in np.logspace(1, nmax, (nmax*2)-1)]
    info, d = {}, {}
    info['n_tot'] = n_tot_range
    for k in ['serial', 'multiprocess (2 procs)', 'multiprocess (4 procs)', 'cluster (4 cores)']: info[k] = {}
    d['mean_time'], d['std_time'] = do_experiment(pi_serial, n_tot_range, reps=reps)
    info['serial'] = d
    d = {}
    d['mean_time'], d['std_time'] = do_experiment(pi_proc, n_tot_range, reps=reps, n_procs=2)
    info['multiprocess (2 procs)'] = d
    d = {}
    d['mean_time'], d['std_time'] = do_experiment(pi_proc, n_tot_range, reps=reps, n_procs=4)
    info['multiprocess (4 procs)'] = d
    d = {}
    d['mean_time'], d['std_time'] = do_experiment(pi_cluster, n_tot_range, reps=reps)
    info['cluster (4 cores)'] = d
    if save is not None:
        # Write to file
        pickle.dump(info, open(save, 'wb'))
    return info


def make_plot(fname='./experiment_times.pkl', save=None):
    """Make a plot with the data from the saved experiment"""
    try:
        info = pickle.load(open(fname, 'rb'))
    except:
        print('No experiment saved under', fname)
        sys.exit()

    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set_style('white')
    sns.set_context('poster')
    palette = sns.color_palette('Set2')

    fig, ax1 = plt.subplots(figsize=(12, 9))
    ax2 = ax1.twinx()
    n_tot_range = info['n_tot']
    lines = []
    for k in ['serial', 'multiprocess (2 procs)', 'multiprocess (4 procs)', 'cluster (4 cores)']:
        v = info[k]
        color = palette.pop(0)
        lines += ax1.plot(n_tot_range, v['mean_time'], '-', label=k, color=color, linewidth=2)
        ax1.fill_between(n_tot_range, v['mean_time']-v['std_time'], v['mean_time']+v['std_time'], alpha=0.5, color=color)
        ax2.plot(n_tot_range, n_tot_range/v['mean_time'], '--', color=color, linewidth=2)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    labels = [l.get_label() for l in lines]
    leg = ax2.legend(lines, labels, loc='lower right')
    ax1.set_xlabel('Number of darts')
    ax1.set_ylabel('Execution time [s] (solid)')
    ax2.set_ylabel('Simulation rate [darts/s] (dotted)')
    ax1.set_title('MacBook Air w/ 1.3 GHz Core i5 (2 cores)')

    if save is not None:
        plt.savefig(save, bbox_inches='tight')
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test parallel computing methods while calculating pi')
    parser.add_argument('--doall', action='store_true', help='Run all tests and save a pickle file of the results')
    parser.add_argument('-n', dest='nmax', type=int, default=7, nargs='?', help='log10(max number of darts to throw). Default: 7')
    parser.add_argument('-r', dest='reps', type=int, default=3, nargs='?', help='Number of repitiions of each experiment (for error). Default: 3')
    parser.add_argument('-s', dest='savetmp', type=str, default='experiment_times.pkl', nargs='?', help='Pickle file name for timing data. Default: experiment_times.pkl')
    parser.add_argument('-o', dest='output', type=str, help='Save figure path. Default: parallel.pdf')
    args = parser.parse_args()
    if args.doall:
        # Initialize IPyParallel client
        c = ipyparallel.Client()
        # Do experiments
        do_all(nmax=args.nmax, save=args.savetmp, reps=args.reps)
    make_plot(fname=args.savetmp, save=args.output)
