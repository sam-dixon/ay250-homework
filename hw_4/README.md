# HW 4

## Running the program

Start the ipyparallel cluster:

`ipycluster start -n 4 &`

Just make the plot (data included as `experiment_times.pkl`):

`python pi.py`

Run everything all tests on `n_darts` up to 10^7 to make your own `experiment_times.pkl`:

`python pi.py --doall`

Options:

```
-n: log10(max number of darts)
-r: number of repitions for calculating error
-s: filename for temporary pickle file
-o: output figure filename
```

## Results

![Results](https://github.com/sam-dixon/ay250-homework/raw/master/hw_4/parallel.png)

We can see that for simulations using fewer points, the overhead for managing the parallel cluster or multiple processes is larger than the time to perform each calculation. Once the total time to perform the calculation grows larger than the overhead from multiprocessing, we can see a speed up around a factor of 2 (since my laptop has 2 cores). Also note that because this is such a CPU intensive task, running multiple processes on the same core doesn't give us any additional boost.
