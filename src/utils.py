import tqdm


class TQDM_CB:
    def __init__(self, iterations, interval):
        self._iterations = iterations
        self._tqdm = tqdm.tqdm(total=iterations)
        self._interval = interval

    def write(self, stuff):
        self._tqdm.update(self._interval)
        stuff = stuff.replace('\t', ' ')
        self._tqdm.write(stuff.rstrip('\n'), end='')
        p = 5523

    def close(self):
        self._tqdm.close()

    def finish(self):
        if self._tqdm.n < self._iterations:
            self._tqdm.update(self._iterations - self._tqdm.n)


# sweep = wandb.controller(sweep_id)
# sweep.run(verbose=True, print_actions=True)

"""
sweep_configuration = {
    'method': 'bayes',
    'name': 'sweep',
    'metric': {'goal': 'minimize', 'name': 'best_score_val'},
    'parameters':
        {
            'learning_rate': {'max': 0.3, 'min': 0.01},
            'depth': {'max': 10, 'min': 6},
            'border_count': {'value': 128}
        }
}
"""
