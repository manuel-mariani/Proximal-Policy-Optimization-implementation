import numpy as np
import wandb


class Logger:
    def log(self, commit=False, **kwargs):
        pass

    def finish(self):
        pass

    def append(self, **kwargs):
        pass


class WandbLogger(Logger):
    def __init__(self, run_name, config: dict):
        self.params = dict(project="AAS-RL", reinit=True, config=config, name=run_name)
        self.run = wandb.init(**self.params)
        self.run.log_code()
        self.cache = dict()

    def finish(self):
        self.run.finish()

    def log(self, commit=False, **kwargs):
        for k, v in kwargs.items():
            self.cache[k] = v
        if commit:
            self._reduce_cache()
            self.run.log(self.cache)
            self.cache = dict()

    def append(self, **kwargs):
        for k, v in kwargs.items():
            if k in self.cache and isinstance(self.cache[k], list):
                self.cache[k].append(v)
            elif k not in self.cache:
                self.cache[k] = [v]
            else:
                self.cache[k] = [self.cache[k], v]

    def _reduce_cache(self):
        cache = dict()
        for k, v in self.cache.items():
            if not isinstance(v, list):
                cache[k] = v
            else:
                cache[k] = np.mean(v)
        self.cache = cache
