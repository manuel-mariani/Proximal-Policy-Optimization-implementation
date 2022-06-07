import wandb


class Logger:
    def log(self, commit=False, **kwargs):
        pass

    def finish(self):
        pass

class WandbLogger(Logger):
    def __init__(self, run_name, config: dict):
        self.params = dict(
            project="AAS-RL",
            reinit=True,
            config=config,
            name=run_name
        )
        self.run = wandb.init(**self.params)
        self.cache = dict()

    def finish(self):
        self.run.finish()

    def log(self, commit=False, **kwargs):
        for k, v in kwargs.items():
            self.cache[k] = v
        if commit:
            self.run.log(self.cache)
            self.cache = dict()
