import os

import paddle

from .base_logger import BaseLogger


class WandbLogger(BaseLogger):
    def __init__(self, 
        project=None, 
        name=None, 
        id=None, 
        entity=None, 
        save_dir=None, 
        config=None,
        log_model=False,
        **kwargs):
        try:
            import wandb
            self.wandb = wandb
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Please install wandb using `pip install wandb`"
                )

        self.project = project
        self.name = name
        self.id = id
        self.save_dir = save_dir
        self.config = config
        self.kwargs = kwargs
        self.entity = entity
        self._run = None
        self._wandb_init = dict(
            project=self.project,
            name=self.name,
            id=self.id,
            entity=self.entity,
            dir=self.save_dir,
            resume="allow"
        )
        self.model_logging = log_model
        self._wandb_init.update(**kwargs)

        _ = self.run

        if self.config:
            self.run.config.update(self.config)

    @property
    def run(self):
        if self._run is None:
            if self.wandb.run is not None:
                print(
                    "There is a wandb run already in progress "
                    "and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()`"
                    "before instantiating `WandbLogger`."
                )
                self._run = self.wandb.run
            else:
                self._run = self.wandb.init(**self._wandb_init)
        return self._run

    def log_metrics(self, metrics, prefix=None, step=None):
        updated_metrics = dict()
        for k, v in metrics.items():
            if prefix:
                key = prefix + "/" + k
            else:
                key = k
            if isinstance(v, paddle.Tensor):
                updated_metrics[key] = v.item()
            else:
                updated_metrics[key] = v
        self.run.log(updated_metrics, step=step)

    def log_model(self, file_path, epoch_id, prefix='rec'):
        if self.model_logging == False:
            return
        model_path = os.path.join(file_path, str(epoch_id))
        model_prefix = os.path.join(model_path, prefix)
        artifact = self.wandb.Artifact('model-{}'.format(self.run.id), type='model')
        artifact.add_file(model_prefix + ".pdparams", name="model.pdparams")
        artifact.add_file(model_prefix + ".pdopt", name="optimizer.pdopt")

        self.run.log_artifact(artifact, aliases=["epoch " + str(epoch_id)])

    def close(self):
        self.run.finish()