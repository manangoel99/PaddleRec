import paddle
from visualdl import LogWriter

from .base_logger import BaseLogger


class VDLLogger(BaseLogger):
    def __init__(self, save_dir):
        super().__init__(save_dir)
        self.vdl_writer = LogWriter(logdir=save_dir)

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
        for k, v in updated_metrics.items():
            self.vdl_writer.add_scalar(tag=k, value=v, step=step)
    
    def log_model(self, file_path, epoch_id, prefix='rec'):
        pass

    def close(self):
        self.vdl_writer.close() 