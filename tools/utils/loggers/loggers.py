class Loggers(object):
    def __init__(self, loggers):
        super().__init__()
        self.loggers = loggers

    def log_metrics(self, metrics, prefix=None, step=None):
        for logger in self.loggers:
            logger.log_metrics(metrics, prefix=prefix, step=step)
    
    def log_model(self, file_path, epoch_id, prefix='rec'):
        for logger in self.loggers:
            logger.log_model(file_path, epoch_id=epoch_id, prefix=prefix)

    def close(self):
        for logger in self.loggers:
            logger.close()