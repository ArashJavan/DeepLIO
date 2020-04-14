import logging

class Logger(object):
    def __init__(self):
        pass

    def error(self, msg, *args , **kwargs):
        pass

    def info(self, msg, *args, **kwargs):
        pass

    def debug(self, msg, *args, **kwargs):
        pass


class DummyLogger(Logger):
    def __init__(self):
        super(DummyLogger, self).__init__()


class PyLogger(object):
    def __init__(self, name="deeploi", filename="deeplio.txt"):
        super(PyLogger, self).__init__()
        self.logger = logging.getLogger(name=name)
        self.logger.setLevel(logging.DEBUG)

        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)

        self.fh = logging.FileHandler(filename)
        self.ch.setLevel(logging.DEBUG)

        # create formatter
        self.formatter = logging.Formatter('%(levelname)s: %(message)s')

        # add formatter to ch
        self.ch.setFormatter(self.formatter)
        self.fh.setFormatter(self.formatter)

        # add ch to logger
        self.logger.addHandler(self.ch)
        self.logger.addHandler(self.fh)

    def error(self, msg, *args , **kwargs):
        self.logger.error(msg, *args)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args)

    def warning(self, msg, *args, **kwargs):
        self.logger.warning(msg, *args)

    def print(self, msg, *args, **kwargs):
        # create formatter
        formatter = logging.Formatter('%(message)s')

        # add formatter to ch
        self.ch.setFormatter(formatter)
        self.fh.setFormatter(formatter)

        self.logger.info(msg, *args)

        self.ch.setFormatter(self.formatter)
        self.fh.setFormatter(self.formatter)

global_logger = None