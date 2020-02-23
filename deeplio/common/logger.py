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
    def __init__(self, name="deeploi"):
        super(PyLogger, self).__init__()
        self.logger = logging.getLogger(name=name)
        self.logger.setLevel(logging.DEBUG)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # create formatter
        formatter = logging.Formatter('%(levelname)s: %(message)s')

        # add formatter to ch
        ch.setFormatter(formatter)

        # add ch to logger
        self.logger.addHandler(ch)

    def error(self, msg, *args , **kwargs):
        self.logger.error(msg, *args)

    def info(self, msg, *args, **kwargs):
        self.logger.info(msg, *args)

    def debug(self, msg, *args, **kwargs):
        self.logger.debug(msg, *args)
