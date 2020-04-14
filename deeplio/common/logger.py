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
    def __init__(self, name="deeploi", filename="deeplio.txt", level=logging.INFO):
        super(PyLogger, self).__init__()
        self.__logger = logging.getLogger(name=name)
        self.__logger.propagate = False
        self.__logger.setLevel(logging.DEBUG)

        # create formatter
        self.formatter = logging.Formatter('%(message)s')

        self.ch = logging.StreamHandler()
        self.ch.setLevel(level)
        self.ch.setFormatter(self.formatter)

        self.fh = logging.FileHandler(filename)
        self.fh.setLevel(level)
        self.fh.setFormatter(self.formatter)

        # add ch to logger
        self.__logger.addHandler(self.ch)
        self.__logger.addHandler(self.fh)

    def error(self, msg, *args , **kwargs):
        msg = "ERROR: {}".format(msg)
        self.__logger.error(msg, *args)

    def info(self, msg, *args, **kwargs):
        msg = "INFO: {}".format(msg)
        self.__logger.info(msg, *args)

    def debug(self, msg, *args, **kwargs):
        self.__logger.debug(msg, *args)

    def warning(self, msg, *args, **kwargs):
        msg = "DEBUG: {}".format(msg)
        self.__logger.warning(msg, *args)

    def print(self, msg, *args, **kwargs):
        self.__logger.info(msg, *args)


global_logger = None