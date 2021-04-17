import logging
import sys
import time


def get_logger(log_file=None, file_level=logging.INFO, stdout_level=logging.DEBUG, logger_name=__name__):
    logging.root.setLevel(0)
    formatter = logging.Formatter('%(asctime)s %(levelname)5s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    _logger = logging.getLogger(logger_name)

    if log_file is not None and len(log_file) > 0:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level=file_level)
        file_handler.setFormatter(formatter)
        _logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setLevel(level=stdout_level)
    stream_handler.setFormatter(formatter)
    _logger.addHandler(stream_handler)
    return _logger


def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def process_bar(current, total, prefix='', auto_rm=True):
    bar = '=' * int(current / total * 50)
    bar = f' {prefix} |{bar.ljust(50)}| ({current}/{total}) {current / total:.1%} | '
    print(bar, end='\r', flush=True)
    if auto_rm and current == total:
        print(end=('\r' + ' ' * len(bar) + '\r'), flush=True)
