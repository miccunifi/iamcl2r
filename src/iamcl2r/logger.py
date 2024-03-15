import logging
import traceback

def setup_logger(logfile, console_log: bool = True, file_log=True, log_level=logging.DEBUG):
    
    string_format = '%(asctime)s %(name)17s[%(lineno)3s] %(levelname)7s %(message)s'
    # '%(asctime)s - %(module)s - %(levelname)s - %(message)s'
    # Console logging
    if console_log:
        ch = logging.StreamHandler()
        formatter = logging.Formatter(string_format)
        ch.setLevel(log_level)
        ch.setFormatter(formatter)

    # file logging
    # file_string_format = '%(asctime)s %(name)17s[%(lineno)3s] %(levelname)7s %(message)s'
    fh = logging.FileHandler(logfile)
    fh.setLevel(log_level)
    formatter = logging.Formatter(string_format, datefmt='%d.%m.%Y %H:%M:%S')
    fh.setFormatter(formatter)

    root = logging.getLogger()
    if console_log:
        root.addHandler(ch)
    root.addHandler(fh)
    root.setLevel(log_level)
    return root


# Catch all uncaught exception in log files
def log_uncaught_exceptions(ex_cls, ex, tb):
    logging.critical(''.join(traceback.format_tb(tb)))
    logging.critical('{0}: {1}'.format(ex_cls, ex))