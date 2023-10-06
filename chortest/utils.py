import json
import errno
import os
import signal
import functools

from chortest.common import TransitionLabel
from chortest.lts import LTSTransitionLabel


class TimeoutError(Exception):
    pass


def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator


def fail_unless(cond: bool, msg: str):
    if not cond:
        raise Exception(msg)


class LangJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, LTSTransitionLabel) or isinstance(o, TransitionLabel):
            return str(o)
        return super().default(o)
