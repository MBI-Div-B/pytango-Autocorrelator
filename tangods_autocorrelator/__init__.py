from .Autocorrelator import Autocorrelator


def main():
    import sys
    import tango.server

    args = ["Autocorrelator"] + sys.argv[1:]
    tango.server.run((Autocorrelator,), args=args)
