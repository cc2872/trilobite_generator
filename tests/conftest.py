import os
os.environ.setdefault("TRILO_RECORD", "0")     # the test suite's own measurements are not data: keep them out of web/records and off the wire


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: a full build (minutes); run with -m slow")
