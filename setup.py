#!/usr/bin/env python
from setuptools import setup

setup(
    name='emulator-1ts',
    version='0.0.1',
    python_requires='>3.10',
    packages=['emulator_1ts'],
    install_requires=[
        'torch',
        'numpy',
        'xarray',
        'pandas',
        'dask[diagnostics]',
        'bottleneck',
        'xbatcher',
    ],
)
