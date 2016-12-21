#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from setuptools import find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('CHANGELOG.md') as history_file:
    history = history_file.read()

requirements = [
    'scipy',
    'numpy',
    'cpc.geofiles',
    'cpc.stats'
]

setup(
    name="cpc.mpp",
    version='v0.2',
    description="Model post processing package",
    long_description=readme + '\n\n' + history,
    author="Mike Charles",
    author_email='mike.charles@noaa.gov',
    url="https://github.com/mikecharles/cpc.mpp",
    packages=find_packages(),
    namespace_packages=['cpc'],
    include_package_data=True,
    install_requires=requirements,
    license="CC",
    zip_safe=False,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        'License :: CC0 1.0 Universal (CC0 1.0) Public Domain Dedication',
    ],
)
