"""
      ___           ___           ___       ___           ___           ___           ___
     /\  \         |\__\         /\__\     /\  \         /\  \         /\  \         /\__\
    /::\  \        |:|  |       /:/  /    /::\  \       /::\  \       /::\  \       /::|  |
   /:/\:\  \       |:|  |      /:/  /    /:/\:\  \     /:/\:\  \     /:/\:\  \     /:|:|  |
  /::\~\:\  \      |:|__|__   /:/  /    /::\~\:\  \   /::\~\:\  \   /::\~\:\  \   /:/|:|  |__
 /:/\:\ \:\__\     /::::\__\ /:/__/    /:/\:\ \:\__\ /:/\:\ \:\__\ /:/\:\ \:\__\ /:/ |:| /\__\
 \/__\:\/:/  /    /:/~~/~    \:\  \    \:\~\:\ \/__/ \/__\:\/:/  / \/_|::\/:/  / \/__|:|/:/  /
      \::/  /    /:/  /       \:\  \    \:\ \:\__\        \::/  /     |:|::/  /      |:/:/  /
       \/__/     \/__/         \:\  \    \:\ \/__/        /:/  /      |:|\/__/       |::/  /
                                \:\__\    \:\__\         /:/  /       |:|  |         /:/  /
                                 \/__/     \/__/         \/__/         \|__|         \/__/
Created by Tomáš Sandrini
"""


import setuptools


try:
    import pylearn
except (ImportError, SyntaxError):
    print("error: PyLearn requires Python 3.5 or greater.")
    quit(1)



VERSION = pylearn.__version__
DOWNLOAD = "https://github.com/tsandrini/pylearn/archive/%s.tar.gz" % VERSION


setuptools.setup(
    name="PyLearn",
    version=VERSION,
    author="Tomáš Sandrini",
    author_email="tomas.sandrini@seznam.cz",
    description="Implementation of popular ML algs in python",
    long_description="Implementation of popular ML algs in python",
    license="MIT",
    url="https://github.com/tsandrini/pylearn",
    download_url=DOWNLOAD,
    classifiers=[
        "Environment :: X11 Applications",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    packages=["pylearn"],
    python_requires=">=3.5",
    test_suite="tests",
    include_package_data=True
)
