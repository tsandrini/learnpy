"""
 __                                       ____
/\ \                                     /\  _`\
\ \ \         __      __     _ __    ___ \ \ \L\ \ __  __
 \ \ \  __  /'__`\  /'__`\  /\`'__\/' _ `\\ \ ,__//\ \/\ \
  \ \ \L\ \/\  __/ /\ \L\.\_\ \ \/ /\ \/\ \\ \ \/ \ \ \_\ \
   \ \____/\ \____\\ \__/.\_\\ \_\ \ \_\ \_\\ \_\  \/`____ \
    \/___/  \/____/ \/__/\/_/ \/_/  \/_/\/_/ \/_/   `/___/> \
                                                       /\___/
                                                       \/__/
Created by Tomáš Sandrini
"""


import setuptools


try:
    import learnpy
except (ImportError, SyntaxError):
    print("error: LearnPy requires Python 3.5 or greater.")
    quit(1)



VERSION = learnpy.__version__
DOWNLOAD = "https://github.com/tsandrini/learnpy/archive/%s.tar.gz" % VERSION


setuptools.setup(
    name="LearnPy",
    version=VERSION,
    author="Tomáš Sandrini",
    author_email="tomas.sandrini@seznam.cz",
    description="Implementation of popular ML algs in python",
    long_description="Implementation of popular ML algs in python",
    license="MIT",
    url="https://github.com/tsandrini/learnpy",
    download_url=DOWNLOAD,
    classifiers=[
        "Environment :: X11 Applications",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    packages=["learnpy"],
    python_requires=">=3.5",
    test_suite="tests",
    include_package_data=True
)
