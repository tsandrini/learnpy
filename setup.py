"""
$$$$$$$\              $$\     $$\                           $$\      $$\ $$\
$$  __$$\             $$ |    $$ |                          $$$\    $$$ |$$ |
$$ |  $$ |$$\   $$\ $$$$$$\   $$$$$$$\   $$$$$$\  $$$$$$$\  $$$$\  $$$$ |$$ |
$$$$$$$  |$$ |  $$ |\_$$  _|  $$  __$$\ $$  __$$\ $$  __$$\ $$\$$\$$ $$ |$$ |
$$  ____/ $$ |  $$ |  $$ |    $$ |  $$ |$$ /  $$ |$$ |  $$ |$$ \$$$  $$ |$$ |
$$ |      $$ |  $$ |  $$ |$$\ $$ |  $$ |$$ |  $$ |$$ |  $$ |$$ |\$  /$$ |$$ |
$$ |      \$$$$$$$ |  \$$$$  |$$ |  $$ |\$$$$$$  |$$ |  $$ |$$ | \_/ $$ |$$$$$$$$\
\__|       \____$$ |   \____/ \__|  \__| \______/ \__|  \__|\__|     \__|\________|
          $$\   $$ |
          \$$$$$$  |
           \______/
Created by Tomáš Sandrini
"""


import setuptools


try:
    import python_ml
except (ImportError, SyntaxError):
    print("error: PythonML requires Python 3.5 or greater.")
    quit(1)



VERSION = python_ml.__version__
DOWNLOAD = "https://github.com/tsandrini/python-ml/archive/%s.tar.gz" % VERSION


setuptools.setup(
    name="PythonML",
    version=VERSION,
    author="Tomáš Sandrini",
    author_email="tomas.sandrini@seznam.cz",
    description="Implementation of popular ML algs in python",
    long_description="Implementation of popular ML algs in python",
    license="MIT",
    url="https://github.com/tsandrini/python-ml",
    download_url=DOWNLOAD,
    classifiers=[
        "Environment :: X11 Applications",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
    ],
    packages=["python_ml"],
    python_requires=">=3.5",
    test_suite="tests",
    include_package_data=True
)
