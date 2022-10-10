from glob import glob
from os import path
from os.path import basename, splitext

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="infopy",
    version="0.0.1",
    description="Information theory related estimators",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jurrutiag/infopy",
    author="Juan Urrutia",
    author_email="juan.urrutia.gandolfo@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    py_modules=[splitext(basename(path))[0] for path in glob("src/*.py")],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        # complete classifier list: http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Operating System :: Unix",
        "Operating System :: POSIX",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    install_requires=["numpy", "scipy", "scikit-learn"],
    extras_require={
        "testing": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "tox>=3.24",
        ]
    },
)
