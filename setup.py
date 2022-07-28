from os import path

from setuptools import find_packages, setup

this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="infopy",
    version="0.0.1",
    description="Information theory related estimators",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/jurrutiag/infopy",
    author="Juan Urrutia",
    author_email="juan.urrutia.gandolfo@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.8",
    ],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "scikit-learn"
    ],
    extras_requires=[],
)

