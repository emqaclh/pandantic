from setuptools import setup

setup(
    name="pandantic",
    version="0.7",
    description="validation and amendment on pandas DataFrames",
    url="https://github.com/emqaclh/pandantic",
    author="emqaclh",
    author_email="villanueva.alexis17@gmail.com",
    license="MIT",
    packages=["pandantic"],
    zip_safe=False,
    install_requires=[
        "numpy==1.23.1",
        "pandas==1.4.3",
        "python-dateutil==2.8.2",
        "pytz==2022.1",
        "six==1.16.0",
    ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest"],
)
