from setuptools import setup, find_packages
setup(
    name='aide_predict',
    version='1.0',
    packages=['aide_predict'],
    extras_require={
        'test': ['pytest', 'pytest-cov']
    }
)
