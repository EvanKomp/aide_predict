from setuptools import setup, find_packages

setup(
    name='aide_prediction',
    version='1.0',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    install_requires=[
        'numpy',
        'scikit-learn',
        'tensorflow',
        'pandas',
    ],
    entry_points={
        'console_scripts': [
            'protein-prediction=src.tools.predict:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)