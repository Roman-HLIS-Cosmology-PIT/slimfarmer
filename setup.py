from setuptools import setup, find_packages

setup(
    name='slimfarmer',
    version='1.0.0',
    description='Streamlined single-band/multi band Tractor model photometry',
    packages=find_packages(),
    package_data={'slimfarmer': ['conv_filters/*.conv']},
    python_requires='>=3.9',
    install_requires=[
        'numpy',
        'astropy>=5.0',
        'scipy',
        'sep',
        'tqdm',
    ],
)
