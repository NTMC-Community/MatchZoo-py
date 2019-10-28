import io
import os

from setuptools import setup, find_packages


here = os.path.abspath(os.path.dirname(__file__))

# Avoids IDE errors, but actual version is read from version.py
__version__ = None
exec(open('matchzoo/version.py').read())

short_description = 'Facilitating the design, comparison and sharing' \
                    'of deep text matching models.'

# Get the long description from the README file
with io.open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

install_requires = [
    'torch >= 1.2.0',
    'pytorch-transformers >= 1.1.0',
    'nltk >= 3.4.3',
    'numpy >= 1.16.4',
    'tqdm >= 4.32.2',
    'dill >= 0.2.9',
    'pandas == 0.24.2',
    'networkx >= 2.3',
    'h5py >= 2.9.0',
    'hyperopt == 0.1.2'
]

extras_requires = {
    'tests': [
        'coverage >= 4.5.3',
        'codecov >= 2.0.15',
        'pytest >= 4.6.3',
        'pytest-cov >= 2.7.1',
        'flake8 >= 3.7.7',
        'flake8_docstrings >= 1.3.0'],
}


setup(
    name="matchzoo-py",
    version=__version__,
    author="MatchZoo-py Authors",
    author_email="fanyixing@ict.ac.cn",
    description=(short_description),
    license="Apache 2.0",
    keywords="text matching models",
    url="https://github.com/NTMC-Community/MatchZoo-py",
    packages=find_packages(),
    include_package_data=True,
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Environment :: Console',
        'Operating System :: POSIX :: Linux',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        "License :: OSI Approved :: Apache Software License",
        'Programming Language :: Python :: 3.6'
    ],
    install_requires=install_requires,
    extras_require=extras_requires
)
