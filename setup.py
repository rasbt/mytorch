# Sebastian Raschka 2018
# mytorch
# Author: Sebastian Raschka <sebastianraschka.com>
#
# License: MIT

from os.path import realpath, dirname, join
from setuptools import setup, find_packages
import mytorch

VERSION = mytorch.__version__
PROJECT_ROOT = dirname(realpath(__file__))

REQUIREMENTS_FILE = join(PROJECT_ROOT, 'requirements.txt')

with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()

install_reqs.append('setuptools')


setup(name='mytorch',
      version=VERSION,
      description='PyTorch-related utility functions',
      author='Sebastian Raschka',
      author_email='mail@sebastianraschka.com',
      url='https://github.com/rasbt/mytorch',
      packages=find_packages(),
      package_data={'': ['LICENSE.txt',
                         'README.md',
                         'requirements.txt']
                    },
      include_package_data=True,
      install_requires=install_reqs,
      extras_require={'testing': ['nose'],
                      'docs': ['mkdocs']},
      license='MIT',
      platforms='any',
      long_description="""

A library of PyTorch-related utility tools.
Currently only intended for personal use.
""")
