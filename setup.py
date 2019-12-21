import os
from setuptools import setup

requires = []
with open(os.path.join(os.path.dirname(__file__), 'requirements.txt')) as f:
    requires = f.read().splitlines()

setup(name='xview_lfs',
      version='0.0.1',
      packages=['xview_lfs'],
      install_requires=requires,
      include_package_data=True
      )
