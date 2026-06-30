from setuptools import find_packages, setup
from os import path
from io import open

ver_file = path.join('txgnn', 'version.py')
with open(ver_file) as f:
    exec(f.read())

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = f.read().splitlines()

setup(name='TxGNN-PyG',
      version=__version__,
      license='MIT',
      description='TxGNN (PyTorch Geometric implementation)',
      url='https://github.com/mims-harvard/TxGNN',
      author='TxGNN Team',
      author_email='kexinh@stanford.edu',
      packages=find_packages(exclude=['test']),
      zip_safe=False,
      include_package_data=True,
      install_requires=requirements,
      setup_requires=['setuptools>=38.6.0']
      )
