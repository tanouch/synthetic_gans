from setuptools import setup, find_packages

setup(
  name = 'synthetic_gans',
  packages = find_packages(),
  include_package_data = True,
  version = '1',
  license='MIT',
  description = 'Warhol',
  authors = 'Ugo Tanielian, Gérard Biau, Maxime Sangnier',
  url = 'https://github.com/tanouch/synthetic_gans',
  keywords = ['GANs'],
  install_requires=[
      'numpy', 
      'matplotlib',
      'scipy',
      'torch', 
      'POT'],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
