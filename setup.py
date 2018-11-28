from setuptools import setup, find_packages
import os


setup(name='HackModules',
      version='1.0.0',
      description='Predict the J-V curve of a semiconductor',
      author='Wan Dongyang',
      author_email='Dongyang@u.nus.edu',
      classifiers=['Programming Language :: Python :: 3.6'],
      long_description=open('README.rst').read(),
      install_requires=['pandas',
                        'numpy',
                        'sklearn',
                        'matplotlib',
                        'sys'
                        'os'               
                         ],
      license='GPL',\
      packages = find_packages('HackModules'),
#       entry_points = {
#      'console_scripts': [
#       'bayesim=bayesim.__main__:main'],
#       },
      zip_safe=False)
