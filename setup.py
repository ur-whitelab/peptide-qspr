from distutils.core import setup
from setuptools import setup,find_packages

with open('README.md') as f:
    desc = ''.join(f.readlines())


# get the dependencies and installs
with open('requirements.txt') as f:
    all_reqs = f.read().split('\n')

install_reqs = [x.strip() for x in all_reqs if 'git+' not in x]

setup(name='peptide-qspr',
      version='0.0.1',
      description='QSPR Modelling with pymc3 and Bayesian Motif modelling',
      long_description=desc,
      author='Rainier Barrett',
      packages=find_packages(),
      install_requires=install_reqs,
      package_data = {'peptideqspr.evaluation':['resources/*', 'resources/gibbs/*','resources/gauss/*','resources/human/*','resources/human/gibbs/*','resources/human/gauss/*']},
      include_package_data=True,
      entry_points=
      {
          'console_scripts':
          [
              'evaluate-peptide=peptideqspr.evaluation.evaluate_peptide:main',
              'train-gaussmix=peptideqspr.gaussmix.APD_multi_gaussmix:main'
          ]
      }
     )
