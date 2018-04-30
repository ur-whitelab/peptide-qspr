from distutils.core import setup

with open('README.md') as f:
    long_description = ''.join(f.readlines())

# get the dependencies and installs
with open('requirements.txt') as f:
    all_reqs = f.read().split('\n')

install_requires = [x.strip() for x in all_reqs if 'git+' not in x]

setup(name='pymc3_qspr',
      version='0.01',
      description='QSPR Modelling with pymc3 and Bayesian Motif modelling',
      long_description=long_description,
      author='Rainier Barrett',
      packages=['evaluation', 'qspr_plots', ],
      install_requires=install_requires,
      entry_points=
      {
          'evaluate_peptide':
          [
              'evaluate_peptide = evaluation.evaluate_peptide.__main__:main'
          ]
      }
     )
