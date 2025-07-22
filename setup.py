from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


baryonification_link = 'https://bitbucket.org/aurel..'

setup(name='baryonification',
      version='0.1',
      description='Correcting N-body outputs to account for baryon effects',
      url=baryonification_link,
      author='Aurel Schneider',
      author_email='schneider.duhem@gmail.com',
      packages=['baryonification'],
      zip_safe=False)
