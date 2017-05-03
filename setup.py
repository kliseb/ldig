from setuptools import setup,find_packages

setup(name='ldig',
      version='1.1',
      description='Language Detection with Infinity Gram,',
      author='Nakatani Shuyo, Clement Demonchy',
      author_email='demonchy.clement@gmail.com',
      license='MIT',
      include_package_data=True,
      packages=find_packages(),
      install_requires=[
          "typing"  
      ],
      test_suite='nose.collector',
      zip_safe=False)
