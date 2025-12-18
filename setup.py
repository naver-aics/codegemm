from setuptools import find_packages, setup

install_requires = [
    'torch',
    'numpy',
    'transformers==4.45.0',
    'datasets==2.19.0',
    'evaluate',
    'lm_eval==0.4.4',
    'peft==0.10.0',
    'aqlm'
]


setup(
    name='codegemm',
    version='0.0',
    author='Gunho Park',
    author_email='gunho.park3@navercorp.com',
    description='Repository for codegemm',
    packages=find_packages(),
    install_requires=install_requires,
)
