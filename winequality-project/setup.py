from setuptools import find_packages, setup

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Machine learning project for prediction of Wine quality',
    author='Alina Chasova',
    install_requires=[
        'matplotlib==3.5.1',
        'pandas==1.4.1',
        'seaborn==0.11.2',
        'scikit-learn==0.24.1',
    ],
    license='MIT',
)
