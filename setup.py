from distutils.util import convert_path

from setuptools import find_packages, setup

version_dict = {}

with open(convert_path('cnp/version.py')) as file:
    exec(file.read(), version_dict)

setup(
    name='cnp',
    version=version_dict['__version__'],
    description='Gaussian neural processes',
    classifiers=['Programming Language :: Python :: 3.6'],
    author='',
    author_email='',
    python_requires='>=3.6',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'numpy',
        'stheno',
        'matplotlib',
        'tensorboard',
        'nvsmi',
        'torch',
        'jupyter',
        'jupyterlab',
        'ipykernel',
        'tqdm',
        'stheno',
        'wbml>=0.3.9'
    ],
    zip_safe=False,
)
