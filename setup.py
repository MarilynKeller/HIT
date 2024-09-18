from setuptools import find_packages, setup

with open("README_release.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='hit',
    version='1.1.0',
    packages=find_packages(),
    url='https://github.com/MarilynKeller/HIT',
    author='Marilyn Keller',
    author_email='marilyn.keller@tuebingen.mpg.de',
    description='HIT model.',
    long_description=long_description,
    python_requires='>=3.8.0',
    install_requires=[
    ],
)