import setuptools


with open('README.md', 'r') as f:
    long_description = f.read()
setuptools.setup(
    name='Fall 2023 DSAN-5400 Final Project',
    version='0.0.1',
    author='Siyuan Zhang',
    author_email='sz687@georgetown.edu',
    description='this is final project for DSAN-5400, Fall 2023, Georgetown University. It is about using GPT2 to generate comments for Amazon products.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    extras_requres={"dev": ["pytest", "flake8", "autopep8"]},
)