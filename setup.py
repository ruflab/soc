from setuptools import setup, find_packages


def load_long_description():
    text = open('README.md', encoding='utf-8').read()
    return text


setup(
    name="soc",
    version='0.1a1',
    python_requires='>=3.6',
    description='Settlers of Catan AI research project',
    url='https://github.com/ruflab/soc',
    download_url='https://github.com/ruflab/soc',
    author='Morgan Giraud',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.8',
    ],
    keywords=['deep learning', 'pytorch', 'AI'],
    long_description=load_long_description(),
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['tests', 'tests/*'])
)
