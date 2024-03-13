## setup.py file for roicat
from pathlib import Path
import copy
import platform

from distutils.core import setup

## Get the parent directory of this file
dir_parent = Path(__file__).parent

## Get requirements from requirements.txt
def read_requirements():
    with open(str(dir_parent / "requirements.txt"), "r") as req:
        content = req.read()  ## read the file
        requirements = content.split("\n") ## make a list of requirements split by (\n) which is the new line character

    ## Filter out any empty strings from the list
    requirements = [req for req in requirements if req]
    ## Filter out any lines starting with #
    requirements = [req for req in requirements if not req.startswith("#")]
    ## Remove any commas, quotation marks, and spaces from each requirement
    requirements = [req.replace(",", "").replace("\"", "").replace("\'", "").strip() for req in requirements]

    return requirements
deps_all = read_requirements()


## Dependencies: latest versions of requirements
### remove everything starting and after the first =,>,<,! sign
deps_names = [req.split('=')[0].split('>')[0].split('<')[0].split('!')[0] for req in deps_all]
deps_all_dict = dict(zip(deps_names, deps_all))
deps_all_latest = dict(zip(deps_names, deps_names))

## Make different versions of dependencies
### Also pull out the version number from the requirements (specified in deps_all_dict values).
deps_core = {dep: deps_all_dict[dep] for dep in [
    'torch',
    'numpy',
    'matplotlib',
    'opt_einsum',
    'bnpm',
]}

deps_core_latest = {dep: deps_all_latest[dep] for dep in deps_core.keys()}


extras_require = {
    'all': list(deps_all_dict.values()),
    'all_latest': list(deps_all_latest.values()),
    'core': list(deps_core.values()),
    'core_latest': list(deps_core_latest.values()),
}

print(extras_require)

## Get README.md
with open(str(dir_parent / "README.md"), "r") as f:
    readme = f.read()

## Get version number
with open(str(dir_parent / "convolutional_reduced_rank_regression" / "__init__.py"), "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().replace("\"", "").replace("\'", "")
            break


setup(
    name='crrr',
    version=version,
    author='Richard Hakim',
    keywords=['data analysis', 'machine learning', 'neuroscience'],
    license='LICENSE',
    description='Convolutional reduced rank regression.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/RichieHakim/tensor_regression',

    packages=['crrr'],

    install_requires=[],
    extras_require=extras_require,
)