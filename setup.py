"""
The setup.py file is an essential part of packaging and distributing python projects.
It is used by setuptools to define the configuration of our projects, such as metadata, dependencies and more

"""
# find_packages: it will scan all the folders inside the repo and check if there is __init__.py file
# setup: responsible to provide package info
# -e. help in building the packages, it sits in requirements.txt, when user run pip install -r requiremnts.txt, it will refer setup.py

from setuptools import find_packages, setup

from typing import List

def get_requirements()->List[str]:
    """
    Return the list of requirements
    """
    requirement_list:List[str]=[]
    try:
        with open('requirements.txt', 'r') as file:
            lines = file.readlines()
            for line in lines:
                requirement = line.strip()
                # ignore the empty lines and -e.
                if requirement and requirement!='-e .':
                    requirement_list.append(requirement)
    except FileNotFoundError:
        print("File not exist.")
    return requirement_list

setup(
    name='NetworkSecurity',
    version='0.0.1',
    author='Hussain Shabbir Madarwala',
    author_email='hussainmadar4@gmail.com',
    packages=find_packages(),
    install_requires= get_requirements()
)