import os
import pkg_resources
from setuptools import setup


def parse_requirements(filename: str):
    requirements_file = os.path.join(os.path.dirname(__file__), filename)
    with open(requirements_file, 'rt') as requirements_stream:
        return [str(requirement) for requirement in pkg_resources.parse_requirements(requirements_stream)]


setup(install_requires=parse_requirements('requirements.txt'))
