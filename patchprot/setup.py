from setuptools import setup, find_packages
import io
import re
import os
import sys

sys.path.append(os.getcwd())

def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding="utf-8") as fd:
        return re.sub(text_type(r":[a-z]+:`~?(.*?)`"), text_type(r"``\1``"), fd.read())


setup(
    name="PROT",
    version="0.0.1",
    packages=find_packages(exclude=("tests",)),
    entry_points={
        "console_scripts": [
            "PROT=PROT.cli:cli"
        ]
    },
)