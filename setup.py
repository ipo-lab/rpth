from setuptools import setup, find_packages

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

setup(
    name="rpth",
    version="0.0.1",
    author="Andrew Butler",
    author_email="",
    description="Risk parity batch solver in Pytorch",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/ipo-lab/",
    packages=find_packages(),
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3.9",
    ],
)
