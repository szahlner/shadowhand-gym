from setuptools import setup, find_packages


with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name="shadowhand_gym",
    description="OpenAI Gym Shadow Dexterous Hand robot environment based on PyBullet.",
    author="szahlner",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/szahlner/shadowhand-gym",
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    version="1.0.0",
    install_requires=["gym", "numpy", "pybullet"],
    classifiers=[
        "License :: MIT License",
        "Programming Language :: Python :: 3.7",
    ],
)
