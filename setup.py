from setuptools import setup, find_packages

setup(
    name="uplc_method_recommendation",
    version="0.1.0",
    description="UPLC method recommendation system",
    author="hzy, xxy",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9"
)