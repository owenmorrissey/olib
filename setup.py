from setuptools import setup, find_packages

setup(
    name="olib",
    version="1.0.0",
    description="Owen's personal utility library",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "pandas",
        "joblib",
        "pathlib",
        "tqdm",
    ],
)

