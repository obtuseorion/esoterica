from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="esoterica",
    version="0.1.0",
    author="Urbas Ekka",
    author_email="urbasekka@gmail.com",
    description="additons to scikit-learn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/obtuseorion/esoterica",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "scikit-learn>=0.24.0",
    ],
)