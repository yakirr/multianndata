import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="multianndata", # Replace with your own username
    version="0.0.4",
    author="Yakir Reshef, Laurie Rumker",
    author_email="yreshef@broadinstitute.org",
    description="Multi-sample version of AnnData",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yakirr/multianndata",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'anndata',
        'numpy',
        ],
)
