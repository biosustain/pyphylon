from setuptools import find_packages, setup

setup(
    name="pyphylon",
    version="0.0.1",
    author="Siddharth M Chauhan",
    author_email="smchauhan@ucsd.edu",
    description="Python package for constructing, analyzing, & visualizing co-occuring gene / allele sets (phylons) within a pangenome.",
    long_description="",
    long_description_content_type="text/markdown",
    url="https://github.com/SBRG/pyphylon/",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "biopython",
        "tqdm",
        "matplotlib",
        "seaborn",
        "plotly",
        "umap-learn",
        "hdbscan",
        "prince",
    ],
    dependency_links=[
        "git+https://github.com/georg-un/kneebow.git#egg=kneebow",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extras_require={"cd-hit": ["cd-hit"]},
    python_requires=">=3.11",
)
