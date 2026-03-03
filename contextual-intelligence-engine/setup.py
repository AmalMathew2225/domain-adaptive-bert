"""
Setup configuration for the Contextual Intelligence Engine package.
"""

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt", encoding="utf-8") as f:
    install_requires = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="contextual-intelligence-engine",
    version="0.1.0",
    author="Your Name",
    author_email="you@example.com",
    description="A domain-adaptive BERT fine-tuning engine for contextual NLP tasks.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/contextual-intelligence-engine",
    packages=find_packages(exclude=["tests*", "examples*"]),
    python_requires=">=3.9",
    install_requires=install_requires,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords=["bert", "nlp", "fine-tuning", "domain-adaptation", "transformers"],
)
