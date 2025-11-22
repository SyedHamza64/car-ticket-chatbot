"""Setup script for the project."""
from setuptools import setup, find_packages

setup(
    name="lcda-rag-assistant",
    version="0.1.0",
    description="RAG-based support assistant for LaCuraDellAuto",
    author="LaCuraDellAuto",
    packages=find_packages(),
    install_requires=[
        "python-dotenv>=1.0.0",
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "beautifulsoup4>=4.12.0",
        "requests>=2.31.0",
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
    ],
    python_requires=">=3.10",
)

