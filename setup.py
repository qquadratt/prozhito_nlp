from setuptools import setup, find_packages
import os

this_directory = os.path.abspath(os.path.dirname(__file__))
readme_path = os.path.join(this_directory, "README.md")

long_description = ""
if os.path.exists(readme_path):
    with open(readme_path, encoding="utf-8") as f:
        long_description = f.read()

setup(
    name="prozhito_nlp",
    version="1.0.0", 
    author="Veronika Mikhaylova", 
    description="NLP-инструменты для анализа дневниковых текстов из корпуса Прожито", 
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qquadratt/prozhito_nlp",
    packages=find_packages(),
    package_data={
        "prozhito_nlp": ["data/*.txt"],
    },
    include_package_data=True,
    install_requires=[
        "pandas",
        "numpy", 
        "natasha", 
        "scikit-learn",
        "plotly", 
        "tqdm", 
        "scipy",
        "statsmodels",

    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", 
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
