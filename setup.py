import setuptools

with open("README.md", "r", encoding = "utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "ensemblem",
    version = "0.3.2",
    author = "Ivan Vigorito",
    author_email = "ivanvigorit@gmail.com",
    description = "Dynamic Weighted Ensemble - Local Fusion",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url = "https://github.com/IvanVigor/Dynamic-Weighted-Ensemble",
    project_urls = {
        "Bug Tracker": "https://github.com/IvanVigor/Local-Fusion-Dynamic-Weighted-Ensemble/issues",
    },
    classifiers = [
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir = {"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires = ">=3.8"
)