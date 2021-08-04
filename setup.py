import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="cerbo", # Replace with your own username
    version="0.3.2",
    author="StartOnAI",
    author_email="startonaicom@gmail.com",
    description="Perform Efficient ML/DL Modelling",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/StartOnAI/Cerbo",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)