from setuptools import setup, find_packages

setup(
    name="box_upload",
    version="0.1.0",
    description="utility to upload files to box from a remote resource",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "boxsdk",
    ],
    entry_points={
        'console_scripts': ['box_upload=box_upload:run'],
    },
    author="National Renewable Energy Laboratory",
    author_email="Reinicke, Nicholas <Nicholas.Reinicke@nrel.gov>",
)
