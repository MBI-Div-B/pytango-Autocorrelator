from setuptools import setup, find_packages

setup(
    name="tangods_autocorrelator",
    version="0.0.1",
    description="This tango device server returns in femtoseconds the pulse duration of the laser",
    author="Daniel Schick",
    author_email="dschick@mbi-berlin.de",
    python_requires=">=3.6",
    entry_points={"console_scripts": ["Autocorrelator = tangods_autocorrelator:main"]},
    license="MIT",
    packages=["tangods_autocorrelator"],
    install_requires=[
        "pytango",
        "lmfit",
        "scipy"
    ],
    url="https://github.com/MBI-Div-b/pytango-Autocorrelator",
    keywords=[
        "tango device",
        "tango",
        "pytango",
    ],
)
