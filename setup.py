from setuptools import setup, find_packages
from pathlib import Path
import os, sys, contextlib


NAME = "spacy_ray"
ROOT = Path(__file__).parent
ABOUT_LOC = ROOT / NAME / "about.py"
REQS_LOC = ROOT / "requirements.txt"


@contextlib.contextmanager
def chdir(new_dir):
    old_dir = os.getcwd()
    try:
        os.chdir(new_dir)
        sys.path.insert(0, new_dir)
        yield
    finally:
        del sys.path[0]
        os.chdir(old_dir)


def read_about():
    about = {}
    with ABOUT_LOC.open() as file_:
        exec(file_.read(), about)
    return about


def setup_package():
    about = read_about()
    setup(
        name="spacy-ray",
        packages=find_packages(),
        version=about["__version__"],
        description=about["__summary__"],
        author=about["__author__"],
        author_email=about["__email__"],
        url=about["__uri__"],
        license=about["__license__"],
        long_description_content_type="text/markdown",
        classifiers=[
            "Environment :: Console",
            "Intended Audience :: Developers",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: POSIX :: Linux",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
            "Programming Language :: Cython",
            "Programming Language :: Python :: 3.6",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Topic :: Scientific/Engineering",
        ],
    )


if __name__ == "__main__":
    setup_package()
