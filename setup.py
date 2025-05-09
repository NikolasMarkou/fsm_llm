from setuptools import setup, find_packages

# Read the contents of README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Import version without importing the entire package
import os
import re

def get_version():
    version_file = os.path.join("src", "llm_fsm", "__version__.py")
    with open(version_file, "r") as f:
        version_match = re.search(r'__version__ = ["\']([^"\']*)["\']', f.read())
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

setup(
    name="llm-fsm",
    version=get_version(),
    author="Nikolas Markou",
    author_email="nikolas.markou@electiconsulting.com",
    description="Finite State Machines for Large Language Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/nikolasmarkou/llm-fsm",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="llm, fsm, finite state machine, language model, conversation",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "loguru>=0.7.3",
        "litellm>=1.68.1",
        "pydantic>=2.11.4",
        "python-dotenv>=1.1.0",
    ],
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "llm-fsm=llm_fsm.main:main_cli",
            "llm-fsm-visualize=llm_fsm.visualizer:main_cli",
            "llm-fsm-validate=llm_fsm.validator:main_cli",
        ],
    },
)