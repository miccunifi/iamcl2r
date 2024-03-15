from setuptools import setup, find_packages

# Read the contents of your README file
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the contents of requirements.txt
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

    
PACKAGES = find_packages(where="src")
entry_points = {
    'console_scripts': [
        'iamcl2r=iamcl2r.main:main',
    ],
}

setup(
    name='iamcl2r',
    version='0.1.0',  # Update with your package's version
    author='Niccolò Biondi, Federico Pernici, Simone Ricci, Alberto Del Bimbo',  # Update with your name
    author_email='niccolo.biondi@unifi.it',  # Update with your email
    description='Official PyTorch Implementation of "Stationary Representations: Optimally Approximating Compatibility and Implications for Improved Model Replacements" CVPR24',  # Update with a short description
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/miccunifi/iamcl2r',  # Update with your project's URL
    package_dir={'': 'src'},
    packages=PACKAGES,
    py_modules=["iamcl2r"],
    entry_points=entry_points,
    install_requires=requirements,
    classifiers=[
        # Update these classifiers according to your project's needs
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Update the Python version if necessary
)