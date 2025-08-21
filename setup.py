import setuptools

def get_requirements():
    with open('requirements.txt', 'r') as f:
        return f.read().splitlines()

setuptools.setup(
    name='grin',
    version='0.1.0',
    author='Murray S. Bennett',
    author_email='bennett.1755@osu.edu',
    description='Neural Networks for rapid GRT model and parameter estimation.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/murraysbennett/grin',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    # Automatically finds packages in the 'src' directory
    packages=setuptools.find_packages(where='src'),
    # Tells setuptools to look for packages inside the 'src' directory
    package_dir={'': 'src'},
    python_requires='>=3.9',
    install_requires=get_requirements(), # A list of dependencies from a requirements file (need to create this)
)
