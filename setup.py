import setuptools

def get_requirements():
    with open('requirements.txt', 'r') as f:
        return f.read().splitlines()

setuptools.setup(
    name='grin', # The name of your package
    version='0.1.0', # The current version of your package
    author='Murray S. Bennett', # Your name or the author's name
    author_email='bennett.1755@osu.edu', # Your email
    description='Neural Networks for rapid GRT model and parameter estimation.', # A short description
    long_description=open('README.md').read(), # A long description from your README file
    long_description_content_type='text/markdown', # The type of the long description
    url='https://github.com/murraysbennett/grin', # A URL to your project repository
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    # Automatically finds packages in the 'src' directory
    packages=setuptools.find_packages(where='src'),
    # Tells setuptools to look for packages inside the 'src' directory
    package_dir={'': 'src'},
    python_requires='>=3.9', # Your project's minimum Python version
    install_requires=get_requirements(), # A list of dependencies from a requirements file (you'll need to create this)
)
