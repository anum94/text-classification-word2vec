import setuptools

REQUIRED_PACKAGES = ['nltk==3.4', "numpy==1.16.2", "pandas==0.24.1", "tensorflow==1.13.1", "tflearn==0.3.2",
                     "fire==0.1.3", "sklearn"]


setuptools.setup(
    name='spooky_author_identification',
    version='0.0.1',
    description='kaggle competition: spooky author identification',
    install_requires=REQUIRED_PACKAGES,
    packages=setuptools.find_packages()
)
