from setuptools import setup


with open('README.md') as file:
    long_description = file.read()


setup(
    name='audacitorch',
    description='',
    version='0.1.0',
    author='Hugo Flores Garcia',
    author_email='hf01049@georgiasouthern.edu',
    url='https://github.com/audacitorch/audacitorch',
    install_requires=[
        'torch==2.0.0',
        'torchaudio==2.0.0'
    ],
    packages=['audacitorch'],
    package_data={'audacitorch': ['assets/*']},
    include_package_data=False,
    long_description=long_description,
    long_description_content_type='text/markdown',
    keywords=[],
    classifiers=['License :: OSI Approved :: MIT License'],
    license='MIT')
