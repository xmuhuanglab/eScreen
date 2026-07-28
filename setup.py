from setuptools import setup, find_packages

setup(
    name='escreen',
    version='0.1.0',
    description='Using Deep Learning to screen functional cis-regulatory elements in silico',
    url='https://github.com/xmuhuanglab/eScreen',
    author='Liquan Lin, Shijie Luo',
    author_email='21620241153548@stu.xmu.edu.cn, sluo112211@163.com',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'torch>=2.0',
        'numpy',
        'einops',
        'tqdm',
        'scikit-learn',
        'scipy',
        'pandas',
        'pyBigWig',
        'pyfaidx',
        'pyliftover',
        'transformers',
        'streamlit>=1.28',
        'captum>=0.6.0',
    ],
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)
