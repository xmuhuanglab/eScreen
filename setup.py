from setuptools import setup, find_packages

setup(
    name='escreen',
    version='0.0.0',
    description='Using Deep Learning to screen functional cis-regulatory elements in silico',
    url='https://github.com/kps333/eScreen-beta/tree/main',
    author='Liquan Lin, Shijie Luo',
    author_email='21620241153548@stu.xmu.edu.cn, sluo112211@163.com',
    license='GNU',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[
        'pyBigWig',
        'pyfaidx',
        'pybedtools'
    ],
    include_package_data=True,
)