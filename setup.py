"""Packaging metadata for the ``escreen`` distribution.

Installing this package provides the eScreen model, the eScreen-local analysis
tools, the StripedHyena2 backbone and the Streamlit front end, i.e. everything
described in Supplementary Methods Sections 2 and 8.

Dependency groups, by the part of the package that needs them:

======================  ==================================================
requirement             needed for
======================  ==================================================
``torch``               the model and the SH2core blocks (Section 2.1)
``numpy``, ``scipy``    tensors, PCHIP target transform (Section 2.3.2),
                        PWM score distributions
``einops``              tensor reshaping in the backbone
``scikit-learn``        the evaluation metrics of Section 7.1
``pandas``              tabular data handling in the tool modules
``pyfaidx``             reading hg38 sequence (Section 2.3.1)
``pyliftover``          hg19-to-hg38 coordinate conversion (Section 2.4.3)
``pyBigWig``            epigenomic signal tracks (Sections 2.4.2, 7.4)
``transformers``        loading the Geneformer checkpoint (Section 2.3.5)
``streamlit``           the web front end in ``escreen/serve.py``
``captum``              Integrated Gradients attribution (Section 3.2)
======================  ==================================================

Notes
-----
* The core prediction path needs only ``torch``, ``numpy``, ``einops``,
  ``transformers`` and the motif file.  ``streamlit`` and ``captum`` are only
  reached from ``serve.py`` and ``model.py`` respectively, so a headless
  training run does not need them at runtime.
* Versions are mostly unpinned.  The environment actually used for the reported
  results is Python 3.10, PyTorch 2.6.0 and CUDA 11.8 (Section 2.3.7); the full
  environment file is ``requirement/environment.yaml``.
* ``include_package_data=True`` but no ``package_data`` / ``MANIFEST.in`` entry is
  declared here, so data files needed at import time are not packaged
  automatically -- the motif database and the checkpoints must be supplied
  separately.
"""

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
