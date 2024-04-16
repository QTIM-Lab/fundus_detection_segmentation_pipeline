import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fundus-detection-segmentation-pipeline",
    version="0.0.1",
    author="Scott Kinder",
    author_email="scott.kinder@cuanschutz.edu",
    description="An end-to-end pipeline to segment optic cup and disc from fundus photos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your_username/example-package",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        'torch>=1.9.0',
        'torchvision>=0.10.0',
        'torchaudio>=0.10.0',
        'numpy>=1.21.0',
        'pandas>=1.3.3',
        'matplotlib>=3.4.3',
        'Pillow>=8.3.2',
        'opencv-python>=4.5.3.56',
        'transformers>=4.11.3',
        'ultralytics>=0.18.2',
        'albumentations>=1.1.0'
    ],
)