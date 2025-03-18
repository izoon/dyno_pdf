from setuptools import setup, find_packages

setup(
    name="dyno_framework",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.2",
        "PyMuPDF==1.25.3",
        "pdfplumber==0.11.5",
        "pytesseract==0.3.13",
        "opencv-python==4.11.0.86",
        "Pillow>=9.1.0",
        "networkx>=3.4.2",
        "llama-index>=0.10.0",
        "llama-index-readers-web>=0.1.0",
        "pytest>=7.4.0",
        "dynoagent @ git+https://github.com/izoon/dynoagent.git",
    ],
    author="V Rafizadeh",
    author_email="vahi.rafizadeh@gmail.com",
    description="A framework for PDF processing and other document workflows",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/dyno_framework",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 