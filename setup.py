from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sightrag",
    version="0.1.3",
    author="Venkatkumar Rajan (VK-Ant)",
    author_email="venkatkumarr.vk99@gmail.com",
    description="SightRAG: Image and Video RAG. See. Search. Retrieve.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VK-Ant/sightrag",
    project_urls={
        "Bug Reports": "https://github.com/VK-Ant/sightrag/issues",
        "Source": "https://github.com/VK-Ant/sightrag",
    },
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.30.0",
        "ultralytics>=8.0.0",
        "opencv-python>=4.8.0",
        "Pillow>=9.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "chroma": ["chromadb>=0.4.0"],
        "api": ["fastapi>=0.100.0", "uvicorn>=0.23.0", "python-multipart>=0.0.6"],
        "all": ["chromadb>=0.4.0", "fastapi>=0.100.0", "uvicorn>=0.23.0", "python-multipart>=0.0.6"],
    },
    entry_points={
        "console_scripts": [
            "sightrag-server=sightrag.api:serve",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    keywords="computer-vision rag image-search video-search retrieval clip yolo",
)
