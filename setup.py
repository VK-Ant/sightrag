from setuptools import setup, find_packages

# Try C++ extension build
ext_modules = []
try:
    from pybind11.setup_helpers import Pybind11Extension
    ext_modules = [
        Pybind11Extension(
            "sightrag_cpp",
            sources=["cpp/src/image_ops.cpp", "cpp/src/pybind_module.cpp"],
            include_dirs=["cpp/include"],
            libraries=["opencv_core", "opencv_imgcodecs", "opencv_imgproc", "opencv_videoio"],
            extra_compile_args=["-O3"],
        )
    ]
except ImportError:
    pass  # No pybind11 = pure Python, still works

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="sightrag",
    version="0.2.0",
    author="Venkatkumar Rajan",
    description="SightRAG — Image and Video RAG. See. Search. Retrieve.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/VK-Ant/sightrag",
    packages=find_packages(),
    ext_modules=ext_modules,
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
        "fast": ["pybind11>=2.11.0"],
        "onnx": ["onnxruntime>=1.16.0"],
        "tensorrt": ["tensorrt", "polygraphy"],
        "openvino": ["openvino>=2024.0.0"],
        "chroma": ["chromadb>=0.4.0"],
        "api": ["fastapi>=0.100.0", "uvicorn>=0.23.0", "python-multipart>=0.0.6"],
        "all": ["pybind11>=2.11.0", "onnxruntime>=1.16.0", "chromadb>=0.4.0",
                "fastapi>=0.100.0", "uvicorn>=0.23.0", "python-multipart>=0.0.6"],
    },
    entry_points={
        "console_scripts": [
            "sightrag-server=sightrag.api:serve",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="computer-vision rag image-search video-search retrieval clip yolo",
)
