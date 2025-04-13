from setuptools import setup, find_packages

setup(
    name="finger_detector",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "mediapipe>=0.8.9",
    ],
    author="Prerak Pithadiya",
    author_email="your.email@example.com",  # Replace with your email
    description="A Python application that detects and counts fingers in real-time using computer vision",
    keywords="computer vision, finger detection, mediapipe, opencv",
    url="https://github.com/PrerakPithadiya/finger-detector",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.6",
)
