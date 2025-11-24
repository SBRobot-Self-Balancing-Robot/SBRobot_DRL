from setuptools import setup, find_packages

setup(
    name="self-balancing-robot-rl",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A reinforcement learning project for training a robot to maintain balance using MuJoCo.",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "gymnasium",
        "mujoco",
        "torch",
        "matplotlib",
        "pyyaml"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)