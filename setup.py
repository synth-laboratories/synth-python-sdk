from setuptools import find_packages, setup

setup(
    name="synth-sdk",
    version="0.0.4",
    packages=find_packages(),
    install_requires=[
        "opentelemetry-api",
        "opentelemetry-sdk",
        "pydantic",
        "requests",
        "asyncio",
    ],
    author="Synth AI",
    author_email="josh@usesynth.ai",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    classifiers=[],
)