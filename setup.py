from setuptools import setup, find_packages

setup(
    name="keras-gpt-copilot",
    version="0.1.4",
    description="Integrate an LLM copilot within your Keras model development workflow",
    long_description="Keras GPT Copilot is the first Python package designed to integrate an LLM copilot within the model development workflow, offering iterative feedback options for enhancing the performance of your Keras deep learning models. Utilizing the power of OpenAI's GPT models, Keras GPT Copilot can use any of the compatible models (GPT4 is recommended). However, the prompt-only mode allows for compatibility with other large language models.",
    author="Fabi Prezja",
    author_email="faprezja@fairn.fi",
    url="https://github.com/fabprezja/keras-gpt-copilot",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "openai",
        "pyperclip",
        "tensorflow",
    ],
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    include_package_data=True,
)
