from setuptools import setup, find_packages

setup(
    name="stock_prediction",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "tensorflow>=2.6.0",
        "keras>=2.6.0",
        "matplotlib>=3.4.2",
        "seaborn>=0.11.1",
        "fastapi>=0.68.0",
        "uvicorn>=0.15.0",
        "plotly>=5.1.0",
        "dash>=2.0.0",
        "mlflow>=1.20.2",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Stock Market Prediction using Machine Learning",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/stock-prediction-ml",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 