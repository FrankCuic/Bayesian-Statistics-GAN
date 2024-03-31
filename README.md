# Bayesian-Statistics-GAN

Comparative Analysis of Bayesian Deep Generative Models and Frequentist Approaches in Stock Market Analysis

## Project Overview
This project explores the application of Bayesian deep generative models and contrasts them with traditional frequentist approaches such as LSTM in the context of stock market analysis. The aim is to evaluate the predictive performance, uncertainty quantification, and adaptability of these models in forecasting stock prices and identifying market trends.

## Team Members
- **Can Cui:** Focused on the development and implementation of Bayesian deep generative models.
- **XinRui Wang:** Specialized in data preprocessing and implementation of frequentist models including LSTM.

## Table of Contents
- [Data Description](#data-description)
- [Methodology](#methodology)
  - [Bayesian Deep Generative Models](#bayesian-deep-generative-models)
  - [Frequentist Regression/Classification Models](#frequentist-regressionclassification-models)
  - [Model Evaluation and Comparison](#model-evaluation-and-comparison)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Discussion](#discussion)
- [Contributions](#contributions)
- [References](#references)

## Data Description
The analysis utilizes two primary datasets:
1. **Historical Stock Prices:** Includes daily closing prices, volume, and other financial indicators for selected stocks.
2. **Market Indicators:** Comprises macroeconomic indicators and market indices that influence stock prices.

## Methodology

### Bayesian Deep Generative Models
- Implement Variational Autoencoders (VAEs) as the core Bayesian model.
- Integrate prior knowledge through carefully chosen prior distributions.
- Conduct posterior analysis to assess model uncertainty and data fit.

### Frequentist Regression/Classification Models
- Employ models like Linear Regression, Logistic Regression, SVM, and LSTM for predictive analysis.
- Train models on historical stock data to forecast prices and classify market trends.

### Model Evaluation and Comparison
- Compare Bayesian and frequentist models based on metrics such as accuracy, precision, recall, and F1 score.
- Utilize Bayesian diagnostics to evaluate model convergence and effectiveness.

## Installation

To set up the project environment, follow these steps:

```bash
# Clone the repository
git clone https://github.com/your-username/your-repo-name.git

# Navigate to the project directory
cd your-repo-name

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
