# Machine Learning in Risk and Finance

## Course Overview

The **Machine Learning in Risk and Finance** course aims to provide participants with practical skills in applying Machine Learning (ML) to quantitative risk management. This hands-on course covers essential concepts, tools, and real-world applications.

## Key Learning Objectives

Participants will learn essential ML concepts and algorithms, including:
- **ML pipelines**
- **Tree-based ML algorithms** (e.g., Regression Trees, Random Forests, Gradient Boosting)
- **Model training and inference**

The course emphasizes a systematic approach to modeling risks and predicting uncertain outcomes, utilizing popular Python libraries like **scikit-learn** and **pandas**.

## Practical Application

A specific use case explored in the course involves assessing the resale value of leased cars at the end of their lease terms, a common risk for car leasing companies. Participants will build and evaluate predictive models using historical data to estimate vehicle resale values.

## Technical Setup

1. Install [Python 3.12](https://www.python.org/downloads/release/python-3120/) (or higher) on your local machine.
2. Install [Git](https://git-scm.com/downloads) on your local machine.
3. Clone the repository on your local machine using Git:
    ```
    git clone https://github.com/julienOlivier3/risk-analytics.git
    ```
4. Create a virtual environment and install required dependencies into it:
    ```
    python -m venv .venv
    ```
5. Activate the newly created virtual environment:
    - On Windows:
      ```
      .venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```
      source .venv/bin/activate
      ```
6. Install all required dependencies from the root directory:
    ```
    pip install -r requirements.txt
    ```
7. Open Jupyter Lab:
    ```
    jupyter lab
    ```
8. Open the notebook `ml.ipynb` and start learning.
