# Natural Language Processing in Risk and Finance

## Course Overview

The **Natural Language Processing in Risk and Finance** course aims to provide participants with practical skills in applying Natural Language Processing (NLP) to quantitative risk management. This hands-on course covers key NLP techniques and their real-world relevance.

## Key Learning Objectives

Participants will learn essential NLP techniques and their evolution in recent years, including:
- **Text vectorization** (e.g., tf-idf, word embeddings)
- **Text clustering** via topic modeling

The course emphasizes a systematic approach to processing and assessing textual data, utilizing popular Python libraries like **sentence_transformers** and **bertopic**.

## Practical Application

A specific use case explored in the course involves assessing insurance claims, where participants will analyze and cluster textual claim descriptions to identify common topics and trends. This helps insurers better understand risk factors from unstructured text data.

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
8. Open Jupyter Lab:
    ```
    jupyter lab
    ```
9. Open the notebook `nlp.ipynb` and start learning.
