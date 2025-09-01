# Advanced Risk Analytics in Python

## Overview
This repository hosts the source code of the **Advanced Risk Analytics in Python** course. The course is designed to equip participants with practical skills and knowledge in the field of Artificial Intelligence (AI). Overall, the course consists of three independent modules focusing on the application of Machine Learning (ML), Natural Language Processing (NLP) and Generative AI (GenAI) within the context of quantitative risk management. As a hands-on seminar, the course will cover a range of methods and real world use cases relevant to practitioners in the field of Risk and Finance.

## Course Structure

### Part 1: Lecture Series

#### [1. Lecture: Machine Learning and Risk Modeling](1_ml/ml.ipynb)
Participants will learn key ML concepts and algorithms, including:
- **ML pipelines**
- **Tree-based ML algorithms** (e.g., Regression Tree, Random Forest, Gradient Boosting)
- **Model training and model inference**

The focus will be on a step-by-step process of modeling risks and predicting future uncertain outcomes using popular Python libraries such as **scikit-learn** and **pandas**. The concrete use case looks into the uncertainty of what a leased out car can be resold at the end of the lease term, a typical risk faced by car leasing entities. The goal is to understand how machine learning can add value in assessing risks that organizations operating under uncertainty are facing.

#### [2. Lecture: Natural Language Processing and Risk Clustering](2_nlp/nlp.ipynb)
This session will cover NLP techniques and their evolution in recent years, including:
- **Text Vectorization** (e.g., tf-idf, word embeddings)
- **Text Clustering** via topic modeling

Participants will explore how these techniques can enhance the processing of insurance policies and the categorization of insurance claims both of which being common use cases in the insurance industry. While the use case is specific to insurers, the techniques can be easily adapted to other use cases and industries where professionals are facing unstructured text data.

#### [3. Lecture Generative AI and Risk Agents](3_genai/genai.ipynb)
On the third day, participants will explore the cutting-edge field of Generative AI (GenAI) and its application in risk analytics. The focus will be on building a Multi-Document Risk Agent, a conversational AI system that allows users to interact with and extract insights from annual reports of publicly listed companies. This session will cover:
- Introduction to **Generative AI** and **Large Language Models** (LLMs)
- **Retrieval-Augmented Generation** (RAG)
- The concept of **agents**

By the end of the day, participants will have a working understanding of how to leverage GenAI, LLMs, and RAG to create advanced tools for making fast amounts of information accesible in risk management using popular Python frameworks such as **LlamaIndex**, **LangChain** and **LangGraph**.

### Part 2: Take-Home Assignment
After the lecture sessions, participants will work on a practical problem. They will apply the methods learned to evaluate whether and under what constraints a specific real world problem can be solved. Participants will have **2-4 weeks** to complete this assignment and will present their findings in a **presentation** during a follow-up session lasting **1-2 days** (depending on the number of participants).
More details about the take home assignment will be given in the first class.

## Materials and Resources
All course materials will be provided via GitHub. Prior to the seminar, participants will receive introductory slides explaining Git and GitHub, along with setup instructions to ensure everyone is well-prepared and participants can seamlessly engage with the course content.

## Prerequisites
Participants should have **sound knowledge of Python programming**. This foundational skill is crucial for effectively following the course materials. Prior to enrollment of the course, students will get access to a short Python quiz which they can use as a self-assessment for their level of Python knowledge. It is generally recommended to enroll in the course only if you feel comfortable doing the quiz. Otherwise it is recommended to prepare for the course by familiarizing with Python first. A great resource is this engaging, free 2-hour crash [course](https://www.youtube.com/watch?v=K5KVEU3aaeQ) for complete beginners.

## Technical Setup-up
Please follow the steps below to get started with the ccourse material:

1. Make sure [python 3.12](https://www.python.org/downloads/release/python-3120/) is installed on your local machine
2. Make also sure [git](https://git-scm.com/downloads) is installed on your local machine
3. Clone this repository on your local machine using git
```
git clone https://github.com/ai-analytics-jlu-ws25-26/courses.git
```
4. Create a virtual environment and install required dependencies into it
```
python -m venv .venv
```
5. Activate newly created virtual environment
```
source .venv/Scripts/activate
```
6. Install all required dependencies into the newly created virtual environment
```
pip install -r requirements.txt
```
7. Open Jupyter Lab:
```
juypter lab
```
