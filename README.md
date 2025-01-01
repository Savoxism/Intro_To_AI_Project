# **Sentiment Analysis Using Fine-Tuned DistilBERT**

## **Abstract**
This project fine-tunes **DistilBERT**, a lightweight variant of **BERT**, for sentiment analysis. The model classifies text as **positive**, **negative**, or **neutral**, leveraging advanced transformer techniques like **self-attention** and **knowledge distillation**. Using a labeled dataset of customer reviews, the fine-tuned model demonstrates high accuracy, precision, and recall, making it an efficient solution for real-world applications with limited computational resources. 

If you wish to test the model, please run the main.py file. If you wish to evaluate all the codes used, including data cleaning, EDA, and finetuning the model, please kindly visit this Kaggle notebook: https://www.kaggle.com/code/phuannguyen/intro-to-ai-project

The dataset is named "twitter-entity-sentiment-analysis"
---

## **Purpose**
The objective is to develop an efficient sentiment analysis model capable of:
- Understanding contextual semantics.
- Addressing class imbalance.
- Operating effectively in resource-constrained environments.

---

## **Requirements**
### **Software and Libraries used**
- Python 3.12+
- PyTorch
- Hugging Face Transformers
- NumPy
- Pandas 
- Other packages specified in requirements.txt

Install dependencies using:
```bash
pip install -r requirements.txt
