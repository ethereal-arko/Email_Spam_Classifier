# Email Spam Classifier

A machine learning project that classifies text messages as **Spam** or **Ham** using TF-IDF and Logistic Regression.

## Project Overview

This project explores text classification through data analysis, text preprocessing, model training, class-imbalance handling, and performance evaluation.

## Objectives

- Explore and clean the dataset.
- Convert text into numerical features using TF-IDF.
- Train a Logistic Regression classifier.
- Compare standard and class-weighted models.
- Analyze classification errors and prediction confidence.
- Evaluate the model using standard classification metrics.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression

## Dataset

The project uses `mail_data.csv`, which contains message text and category labels.

| Item | Value |
|---|---:|
| Original records | 5,572 |
| Duplicate records removed | 415 |
| Records after cleaning | 5,157 |
| Ham messages | 4,516 |
| Spam messages | 641 |

## Methodology

1. Data loading and inspection
2. Exploratory Data Analysis (EDA)
3. Duplicate removal and label encoding
4. Stratified train-test split
5. TF-IDF feature extraction
6. Baseline Logistic Regression
7. Class-imbalance experiment using balanced class weights
8. Error and prediction-confidence analysis
9. Decision-threshold comparison
10. Final evaluation

## Results

Held-out test-set performance:

| Metric | Baseline Model | Balanced Model |
|---|---:|---:|
| Accuracy | 96.51% | 97.48% |
| Spam Precision | 97.92% | 89.84% |
| Spam Recall | 73.44% | 89.84% |
| Spam F1-score | 83.93% | 89.84% |

The balanced model improved spam recall from **73.44% to 89.84%**, detecting more spam messages at the cost of additional false positives.

### Confusion Matrix — Balanced Model

| Actual / Predicted | Ham | Spam |
|---|---:|---:|
| Ham | 891 | 13 |
| Spam | 13 | 115 |

## How to Run

1. Open `Email_Spam_Classifier.ipynb` in Google Colab or Jupyter Notebook.
2. Upload `mail_data.csv` to the expected location or update the dataset path.
3. Install the required libraries if necessary:

   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

4. Run the notebook cells in order.

## Limitations

- The dataset contains SMS-style messages and may not represent all email spam.
- TF-IDF and Logistic Regression have limitations in understanding context.
- Performance on independent, real-world datasets has not yet been established.

## Future Improvements

- Test on independent email datasets.
- Explore additional text features and classification algorithms.
- Evaluate generalization on unseen messages.
- Develop an interactive spam-classification application.

## Conclusion

This project demonstrates a machine learning workflow for text classification, including data exploration, preprocessing, model comparison, and error analysis. It provides a foundation for further study in natural language processing and machine learning.

## Learning Resources & Acknowledgements

This project was developed as a hands-on machine learning learning project.

- **YouTube Tutorial:** Used as a learning resource to understand the project workflow and related machine learning concepts.
- **ChatGPT (OpenAI):** Used for conceptual explanations, code assistance, debugging support, and understanding model evaluation.
- **My Work:** I worked through the notebook, executed the code, examined the evaluation results, and explored the model's classification errors to better understand its strengths and limitations.

The project helped me gain practical experience with text preprocessing, TF-IDF feature extraction, Logistic Regression, class imbalance, and model evaluation.

*Note: The project is based on learning resources and AI-assisted guidance. The original tutorial and dataset sources should be credited where applicable.*
