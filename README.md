# Twitter Sentiment Analysis
This project focuses on predicting Twitter Sentiment using machine learning model trained on the **twitter_training**. The dataset contains **Id, Entity, Sentiment & Text**. Twitter's vast stream of real-time user-generated text (tweets) contains valuable insights into public opinion, but manually analyzing this volume is impossible. Understanding the sentiment (positive, negative, neutral & irrelevant) expressed in these tweets regarding specific topics, products, or events is crucial for businesses, researchers, and policymakers. The objective is to develop a machine learning model that accurately classifies the sentiment of Twitter data, enabling automated analysis of public opinion and providing real-time insights into trends and reactions.
## Dataset Overview
- **Source**: twitter_training
- **Rows**: 74682
- **Columns**: 4 (including the target variable)
- **Target Variable**:
  - Sentiment:
     - Positive review
     - Negative review
     - Neutral review
     - Irrelevant review
- **Features**:
  - Id
  - Entity
  - Text
       
## Project Workflow
### 1. **Data Preprocessing**
- Handling missing values
- Removing unwanted columns
- Removing duplicates
- Tokenization
- Lowercasing
- Stopword Removal
- Lemmatization
- Removing Punctuation & Special Characters
- Removing URLs & Numbers
- Removing email id
- Word Embedding (Vectorization):
  Applied & verified individually
  - Bag of Words (BoW) – Counts word occurrences
  - TF-IDF (Term Frequency-Inverse Document Frequency) – Measures word importance
- Encoding target feature(`LabelEncoder`)
- Splitting dataset into training & testing sets (`train_test_split`)
  
### 2. **Machine Learning Model**
The project implements machine learning model for multi-class classification:
- **RandomForestClassifier**

### 3. **Model Evaluation**
- **Accuracy Score**
- **Confusion Matrix**
- **Classification Report**

## Installation & Usage

### Prerequisites
Ensure you have the following installed:
- Python 3.10 and above
- Jupyter Notebook
- Required libraries (`pandas`, `numpy`, `scikit-learn`, `nltk`)

### Running the Notebook
1. Clone the repository:
   git clone https://github.com/SPV-413/Twitter-Analysis.git
2. Navigate to the project folder:
   cd Twitter-Analysis
3. Open the Jupyter Notebook:
   jupyter notebook Twitter analysis.ipynb

## Results
- Using **Bag of Words (BoW)** for feature extraction resulted the highest accuracy score with the **RandomForestClassifier** model when compared to TF-IDF method. This suggests that Bag of Words (BoW) method effectively captured the relevant information in the dataset, highlighting their robustness in certain contexts. Enabling automated analysis of public opinion and providing real-time insights into trends and reactions.

## Contact
For any inquiries, reach out via GitHub or email.
