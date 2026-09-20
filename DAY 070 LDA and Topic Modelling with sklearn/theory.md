# DAY 20 — Topic Modeling With Scikit-Learn | LDA Topic Modeling in Python

## 1. Introduction

**Topic Modeling** is an NLP technique used to discover the hidden or main topics present in a collection of text documents.

Instead of manually reading every document and assigning a topic, a Topic Modeling algorithm analyzes the words in the documents and identifies groups of words that frequently occur together.

In this project, we use **LDA (Latent Dirichlet Allocation)** with **Scikit-Learn** to discover topics from a textual dataset.

### Main Workflow

**Text Documents → Preprocessing → CountVectorizer → Term-Document Matrix → LDA → Topics**

---

## 2. What is Topic Modeling?

Topic Modeling is an **unsupervised learning technique** used to automatically find topics in a collection of documents.

It does not require predefined topic labels.

For example, if a collection of documents contains words such as:

- Python
- Java
- programming
- languages

the model may identify these words as belonging to one common topic.

Similarly, words such as:

- machine learning
- deep learning
- neural networks

may form another topic.

The important point is that Topic Modeling discovers these groups from the text data itself.

---

## 3. What is LDA?

**LDA stands for Latent Dirichlet Allocation.**

LDA is a popular probabilistic model used for Topic Modeling.

The basic idea is that:

- A document can contain multiple topics.
- A topic is represented by a group of related words.
- Words that commonly occur together help describe a topic.
- LDA learns these topic-word relationships from the collection of documents.

For example, a document may contain information about both **Machine Learning** and **NLP**. LDA can represent the document using a mixture of these topics instead of forcing it into only one category.

---

## 4. Why is LDA Called an Unsupervised Technique?

LDA is generally used without predefined target labels.

The input consists of text documents, and the algorithm attempts to discover meaningful topic structures from those documents.

Therefore, Topic Modeling is different from supervised classification where the model is trained using known labels.

---

## 5. Text Preprocessing

Before applying Topic Modeling, the text needs to be cleaned.

The notebook uses the following preprocessing steps:

### 5.1 Lowercasing

All characters are converted into lowercase.

Example:

`Python Programming`

becomes:

`python programming`

This prevents the model from treating uppercase and lowercase versions of the same word as different terms.

### 5.2 Removing Punctuation

Punctuation marks are removed from the text.

For example:

`NLP is useful!`

becomes:

`NLP is useful`

### 5.3 Removing Stopwords

Stopwords are common words that usually carry less useful information for topic discovery.

Examples include:

- and
- the
- is
- are

The notebook uses Scikit-Learn's `ENGLISH_STOP_WORDS` for this purpose.

### 5.4 Tokenization

The cleaned text is divided into individual words called **tokens**.

For example:

`machine learning is fascinating`

can be represented as:

`machine`, `learning`, `fascinating`

### 5.5 Stemming/Lemmatization

Stemming or lemmatization can optionally be used to reduce words to their root or base form.

These steps are mentioned as optional preprocessing techniques in the notebook.

---

## 6. CountVectorizer

Machine learning algorithms cannot directly work with raw text.

Therefore, the text must be converted into numerical features.

The notebook uses:

**CountVectorizer**

CountVectorizer converts a collection of text documents into a numerical **term-document matrix**.

It counts how many times each word occurs in each document.

### Example Concept

Suppose we have:

- Document 1: `python programming`
- Document 2: `python language`

The vocabulary may contain:

`python`, `programming`, `language`

The documents can then be represented numerically according to the occurrence of these words.

This numerical representation is used as the input for LDA.

---

## 7. Term-Document Matrix

A term-document matrix represents the relationship between words and documents.

In this representation:

- Rows represent documents.
- Columns represent terms/features.
- Values represent word counts.

The CountVectorizer output is stored as a sparse matrix because most documents contain only a small portion of the complete vocabulary.

The notebook also demonstrates converting the sparse matrix into a dense format using `.todense()` for inspection.

---

## 8. Applying LDA

After converting the cleaned documents into numerical features, the LDA model is applied.

Scikit-Learn provides LDA through:

`LatentDirichletAllocation`

The notebook creates the model with:

- `n_components=2` → the model is asked to discover 2 topics.
- `random_state=0` → helps make the result reproducible.

The LDA model is then fitted using the CountVectorizer output.

### Important

`n_components` represents the **number of topics we want the model to discover**.

Choosing the number of topics is an important part of Topic Modeling because different values can produce different topic structures.

---

## 9. How LDA Represents Topics

After training, LDA provides information about the importance of words for each discovered topic.

The notebook accesses this information through:

`lda.components_`

Each topic contains values associated with the vocabulary words.

Words with higher topic-related values are treated as important words for that topic.

By sorting these values, we can display the top words associated with each topic.

---

## 10. Displaying Topics

The notebook uses the vocabulary generated by CountVectorizer:

`vectorizer.get_feature_names_out()`

This allows the numerical topic information to be converted back into understandable words.

The top words are selected using the sorted topic values.

The result can look conceptually like:

**Topic 1**
- programming
- python
- languages
- java

**Topic 2**
- learning
- machine
- neural
- networks

The exact topics depend on the dataset and model configuration.

---

## 11. Making Topic Output More Representative

The notebook defines a `display_topics()` function to organize the discovered topics.

The function:

1. Iterates through every topic.
2. Finds the highest-valued words.
3. Stores the important words for each topic.
4. Returns the results in a dictionary.

The topics are then converted into a **Pandas DataFrame**.

This makes the discovered topics easier to read and inspect.

---

## 12. Important Scikit-Learn Components

### `CountVectorizer`

Converts text documents into numerical word-count features.

### `LatentDirichletAllocation`

Applies LDA to discover hidden topics in the numerical text representation.

### `ENGLISH_STOP_WORDS`

Provides a collection of English stopwords that can be removed during preprocessing.

### `get_feature_names_out()`

Returns the vocabulary/features learned by CountVectorizer.

### `components_`

Contains the learned topic-word information from the LDA model.

### `Pandas DataFrame`

Used to organize and display the discovered topics in a readable tabular format.

---

## 13. Complete Conceptual Pipeline

The project follows this pipeline:

### Step 1 — Create or collect text documents

A collection of textual documents is prepared.

### Step 2 — Preprocess the text

The documents are cleaned using:

- Lowercasing
- Punctuation removal
- Stopword removal
- Tokenization
- Optional stemming/lemmatization

### Step 3 — Convert text into numbers

CountVectorizer converts the cleaned text into a term-document matrix.

### Step 4 — Apply LDA

LDA analyzes the numerical document representation and discovers the requested number of topics.

### Step 5 — Extract important words

The highest-valued words for each topic are selected.

### Step 6 — Display the topics

The discovered topics are displayed as lists of important words and organized into a DataFrame.

---

## 14. Topic Modeling vs Text Classification

Topic Modeling and Text Classification are different tasks.

### Topic Modeling

- Usually unsupervised.
- Finds hidden topics in text.
- Does not require predefined class labels.
- LDA is a commonly used technique.

### Text Classification

- Usually supervised.
- Learns from labeled examples.
- Predicts a predefined class or category.

For this project, the goal is **discovering topics**, not predicting predefined labels.

---

## 15. Advantages of Topic Modeling

Topic Modeling can be useful when working with a large collection of documents.

It can help with:

- Discovering hidden themes.
- Exploring large text datasets.
- Organizing documents.
- Understanding the main subjects in a corpus.
- Supporting text analysis and information retrieval.

---

## 16. Limitations

Topic Modeling does not automatically guarantee that every discovered topic will have a perfectly clear human interpretation.

The quality of the topics can depend on:

- The quality of the text data.
- Text preprocessing.
- The number of topics selected.
- The vocabulary.
- The amount of available data.

Therefore, the generated topic words should be interpreted in the context of the dataset.

---

## 17. Key Terms to Remember

| Term | Meaning |
|---|---|
| NLP | Natural Language Processing |
| Topic Modeling | Technique for discovering hidden topics in text |
| LDA | Latent Dirichlet Allocation |
| Unsupervised Learning | Learning without predefined target labels |
| Document | A text item used as input |
| Token | Individual word or text unit |
| Stopword | Common word removed during preprocessing |
| CountVectorizer | Converts text into word-count features |
| Vocabulary | Collection of features/words learned from the text |
| Term-Document Matrix | Numerical representation of documents and terms |
| `n_components` | Number of topics requested from LDA |
| `components_` | Learned topic-word information |

---

## 18. Important Points for Revision

1. **LDA = Latent Dirichlet Allocation.**
2. LDA is used for **Topic Modeling**.
3. Topic Modeling is generally an **unsupervised learning** task.
4. Raw text must be converted into numerical features before applying LDA.
5. This notebook uses **CountVectorizer** for text-to-numeric conversion.
6. Text preprocessing includes lowercasing, punctuation removal, stopword removal, and tokenization.
7. `n_components` specifies the number of topics to discover.
8. `lda.components_` provides topic-word information.
9. `get_feature_names_out()` provides the vocabulary words.
10. The top words of a topic help us interpret what that topic represents.
11. Pandas DataFrame can be used to display the discovered topics clearly.
12. The notebook uses **Scikit-Learn** for CountVectorizer and LDA.

---

## 19. Conclusion

In this project, we learned how **Topic Modeling** can be performed using **LDA with Scikit-Learn**.

The text documents are first cleaned and preprocessed. CountVectorizer then converts the cleaned text into a numerical term-document matrix. LDA analyzes this representation and discovers the requested number of hidden topics.

Finally, the most important words associated with each topic are extracted and displayed so that the topics can be interpreted.

This workflow provides a practical foundation for analyzing collections of text documents using NLP and unsupervised machine learning.
