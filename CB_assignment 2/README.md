# Assignment 1 - Contextual Bandit–Based News Recommendation System

This repository implements a **Contextual Bandit** system for news article recommendation, combined with a **user classification** stage. 


## 1. Project Overview

1. **Goal**: Build a system that first **classifies** users into one of three categories (`User1`, `User2`, `User3`) and then **recommends** a suitable news category (among `Entertainment`, `Education`, `Tech`, `Crime`) using a **Contextual Bandit** approach.
2. **Approach**:
   - **Step 1**: Preprocess user and news article data.
   - **Step 2**: Train a **Decision Tree** to classify users into `User1`, `User2`, or `User3`.
   - **Step 3**: Train **Contextual Bandit** policies (Epsilon-Greedy, UCB, SoftMax) for each user context.  
   - **Step 4**: Given a **new user**, classify them → pick a bandit arm → retrieve and return a random article from that category.
3. **Dataset**:
   - **train_users.csv** and **test_users.csv**: user features + label (User1, User2, User3).
   - **news_articles.csv**: news articles with various categories, from which we only keep `Entertainment`, `Education`, `Tech`, and `Crime`.

---


## Instructions to Run & Replicate

1. **Clone or Download** this repository.
2. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   # Also ensure you install sampler-1.0-py3-none-any.whl if needed:
   pip install /path/to/sampler-1.0-py3-none-any.whl
3. **Run Bandit_Assignment.ipynb file**

