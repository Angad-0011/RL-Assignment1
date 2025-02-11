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


## 2. Data and Preprocessing

- **User data columns**: `['user_id', 'age', 'income', 'clicks', 'purchase_amount', 'label']`
- **News data columns**: `['link', 'headline', 'category', 'short_description', 'authors', 'date']`

**Preprocessing Steps**:

1. **Map user labels** to numeric: `User1->0`, `User2->1`, `User3->2`.  
2. **Clean** any missing data in user labels.  
3. **Clean news categories** by converting to title case and **filtering** for the 4 target categories:
   - `Entertainment->0`
   - `Education->1`
   - `Tech->2`
   - `Crime->3`
4. **Drop** rows that don’t match the above categories.

No NaNs remained in the user data after this process:


[Classification] Decision Tree Accuracy: 33.05%


Given 3 classes, ~33% is near random. The **classification report**:

          precision    recall  f1-score   support

   User1       0.32      0.26      0.29       672
   User2       0.34      0.51      0.41       679
   User3       0.33      0.21      0.26       649

accuracy                           0.33      2000




**Observation**: The model does relatively better on `User2` than on `User1` or `User3`. This suggests we may need more features or more data to distinguish user classes effectively.


## 4. Contextual Bandit Models

We treat **3 contexts** (`User1->0`, `User2->1`, `User3->2`) and **4 arms** (0..3). For a given **context** \(c\), pulling an **arm** \(a\) yields a reward via:

reward = reward_sampler.sample(j)

where `j = c*4 + a`.

We trained:

1. **Epsilon-Greedy (ε=0.1)**
2. **UCB (C=1)**
3. **SoftMax (T=1.0)**

Each for **10,000 steps** per context in isolation. We also **compared** multiple hyperparameters for each algorithm.

---

<a name="epsgreedy"></a>
### 4.1 Epsilon-Greedy

- Chooses a random arm with probability \(\epsilon\) or the **best arm** (argmax Q) with probability \(1-\epsilon\).
- We tested \(\epsilon \in \{0.1, 0.5, 0.7\}\).

<a name="ucb"></a>
### 4.2 UCB

- Uses an **upper confidence bound** for each arm: \( Q[a] + C \sqrt{\ln t / N[a]} \).
- We tested \(C \in \{1,2,3\}\).

<a name="softmax"></a>
### 4.3 SoftMax

- Converts Q-values into probabilities using a **softmax** function.  
- We tested **temperatures** \(T \in \{0.5, 1.0, 2.0\}\).

We store two main things:

1. **Actual rewards** from the sampler.
2. **Learned Q-values** (the bandit’s internal estimate of expected reward).

---

<a name="results"></a>
## 5. Results & Discussion

<a name="classresults"></a>
### 5.1 Classification Results

- **Accuracy**: ~33%.  
- **Confusion Matrix**: The model struggles to correctly identify `User1` and `User3`—partial improvement for `User2`.  
- **Future**: We might add more user-related features or try advanced models (e.g., RandomForest, XGBoost) or hyperparameter tuning (e.g., a deeper decision tree) to push beyond baseline.

---

<a name="banditresults"></a>
### 5.2 Bandit Results (Single Hyperparameters)

Below are final average rewards (actual sampler reward) for **ε=0.1**, **C=1**, and **T=1.0**:

| Model                  | Context0 | Context1 | Context2 | Overall |
|------------------------|----------|----------|----------|---------|
| **Epsilon-Greedy**(0.1)| 7.10     | 4.43     | 5.51     | 5.68    |
| **UCB**(1)            | 7.98     | 4.99     | 5.99     | 6.32    |
| **SoftMax**(1.0)      | 7.99     | 4.83     | 5.83     | 6.22    |

**Observations**:
- **UCB** and **SoftMax** yield slightly higher overall averages (around 6.2–6.3).
- Epsilon-Greedy’s best average is 5.68 with \(\epsilon = 0.1\). 
- **Context0** consistently achieves the highest rewards (~7–8). 
- **Context1** yields the lowest (~4–5). Possibly the sampler’s reward distribution is intrinsically worse for user2.

---

<a name="hyperparams"></a>
### 5.3 Hyperparameter Comparison

We tested:

- **Epsilon-Greedy**: \(\epsilon \in \{0.1, 0.5, 0.7\}.\)
  - \(\epsilon=0.1\) gave the best overall reward (~5.70).  
  - Higher \(\epsilon\) (0.5, 0.7) drastically reduced average reward (~3.18 and 1.93) due to over-exploration.

- **UCB**: \(C \in \{1,2,3\}\)
  - All gave ~6.32 overall reward. No major difference.

- **SoftMax**: \(T \in \{0.5,1.0,2.0\}\)
  - \(T=1.0\) gave ~6.23 overall reward, a bit higher than \(T=0.5\) (~6.19) and \(T=2.0\) (~5.73).  
  - Larger \(T\) means more random exploration, which can lower the final reward.



The bandit heavily favors **Education** for `Context=0` in this run.

---

<a name="recommendexample"></a>
### 5.4 Recommendation Example

For a new user `[age=29, income=29862, clicks=91, purchase_amount=270.91]`:

[Recommendations for new user]:

egreedy: Category=Education, Headline=Talk, Read and Sing to Kids to Close the Word Gap
ucb: Category=Education, Headline=The End of Reading in America and Other Related Matters
softmax: Category=Education, Headline=Education and Philanthropy



All policies chose **Education**, returning random articles from the `news_articles.csv` in that category.




---

## 6. Instructions to Run & Replicate

1. **Clone or Download** this repository.
2. **Install Requirements**:
   ```bash
   pip install -r requirements.txt
   # Also ensure you install sampler-1.0-py3-none-any.whl if needed:
   pip install /path/to/sampler-1.0-py3-none-any.whl
3. **Run Bandit_Assignment.ipynb file**

