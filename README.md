# Brand_Delights_Disappointment_Maps
Build Brand Delights/Disappointment maps using the Amazon Product Reviews Dataset from Kaggle. It includes objectives, data &amp; preprocessing steps, modeling &amp; visualization.

## This repo contains notebooks and scripts to:

Preprocess Amazon reviews from Kaggle.

Build TF-IDF representations and word clouds per brand.

Run LDA and K-means to extract topics and clusters.

Create a Brand Map from brand-level centroids or topic mixes.

Extract delighters and disappointers using sentiment-partitioned term scoring.
The analysis is exploratory — useful for hypothesis generation — but not a replacement for representative brand health measurement.

# 📂 Dataset

Source: Amazon Product Reviews Dataset – Kaggle

Fields: id, brand , categories , colors , dateAdded  , dateUpdated , dimension , keys , reviews.rating , reviews.text , reviews.title , reviews.userCity etc
Subset: Choose a single product category (e.g., “Headphones”, “Tablet”) and focus on leading brands by review volume.

# 🧹 Data Preprocessing

Steps performed:

Filtering: Select top brands within a chosen product category.

Cleaning:

Convert text to lowercase

Remove special characters, punctuation, and numbers

Remove stopwords (NLTK/spaCy) and custom stop words (amazon, product, device..etc)

Lemmatization for better interpretability

Tokenization: Keep unigrams and bigrams (e.g., battery life, customer service).

Balancing: Handle brand imbalance by sampling proportionally.

Feature Generation: Build TF-IDF matrices using TfidfVectorizer with ngram_range=(1,2) and min_df=5.

☁️ Wordclouds per Brand

Generate Wordclouds for each brand based on TF-IDF frequencies.

Create separate clouds for:

All reviews

Positive reviews (rating ≥ 4)

Negative reviews (rating ≤ 2)

Compare to identify dominant discussion themes per brand.

Example insights:

Brand A’s positive word cloud emphasizes “sound quality” and “battery life,” while negatives focus on “connection” and “support.”

# 🧾 Topic Modeling (LDA)

Goal: Discover latent discussion themes within reviews.

Apply Latent Dirichlet Allocation (LDA) to the TF-IDF or Count matrix.

Tune the number of topics (8–20) using coherence scores.

Extract and label top terms per topic.

Compute average topic weights per brand → reveal brand-specific themes.

Output files:

lda_topics.csv — list of top words per topic

topic_distribution_per_brand.csv

# 🔍 Document Clustering (K-Means)

Objective: Group reviews into semantic clusters using term patterns.

Apply K-Means clustering on reduced TF-IDF features (via SVD/PCA).

Extract 3–5 clusters and interpret each cluster’s main theme.

Compare brand composition across clusters.

Output files:

cluster_summary.csv — key terms and representative reviews per cluster

# 🗺️ Brand Map Generation

Approach 1 – TF-IDF Centroid Method:

Compute each brand’s mean TF-IDF vector.

Apply MDS / PCA / UMAP to project into 2D.

Plot brands in a semantic space — nearby brands share similar review language.

Approach 2 – Topic Distribution Method:

Average topic probabilities per brand.

Project via PCA to visualize topic-based brand proximity.

Output:
brand_map.png — a 2D brand positioning map with labels, colors by average sentiment, and sizes by review count.

# 🌟 Delighters & ⚠️ Disappointers

Partition reviews into Positive (≥4★) and Negative (≤2★).

Compute term associations using log-odds ratios or term frequency contrasts.

Rank terms most associated with each sentiment per brand.

Interpret results:

High-scoring positive terms → Delighters

High-scoring negative terms → Disappointers

Output:
brand_delighters_disappointers.csv

# 📊 Evaluation Metrics

Topic coherence (LDA)

Silhouette score (K-Means)

Brand-topic distinctiveness via cosine similarity

Manual review of sample documents for interpretability
