# Rural Theater Reviews NLP Pipeline

A dual-model sentiment analysis pipeline using VADER and RoBERTa to analyze theater reviews, evaluated with a full classification framework including precision, recall, and F1.
Layered in BERTopic topic modeling to surface 19 distinct themes across the reviews, giving the organization its first structured understanding of audience perception.

Work done for a summer internship with Door Shakespeare, a nonprofit performing arts company in Door County, Wisconsin.

## Research Question

---

**What factors most influence customer satisfaction at rural performing arts venues?**

- Are visitors discussing the performance itself or the broader venue experience?
- What operational issues most affect customer ratings?
- Do different venues attract reviews about different aspects of the experience?


## Technology

---

`Python`  
`Jupyter Notebook`  
`Pandas`  
`NLTK's VADER`  
`HuggingFace's RoBERTa`  
`BERTOPIC`  
`Scikit-learn`  
`Seaborn`  
`Matplotlib`  

## Data

---

650+ publicly available Google reviews scraped across five venues using Apify and Make.com. 420 reviews were usable after removing entries with no to little text.
The dataset is heavily skewed positive, with about 88.5% five-star and nearly 96% positive reviews (direct implications for model evaluation and findings).

## Methodology

---

**Sentiment Analysis**
- Baseline model: NLTK's VADER - lexicon-based analysis
- Progressive model: HuggingFace's RoBERTa - fine tuned transformer analysis (`cardiffnlp/twitter-roberta-base-sentiment`)
- Models evaluated on classification metrics (precision, recall, F1, confusion matrix) against star-rating-derived ground truth labels
- Binary evaluation (positive/negative) used as primary metric given near-zero neutral class support

**Topic Modeling**
- BERTopic with c-TF-IDF and bigram CountVectorizer
- 19 distinct audience perception themes clustered across 420 reviews
- Multi-label topic assignment using probability thresholding to better handle long, multi-theme reviews
- Topics manually labeled and grouped into four overarching categories: Performance, Operational, Tourism Context, and General

## Key Findings

---

RoBERTa outperformed VADER on minority class detection (negative F1: 0.72 vs 0.64), driven by its ability to understand context rather than relying on lexicon matching.
VADER over-classified reviews as neutral, a known limitation of lexicon-based approaches on domain-specific text.

Operational topics had the lowest average satisfaction of any category (4.51 avg stars, 0.78 avg sentiment), well below Performance (4.81, 0.91) and Tourism Context (4.90, 0.94).
Audiences at rural performing arts venues are highly satisfied with the artistic experience, whereas dissatisfaction is almost entirely driven by operational issues, with a focus on seating and ticketing.

Venue-level topic distributions revealed meaningful differences in what each organization's audiences discuss, with Northern Sky reviews defined by environment and performer talent, TAP dominated by general enjoyment, and Door Shakespeare showing the highest seating complaint concentration of any venue.
