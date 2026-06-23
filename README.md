# Rural Theater Reviews NLP Pipeline

A dual-model sentiment analysis pipeline using VADER and RoBERTA to analyze theater reviews, evaluated with a full classification framework including precision, recall, and F1.
Layered in BERTopic topic modeling to surface 19 distinct themes across the reviews, giving the organization its first structured understanding of audience perception.

Work done for summer internship with Door Shakespeare, a nonprofit performing arts company in Door County, Wisconsin.

### Technologies
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

### The Process
---
- Webscraping using Apify and Make.com performed to gather reviews across five nonprofit performing arts venues
- Cleaned data for ease of use and increased accuracy later in the project
- Initialized VADER model and performed baseline sentiment analysis modeling
- Initialized RoBERTa model and performed fine-tuned sentiment analysis
- Used classification report to compare metrics across both models
- Initialized BERTopic model for unsupervised learning in order to find sentiment themes
- Visualized topic modeling results
- Applied business logic and industry knowledge to find actionable results and improve customer satisfaction
