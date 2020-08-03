# TeachEasy
Levaraging the power of NLP, we have developed **TeachEasy** with the aim of reducing the manual work for School Teachers in the domain of Question and Answer generation, Text summarization, and Paraphrasing. 

## Requirements.
```
pip install -U transformers==3.0.2
pip install streamlit==0.64.0
pip install -U torch==1.6.0

```
## Technical stuff

Question generation, Summarization and Paraphrasing is achieved using T5 model. Answer generation is developed via DistilBert.

## How to use repo

```streamlit run app.py```
or 
```streamlit run main.py```

# Results and App snapshots

<h1> Main template interface </h1>
<img src="1.jpg">
<h1> Answer extraction from questions </h1>

<img src="2.jpg">

<h1> Summarization with spacy using Text Summarization( Extractive Summarization) </h1>

<img src="3.jpg">
<h1> Summarization using T5 (Abstractive Summarization) </h1>

<img src="4.jpg">
<h1> Paraphrasing of Question</h1>

<img src="5.jpg">
<h1> Boolean Question generation </h1>

<img src="6.jpg">
