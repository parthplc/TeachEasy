import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import numpy as np
import streamlit as st 
from fastpunct import FastPunct
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

from textblob import TextBlob 
import spacy
from gensim.summarization import summarize

from spacy.lang.pt.stop_words import STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import en_core_web_sm


# NLP Pkgs

# def set_seed(seed):
#   torch.manual_seed(seed)
#   if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(seed)

# set_seed(42)

# model = T5ForConditionalGeneration.from_pretrained('t5-small')
# tokenizer = T5Tokenizer.from_pretrained('t5-small')
# # device = torch.device('cpu')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # print ("device ",device)
# model = model.to(device)
# # fastpunct = FastPunct('en')
# tokenizer2 = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
# model2 = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
# model2 = model2.to(device)

# Function for Sumy Summarization
def sumy_summarizer(text):
  return summarize(text)


def punctuation(text):
  text_correct = text.lower()
  final = fastpunct.punct([text_correct])
  return str(final[0])


def summarization_spacy(text):

        
    nlp = en_core_web_sm.load()
    
    
    doc = nlp(text)


    corpus = [sent.text.lower() for sent in doc.sents ]
    
    cv = CountVectorizer(stop_words=list(STOP_WORDS))   
    cv_fit=cv.fit_transform(corpus)    
    word_list = cv.get_feature_names();    
    count_list = cv_fit.toarray().sum(axis=0)    

    
    word_frequency = dict(zip(word_list,count_list))

    val=sorted(word_frequency.values())

    # Check words with higher frequencies
    higher_word_frequencies = [word for word,freq in word_frequency.items() if freq in val[-3:]]
    # print("\nWords with higher frequencies: ", higher_word_frequencies)

    # gets relative frequencies of words
    higher_frequency = val[-1]
    for word in word_frequency.keys():  
        word_frequency[word] = (word_frequency[word]/higher_frequency)


    # SENTENCE RANKING: the rank of sentences is based on the word frequencies
    sentence_rank={}
    for sent in doc.sents:
        for word in sent :       
            if word.text.lower() in word_frequency.keys():            
                if sent in sentence_rank.keys():
                    sentence_rank[sent]+=word_frequency[word.text.lower()]
                else:
                    sentence_rank[sent]=word_frequency[word.text.lower()]
            else:
                continue

    top_sentences=(sorted(sentence_rank.values())[::-1])
    top_sent=top_sentences[:3]

    # Mount summary
    summary=[]
    for sent,strength in sentence_rank.items():  
        if strength in top_sent:
            summary.append(sent)

    summary = str(summary[0])+str(summary[1])+str(summary[2])
    # return orinal text and summary
    return summary



def answergen(context, question):

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

    encoding = tokenizer.encode_plus(question, context)

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    start_scores, end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))

    ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)

    # print ("\nQuestion ",question)
    #print ("\nAnswer Tokens: ")
    #print (answer_tokens)

    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
    #print ("\nAnswer : ",answer_tokens_to_string)
    return answer_tokens_to_string

  
def summary_t5(text):

  model = T5ForConditionalGeneration.from_pretrained('t5-small')
  tokenizer = T5Tokenizer.from_pretrained('t5-small')

  preprocess_text = text.strip().replace("\n","")
  t5_prepared_Text = "summarize: "+preprocess_text
  # print ("original text preprocessed: \n", preprocess_text)

  tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)


  # summmarize 
  summary_ids = model.generate(tokenized_text,
                                      num_beams=4,
                                      no_repeat_ngram_size=2,
                                      min_length=50,
                                      max_length=70,
                                      early_stopping=True)

  output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  return output


def paraphraser(text):

  model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')
  tokenizer = T5Tokenizer.from_pretrained('t5-small')

  sentence = text
  # sentence = "What are the ingredients required to bake a perfect cake?"
  # sentence = "What is the best possible approach to learn aeronautical engineering?"
  # sentence = "Do apples taste better than oranges in general?"


  text =  "paraphrase: " + sentence + " </s>"
  max_len = 128

  encoding = tokenizer.encode_plus(text,pad_to_max_length=True, return_tensors="pt")
  input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

  # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
  beam_outputs = model.generate(
      input_ids=input_ids, attention_mask=attention_masks,
      do_sample=True,
      max_length=128,
      top_k=120,
      top_p=0.98,
      early_stopping=True,
      num_return_sequences=3
  )
  final_outputs =[]
  for beam_output in beam_outputs:
      sent = tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
      if sent.lower() != sentence.lower() and sent not in final_outputs:
          final_outputs.append(sent)
  return final_outputs


def greedy_decoding (inp_ids,attn_mask):
  greedy_output = model.generate(input_ids=inp_ids, attention_mask=attn_mask, max_length=256)
  Question =  tokenizer.decode(greedy_output[0], skip_special_tokens=True,clean_up_tokenization_spaces=True)
  return Question.strip().capitalize()


def beam_search_decoding (inp_ids,attn_mask,model,tokenizer):
  # model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_boolean_questions')
  # tokenizer = T5Tokenizer.from_pretrained('t5-small')

  beam_output = model.generate(input_ids=inp_ids,
                                 attention_mask=attn_mask,
                                 max_length=256,
                               num_beams=10,
                               num_return_sequences=3,
                               no_repeat_ngram_size=2,
                               early_stopping=True
                               )
  Questions = [tokenizer.decode(out, skip_special_tokens=True, clean_up_tokenization_spaces=True) for out in
               beam_output]
  return [Question.strip().capitalize() for Question in Questions]


def question_generation(text):

  model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_boolean_questions')
  tokenizer = T5Tokenizer.from_pretrained('t5-small')
  passage = text
  truefalse ="yes"

  text = "truefalse: %s passage: %s </s>" % (passage, truefalse)


  max_len = 256

  encoding = tokenizer.encode_plus(text, return_tensors="pt")
  input_ids, attention_masks = encoding["input_ids"], encoding["attention_mask"]

  output = beam_search_decoding(input_ids,attention_masks,model,tokenizer)

  return output



def main():
  """ NLP Based App with Streamlit """

  # Title
  st.title("NLP with TeachEasy")
  st.subheader("Natural Language Processing On the Go..")
  st.markdown("""
      #### Description
      + This is a Natural Language Processing(NLP) Based App useful for Teachers and students
      """)
  # Tokenization
  
  if st.checkbox("Answer Extraction"):
    st.subheader("Extract Answer from text")

    context =  st.text_area("Type your context here")
    question= st.text_input('Question: ', None)

    if st.button("Extract"):
      answer = answergen(context, question)
      st.success(answer)

# Summarization
  if st.checkbox("Show Text Summarization"):
    st.subheader("Summarize Your Text")

    message = st.text_area("Enter Text","Type Here ..")
    summary_options = st.selectbox("Choose Summarizer",['spacy','T-5'])
    if st.button("Summarize"):
      if summary_options == 'spacy':
        
        # st.warning("Using Default Summarizer")
        st.text("Using Spacy ..")
        summary_result = summarization_spacy(message)
      else:
        st.text("Using T5")
        summary_result = summary_t5(message)

    
      st.success(summary_result)
  # Paraphrasing
  if st.checkbox("Show Question Paraphrasing"):
    st.subheader("Create similar questions")
    question= st.text_input('Question: ', None)

    if st.button("Paraphrase"):
      l = paraphraser(question)
      # st.success(l)
      st.write(l[0])
      st.write(l[1])
      st.write(l[2])
      # Paraphrasing
  if st.checkbox("Show Question Generation"):
    st.subheader("Create new question from context")
    message = st.text_area("Enter Text","Type Here ..")
    if st.button("Generate Questions"):
      l = question_generation(message)
      # st.success(l)
      st.write(l[0])
      st.write(l[1])
      st.write(l[2])


  st.sidebar.subheader("About App")
  st.sidebar.text("NLP with Teach-Easy")
  st.sidebar.info("Kudos to Vaibhav,Megha and Kritika ")
  

  st.sidebar.subheader("By")
  st.sidebar.text("Team Tinker AI")
  st.sidebar.text("Parth is awesome")
  
  

if __name__ == '__main__':
  main()
