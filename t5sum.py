import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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