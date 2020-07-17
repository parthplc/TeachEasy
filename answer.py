import transformers
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

def answergen(context, question):

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

    encoding = tokenizer.encode_plus(question, context)

    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]

    start_scores, end_scores = model(torch.tensor([input_ids]), attention_mask=torch.tensor([attention_mask]))

    ans_tokens = input_ids[torch.argmax(start_scores) : torch.argmax(end_scores)+1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens , skip_special_tokens=True)

    print ("\nQuestion ",question)
    #print ("\nAnswer Tokens: ")
    #print (answer_tokens)

    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
    #print ("\nAnswer : ",answer_tokens_to_string)
    return answer_tokens_to_string