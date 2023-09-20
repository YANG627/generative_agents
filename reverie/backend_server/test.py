"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import json
import random
import openai
import time 


import requests
import json
def gewu_api_request(text):
  # url = "http://1.117.203.227/gpu/gewu_api/chatbot/gewu"
  url = "http://222.222.172.114:81/gpu/small_vile_llm_api"

  payload = json.dumps({
    "text": text,
    "top_p": 0.85,
    "temperature": 0.1
  })
  headers = {
    'Content-Type': 'application/json'
  }

  response = requests.request("POST", url, headers=headers, data=payload)

  # print(response.text)
  return response.text


from utils import *
openai.api_key = openai_api_key

def ChatGPT_request(prompt): 
  """
  Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
  server and returns the response. 
  ARGS:
    prompt: a str prompt
    gpt_parameter: a python dictionary with the keys indicating the names of  
                   the parameter and the values indicating the parameter 
                   values.   
  RETURNS: 
    a str of GPT-3's response. 
  """
  # temp_sleep()
  try: 
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": prompt}]
    )
    return completion["choices"][0]["message"]["content"]
    chatgpt_response = completion["choices"][0]["message"]["content"]
    print("chatgpt_response: ",chatgpt_response)
    response = gewu_api_request(prompt)
    print("llm_response: ",response)
    return response
  except: 
    print ("ChatGPT ERROR")
    return "ChatGPT ERROR"

prompt = """
---
Character 1: Maria Lopez is working on her physics degree and streaming games on Twitch to make some extra money. She visits Hobbs Cafe for studying and eating just about everyday.
Character 2: Klaus Mueller is writing a research paper on the effects of gentrification in low-income communities.

Past Context: 
138 minutes ago, Maria Lopez and Klaus Mueller were already conversing about conversing about Maria's research paper mentioned by Klaus This context takes place after that conversation.

Current Context: Maria Lopez was attending her Physics class (preparing for the next lecture) when Maria Lopez saw Klaus Mueller in the middle of working on his research paper at the library (writing the introduction).
Maria Lopez is thinking of initating a conversation with Klaus Mueller.
Current Location: library in Oak Hill College

(This is what is in Maria Lopez's head: Maria Lopez should remember to follow up with Klaus Mueller about his thoughts on her research paper. Beyond this, Maria Lopez doesn't necessarily know anything more about Klaus Mueller) 

(This is what is in Klaus Mueller's head: Klaus Mueller should remember to ask Maria Lopez about her research paper, as she found it interesting that he mentioned it. Beyond this, Klaus Mueller doesn't necessarily know anything more about Maria Lopez) 

Here is their conversation. 

Maria Lopez: "
---
Output the response to the prompt above in json. The output should be a list of list where the inner lists are in the form of ["<Name>", "<Utterance>"]. Output multiple utterances in ther conversation until the conversation comes to a natural conclusion.
Example output json:
{"output": "[["Jane Doe", "Hi!"], ["John Doe", "Hello there!"] ... ]"}
"""

# print ()
# ChatGPT_request(prompt)



from sentence_transformers import SentenceTransformer
def get_embedding(text, model_name="paraphrase-distilroberta-base-v1"):
    # Initialize the SentenceTransformer model
    model = SentenceTransformer(model_name)

    text = text.replace("\n", " ")
    if not text:
        text = "this is blank"

    # Get the embedding
    embedding = model.encode(text)
    print("local embedding", numpy.array(embedding).shape,len(embedding))

    return embedding



def llm_api_embedd_request(text):
  # url = "http://1.117.203.227/gpu/gewu_api/chatbot/gewu"
  url = "http://222.222.172.114:81/gpu/small_vile_llm_api"

  payload = json.dumps({
    "text": text,
    "action": "embedding",
  })
  headers = {
    'Content-Type': 'application/json'
  }

  response = requests.request("POST", url, headers=headers, data=payload)

  # print(response.text)
  return response.text


import numpy
def get_embedding_gewu(text, model_name="paraphrase-distilroberta-base-v1"):
    
    text = text.replace("\n", " ")
    if not text:
        text = "this is blank"
    embedds = llm_api_embedd_request(text)
    print("gewu embedding", numpy.array(embedds).shape,len(embedds))
    return embedds

def get_embedding_openai(text, model="text-embedding-ada-002"):
  text = text.replace("\n", " ")
  if not text: 
    text = "this is blank"
  res = openai.Embedding.create(
          input=[text], model=model)['data'][0]['embedding']
  print("openai, embedding",numpy.array(res).shape, len(res))
  return res
  
text = "Bill is sleeping"
get_embedding(text)
get_embedding_openai(text)
get_embedding_gewu(text)








