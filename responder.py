from transformers import pipeline,BertForQuestionAnswering,AutoTokenizer
from random import choice
from requests import get
from os import getenv
from web_parser import parser

SEARCH_ENGINE_ID="d6e0e4194507c4642"
API_KEY="AIzaSyA-vkDczHM8JLKrdER_BcPADXx5uOJjQeo"
google_api_url="https://www.googleapis.com/customsearch/v1"

QA_MODEL=BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
tokenizer=AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')
QA_PIPELINE=pipeline("question-answering",model='deepset/bert-base-cased-squad2',tokenizer='deepset/bert-base-cased-squad2')


pipe = pipeline("text-classification", model="dipesh/Intent-Classification-Bert-Base-Cased")
intents={'greet':["Hey! I appreciate the friendly hello. What's your question?","Hi there! Thanks for the greeting. Is there a particular query or question on your mind?","Hello! I'm at your service. Any specific questions or topics you'd like to explore about science?","Greetings! Your hello is welcomed. What's your question or inquiry for today?","Hey! I'm here for you. Do you have a query or something you'd like to discuss?","Hi! Thanks for reaching out. What can I help you with today? Any specific questions?","Hey! Your greeting is much appreciated. Do you have a question or something you'd like to know?"],
         'goodbye':["Goodbye! Feel free to return anytime.",
                    "Take care! See you soon.",
                    "Farewell! Reach out if needed.",
                    "So long! Have a great day.",
                    "Goodbye! Stay in touch.",
                    "Bye for now! Questions? Just ask.",
                    "See you later! Take care.",
                    "Adios! Need anything else?",
                    "Bye! Until we meet again."]}

def getResponse(text):
    detected_intent=None
    answer_found=False
    conversation="User : "
    usr_inp=text
    conversation+=usr_inp+"\n"
    result=pipe(usr_inp)[0]

    if result['score']>0.5:
        detected_intent=result['label']
        if detected_intent in intents.keys():
            res=choice(intents[detected_intent])
            conversation+=f"Assistant : {res}"
            answer_found=True
            return res
        
    else:
    # Check if Answer availble in previous conversation
        with open("page_content.txt",'r') as f:
            QA_input={
                'question':usr_inp,
                'context': f.read()
            }
            qa_res=QA_PIPELINE(QA_input)
            if qa_res['score']>0.1:
                conversation+=f"Assistant : {qa_res['answer']}"
                answer_found=True
                return qa_res['answer']

        # if not available, search on web and retrieve answer
        if not answer_found:
            params={
                'q':usr_inp,
                'key':API_KEY,
                'cx':SEARCH_ENGINE_ID,
            }
            


            response=get(google_api_url,params=params)
            results=response.json()['items']
            if len(results)>0:
                link=results[0]['link']
                final_text=parser(link)

                with open("page_content.txt",'a') as f:
                    f.write(f"\n\n{usr_inp}\n")
                    f.write(final_text+"\n\n")
                
                QA_input={
                    'question':usr_inp,
                    'context': final_text
                }
                qa_res=QA_PIPELINE(QA_input)
                if qa_res['score']>0.1:
                    conversation+=f"Assistant : {qa_res['answer']}"
                    return qa_res['answer']
                else:
                    return choice(['sorry! can\'t answer that','I don\'t know','No idea about that','Don\'t know!'])
            else:
                return choice(['sorry! can\'t answer that','I don\'t know','No idea about that','Don\'t know!'])
    return choice(['sorry! can\'t answer that','I don\'t know','No idea about that','Don\'t know!'])        
    