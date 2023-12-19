from transformers import pipeline,BertForQuestionAnswering,AutoTokenizer
from random import choice
from requests import get
from web_parser import parser
from dotenv import load_dotenv
from os import getenv

# Load Enviroment Variables
load_dotenv()


# Get search engine ID (Programmable Search Engine)
SEARCH_ENGINE_ID=getenv("SEARCH_ENGINE_ID")
API_KEY=getenv("API_KEY")
google_api_url="https://www.googleapis.com/customsearch/v1"


# Load Question Answer Model from HuggingFace
QA_MODEL=BertForQuestionAnswering.from_pretrained('deepset/bert-base-cased-squad2')
tokenizer=AutoTokenizer.from_pretrained('deepset/bert-base-cased-squad2')

# Create Pipeline for QA model
QA_PIPELINE=pipeline("question-answering",model=QA_MODEL,tokenizer=tokenizer)

# Create Pipeline for Text Classification
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

# Method for getting output message
def getResponse(text):
    detected_intent=None
    answer_found=False
    conversation="User : "
    usr_inp=text
    conversation+=usr_inp+"\n"
    result=pipe(usr_inp)[0]

    # Check if Classified intent is 'greet' or 'goodbye'
    if result['score']>0.5: # 0.5 is confidence level
        detected_intent=result['label']
        if detected_intent in intents.keys():
            res=choice(intents[detected_intent])
            conversation+=f"Assistant : {res}"
            answer_found=True
            return res
        
    # Check if Answer availble in previous conversation
    else:
        with open("page_content.txt",'r') as f:  # Open Page_content file 
            QA_input={
                'question':usr_inp,
                'context': f.read()
            }
            qa_res=QA_PIPELINE(QA_input) # Try to find answer from Page_content file
            if qa_res['score']>0.1:  # If confidence is > 0.1
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

            response=get(google_api_url,params=params)  # Make GET request on search engine
            results=response.json()['items']  # retrive first search item
            if len(results)>0: # if search item present
                link=results[0]['link']
                final_text=parser(link)   # parse the output, remove html tags

                with open("page_content.txt",'a') as f:  # append text to file for future conversation
                    f.write(f"\n\n{usr_inp}\n")
                    f.write(final_text+"\n\n")
                
                QA_input={
                    'question':usr_inp,
                    'context': final_text
                }
                qa_res=QA_PIPELINE(QA_input)
                if qa_res['score']>0.1:  # if confidence score is > 0.1 return answer
                    conversation+=f"Assistant : {qa_res['answer']}"
                    return qa_res['answer']
                else: # else return any one of this answers
                    return choice(['sorry! can\'t answer that','I don\'t know','No idea about that','Don\'t know!'])
            else: # else return one random answer from list
                return choice(['sorry! can\'t answer that','I don\'t know','No idea about that','Don\'t know!'])
    return choice(['sorry! can\'t answer that','I don\'t know','No idea about that','Don\'t know!'])        
    