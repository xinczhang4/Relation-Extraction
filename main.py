import itertools
import math, string
import sys
import openid
import openai
import requests
import re
import spacy
from spacy_help_functions import extract_relations, create_entity_pairs
from spanbert import SpanBERT 
from bs4 import BeautifulSoup
import os
import time
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
# -*- coding: utf-8 -*-

# from sklearn.feature_extraction.text import TfidfVectorizer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# import itertools

# import nltk
# from nltk.lm import MLE
# from nltk.lm.preprocessing import padded_everygram_pipeline
# import sys

prompt_text = ["""Given a sentence, extract all relationship pairs of the following relationship types as possible:
relationship type: Schools_Attended
Output: person: PERSON, relationship: RELATIONSHIP, school:SCHOOL
Output each relationship pair in separate lines.
Sample Output: Jeff Bezos, Schools_Attended, Princeton University
Sentence: """,
               """Given a sentence, extract all relationship pairs of the following relationship types as possible:
relationship type: Work_For
Output: person: PERSON, relationship: RELATIONSHIP, company:COMPANY
Output each relationship pair in separate lines.
Example Output: Alec Radford, Work_For, OpenAI
Sentence: """,
               """Given a sentence, extract all relationship pairs of the following relationship types as possible:
relationship type: Live_In
Output: person: PERSON, relationship: RELATIONSHIP, location:LOCATION
Output each relationship pair in separate lines.
Sample Output: Mariah Carey, Live_In, New York City
Sentence: """,

              """Given a sentence, extract all relationship pairs of the following relationship types as possible:
relationship type: Top_Member_Employees
Output: person: PERSON, relationship: RELATIONSHIP, company:COMPANY
Output each relationship pair in separate lines.
Sample Output: Jensen Huang, Top_Member_Employees, Nvidia
Sentence: """]

def get_openai_completion(prompt, model, max_tokens, temperature = 0.2, top_p = 1, frequency_penalty = 0, presence_penalty =0):
    response = openai.Completion.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )
    response_text = response['choices'][0]['text']
    return response_text
def is_available(pair,relation):
    if len(pair) == 3 and pair[1] == relation:
        words = pair[0].split(' ')
        if len(words) <= 6:
            return True
    return False



# set search engine and api keys
search_engine_id = '4bb444072b2573605'
api_key = 'AIzaSyCoZ-0Vllrj5DHiEdjdDtYCD1AkPR2tTOs'
entities_of_interest = ["PERSON", "CITY"]
spanbert = SpanBERT("./pretrained_spanbert")  
# Using spaCy to split sentence and recognize entities
nlp = spacy.load("en_core_web_lg")
# python3 project2.py [-spanbert|-gpt3] <google api key> <google engine id> <openai secret key> <r> <t> <q> <k>
# method = sys.argv[1]
# api_key = sys.argv[2]
# search_engine_id = sys.argv[3]
# openai_secret_key = sys.argv[4]
# relation = sys.argv[5]
# threshold = sys.argv[6]
# seed_query = sys.argv[7] # plausible relation tuple e.g."bill gates microsoft"
# k = sys.argv[8]

relation_pair = {1: 'Schools_Attended', 2: 'Work_For', 3: 'Live_In', 4: 'Top_Member_Employees'}

retry_queries = [] # for when we don't have enough results after extraction

try:
    method = sys.argv[1]
    api_key = sys.argv[2]
    search_engine_id = sys.argv[3]
    # openai_secret_key = sys.argv[4]
    openai.api_key = sys.argv[4]

    relation = int(sys.argv[5])
    threshold = float(sys.argv[6])
    seed_query = " ".join(sys.argv[7: len(sys.argv)-1]) # plausible relation tuple e.g."bill gates microsoft"
    k = int(sys.argv[len(sys.argv)-1])
except Exception as e:
    print(e)
    # print("Intended input: python3 project2.py [-spanbert|-gpt3] <google api key> <google engine id> <openai secret key> <r> <t> <q> <k>")
    sys.exit(1)
if relation == 1:
    label1 = 'PERSON'
    label2 = 'ORG'
    entities_of_interest = ['PERSON', 'ORGANIZATION']
    internal_name = 'per:schools_attended'
elif relation == 2:
    label1 = 'PERSON'
    label2 = 'ORG'
    entities_of_interest = ['PERSON', 'ORGANIZATION']
    internal_name = 'per:employee_of'

elif relation == 3:
    label1 = 'PERSON'
    label2 = 'GPE'
    entities_of_interest = ['PERSON', 'CITY']
    internal_name = 'per:cities_of_residence'


elif relation == 4:
    label1 = 'PERSON'
    label2 = 'ORG'
    entities_of_interest = ['PERSON', 'ORGANIZATION']
    internal_name = 'org:top_members/employees'
query = seed_query.strip('\'\"”“')


print("Parameters:")
print("Client key      = " + api_key)
print("Engine key      = " + search_engine_id)
print("OpenAI key      = " + openai.api_key)
print("Method  = " +  method[1:])
print("Relation        = " + relation_pair[relation])
print("Threshold       = " + str(threshold))
print("Query           = " + query)
print("# of Tuples     = " + str(k))
print("Loading necessary libraries; This should take a minute or so ...)")

print("======================")

if method == '-spanbert':
    done = False
    iterations = 0
    while not done:
        # make search request with api
        response = requests.get(f'https://www.googleapis.com/customsearch/v1?q={query}&cx={search_engine_id}&key={api_key}')

        if response.status_code == 200: # 200 means response was a success
            # print(response.json()['snippet'])
            results = response.json()['items']

            relevancies = [None for result in results]
            
            #initialize extracted tuples X
            X = set()
            X_confidence = {}
            
            URLs = []
            num_extracted = 0
            for i in range(10): # get top 10 URLs, retrieve webpages
                URLs.append(results[i]["link"])
            for i, url in enumerate(URLs):
                print(f'URL ({i + 1}/10): {url}')
                # print(url)
                print("Fetching text from URL...")
                webpage = requests.get(url)
                if webpage.status_code == 200:
                    html = webpage.content
                    
                    # apply BeautifulSoup to handle the content
                    soup = BeautifulSoup(html, 'html.parser')
                    text = soup.find_all(string=True)
                    
                    # remove useless elements
                    output = ''
                    blacklist = [
                        '[document]',
                        'noscript',
                        'header',
                        'html',
                        'meta',
                        'head', 
                        'input',
                        'script',
                        'style',
                    ]

                    for t in text:
                        if t.parent.name not in blacklist:
                            output += '{} '.format(t)
                            
                    #Removing redundant newlines and some whitespace characters

                    preprocessed_text = re.sub(u'\xa0', ' ', output) 
                    preprocessed_text = re.sub('\t+', ' ', preprocessed_text) 
                    preprocessed_text = re.sub('\n+', ' ', preprocessed_text) 
                    preprocessed_text = re.sub(' +', ' ', preprocessed_text) 
                    preprocessed_text = preprocessed_text.replace('\u200b', '')
                    
                    if len(preprocessed_text) > 10000:
                        print(f'Trimming website content from {len(preprocessed_text)} to 10000 characters')
                        preprocessed_text = preprocessed_text[:10000]
                    print(f'Webpage Length (num characters): {len(preprocessed_text)}')
                    
                    print('Annotating the webpage using spacy')
                    doc = nlp(preprocessed_text)
                    print(f'Extracted {len(list(doc.sents))} sentences. Processing each sentence one by one to check for the presence of right pair of named entity types; if so, will run the second pipeline...')
                    ent_constraints = {'PERSON', 'ORG'}
                    num_annotated_here = 0
                    num_extracted_here = 0
                    # Parse the sentence using default dependency parse - the most accurate
                    for i, sentence in enumerate(doc.sents):
                        if i % 5 == 0:
                            print(f'Processed {i} / {len(list(doc.sents))} sentences')
                        sent = nlp(sentence.text)
                        
                        entities = [ent for ent in sent.ents]

                        entity_pairs = []
                        found_pair = False
                        for i, entity1 in enumerate(entities):
                            if found_pair:
                                break
                            for j, entity2 in enumerate(entities):

                                if entity1.label_ == label1 and entity2.label_ == label2:


                                    # print("FOUND PAIR")
                                    found_pair = True
                                    break


                        if found_pair:
                            # print("EXTRACTING")
                            try:
                                relations = extract_relations(sent, spanbert, internal_name, entities_of_interest, threshold)
                                # print("EXTRACTED")
                                if len(relations) > 0:
                                    num_annotated_here += 1
                                    for relation in relations.keys():
                                        if relation[1] == internal_name: 
                                            if relation in X_confidence.keys():
                                                X_confidence[relation] = max(X_confidence[relation], relations[relation])
                                            else:
                                                num_extracted_here += 1
                                                X_confidence[relation] = relations[relation]
                                    continue
                            except IndexError: # for when extract_relations returns an empty dict
                                pass


                num_extracted += num_extracted_here   
                print(f'Extracted annotations for {num_annotated_here} out of total {len(list(doc.sents))} sentences')
                print(f'Relations extracted from this website: {num_extracted_here} (Overall: {num_extracted})')

            sorted_results = sorted([[k, v] for k, v in X_confidence.items()], key=lambda x: x[1], reverse=True)

            if len(sorted_results) >= k:
                sorted_list = sorted_results[:k]
                done = True
                
                print(sorted_results)
                
                print(f'================== ALL RELATIONS FOR {internal_name} ( {len(sorted_results)} ) =================')
                for result in sorted_results:
                    print(f'Confidence: {result[1]}          | Subject: {result[0][0]}          | Object: {result[0][2]}')
                print(f'Total # of iterations = {iterations + 1}')
            elif len(retry_queries) == 0: # not enough relations
                retry_queries = ['{} {} {}'.format(*t[0]) for t in sorted_results] # put tuples together to make new queries
                query = retry_queries[iterations]
            else:
                if iterations > len(retry_queries):
                    print(f"All requeries ran. Exiting program without reaching {k} results.")
                    done = True
                query = retry_queries[iterations]
            iterations += 1
elif method == '-gpt':
    iteration = 0
    done = False

    while not done:
        print('=========== Iteration: {} - Query: {} ==========='.format(iteration, query))
        iteration += 1
        prompt = ""
        # make search request with api
        response = requests.get(f'https://www.googleapis.com/customsearch/v1?q={query}&cx={search_engine_id}&key={api_key}')

        if response.status_code == 200: # 200 means response was a success
            # print(response.json()['snippet'])
            results = response.json()['items']
            relevancies = [None for result in results]

            #initialize extracted tuples X
            X = set()

            URLs = []
            for i in range(10): # get top 10 URLs, retrieve webpages
                URLs.append(results[i]["link"])
            url_cnt = 1
            for url in URLs:
                sent_used = 0
                relation_total = 0
                relation_extracted = 0
                print('\nURL ( {} / 10): {}'.format(url_cnt, url))
                url_cnt += 1
                print('        Fetching text from url ...')
                webpage = requests.get(url)
                if webpage.status_code == 200:
                    html = webpage.content

                    # apply BeautifulSoup to handle the content
                    soup = BeautifulSoup(html, 'html.parser')
                    text = soup.find_all(string=True)

                    # remove useless elements
                    output = ''
                    blacklist = [
                        '[document]',
                        'noscript',
                        'header',
                        'html',
                        'meta',
                        'head', 
                        'input',
                        'script',
                        'style',
                    ]

                    for t in text:
                        if t.parent.name not in blacklist:
                            output += '{} '.format(t)

                    #Removing redundant newlines and some whitespace characters

                    preprocessed_text = re.sub(u'\xa0', ' ', output) 
                    preprocessed_text = re.sub('\t+', ' ', preprocessed_text) 
                    preprocessed_text = re.sub('\n+', ' ', preprocessed_text) 
                    preprocessed_text = re.sub(' +', ' ', preprocessed_text) 
                    preprocessed_text = preprocessed_text.replace('\u200b', '')

                    if len(preprocessed_text) > 10000:
                        print('        Trimming webpage content from {} to 10000 characters'.format(len(preprocessed_text)))
                        preprocessed_text = preprocessed_text[:10000]
                        print('       Webpage length (num characters): '+str(len(preprocessed_text)))

                        # Using spaCy to split sentence and recognize entities
                        print('        Annotating the webpage using spacy...')
                        nlp = spacy.load("en_core_web_lg")
                        doc = nlp(preprocessed_text)
                        total_sent = len(list(doc.sents))
                        print('        Extracted {} sentences. Processing each sentence one by one to check for presence of right pair of named entity types; if so, will run the second pipeline ...'.format(total_sent))
                        # Parse the sentence using default dependency parse - the most accurate
                        # i = 0
                        # for sentence in doc.sents:
                        #     i = i+1
                        #     sent = nlp(sentence.text)
                        #     print(sentence)
                        #     for ent in sent.ents: 
                        #         print(ent.text, ent.label_)
                        # print(i)

                        # Method: Open AI
                        # filter sentences containing the named entities pairs of the right type
                        sent_cnt = 1
                        for sentence in doc.sents:
                            labels = []
                            sent = nlp(sentence.text)
                            for ent in sent.ents:
                                labels.append(ent.label_)
                            # if(len(X)<= 10):
                            if label1 in labels and label2 in labels:
                                sent_used += 1
                                # feed the sentence to GPT-3 API
                                prompt = prompt_text[relation-1] + sentence.text
                                model = 'text-davinci-003'
                                max_tokens = 100
                                temperature = 0.2
                                top_p = 1
                                frequency_penalty = 0
                                presence_penalty = 0

                                response_text = get_openai_completion(prompt, model, max_tokens, temperature, top_p, frequency_penalty, presence_penalty)
                                response_list = response_text.splitlines()
                                for item in response_list:
                                    pair = item.split(', ')
                                    if is_available(pair,relation_pair[relation]):
                                        print('\n                === Extracted Relation ===')
                                        print('                Sentence:  ' + sentence.text)
                                        print('                Subject: {} ; Object: {} ;'.format(pair[0], pair[2]))                           
                                        relation_total += 1
                                        # print(X)
                                        if tuple((pair[0],pair[2])) in X:
                                            print('                Duplicate. Ignoring this.')
                                        else:
                                            X.add(tuple((pair[0],pair[2])))
                                            relation_extracted += 1
                                            print('                Adding to set of extracted relations')
                                        print('                ==========')
                                    time.sleep(1.5)
                            if sent_cnt % 5 == 0:
                                print('        Processed {} / {} sentences'.format(sent_cnt,total_sent))
                            sent_cnt += 1
                    print('\n')
                    print('        Extracted annotations for  {}  out of total  {}  sentences'.format(sent_used, total_sent))
                    print('        Relations extracted from this website: {} (Overall: {})'.format(relation_extracted, relation_total))

                if len(X) >= k:
                    done = True
                    X = list(X)[:k]
                    break
                elif len(retry_queries) == 0: # not enough relations
                    retry_queries = ['{} {} {}'.format(*t[0]) for t in X] # put tuples together to make new queries
                    query = retry_queries[iteration]
                else:
                    if iteration >= len(retry_queries):
                        print(f"All requeries ran. Exiting program without reaching {k} results.")
                        done = True
                        break
                    query = retry_queries[iteration]
                iteration += 1

        print('================== ALL RELATIONS for {} ( {} ) ================='.format(relation_pair[relation], len(X)))
        for item in X:
            print('Subject: {}             | Object: {}'.format(item[0], item[1]))
        print('Total # of iterations = ' + str(iteration))
        
        
        
