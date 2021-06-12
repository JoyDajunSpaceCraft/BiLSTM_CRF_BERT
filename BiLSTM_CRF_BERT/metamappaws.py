from pymetamap import MetaMap
from pymetamap.Concept import ConceptMMI
import lib.nlp as nl
import unicodedata as ud
import re
import pandas as pd
import json
import pandas as pd
from itertools import chain
from elasticsearch import Elasticsearch
# from pyfasttext import FastText
import fasttext


mm = MetaMap.get_instance('data/metamap/public_mm/bin/metamap18')

def converascii(s):
    n = ud.normalize('NFD', s)
    ns = re.sub(r'[^\x00-\x7f]', r' ', n)
    return ns


def getTriggeredWordsNLM(sentence):
    # sentstemp =nl.sent_tokenize(str(sentence))
    # sents = [ converascii(sent) for sent in sentstemp]
    sents=[sentence]
    trigger_wordlist=[]
    metamap_cui = []
    metamap_umls=[]
    present_umls_terms=[]
    concepts,error = mm.extract_concepts(sents,list(range(len(sents))),composite_phrase=3,restrict_to_chv=False,word_sense_disambiguation=True)
    ngramlist=[]
    for sent in sents:
        ngramlisttemp = nl.getNgramList(sent.lower(),5)
        ngramlist += ngramlisttemp

    # print(ngramlist)
    for concept in concepts:
        # print(concept)
        if isinstance(concept,ConceptMMI):
            metamap_cui.append(concept.cui)
            metamap_umls.append((concept.preferred_name,concept.score))

            print(concept)
            trigger_words = [trigger.replace("\"", "").split("-tx")[0].lower() for trigger in
                             concept.trigger.replace("[", "").split(",")]
            trigger_words = list(set(trigger_words))
            # print(trigger_word.lower())
            # print(sents[0].lower())
            for trigger_word in trigger_words:
                if trigger_word.strip() in ngramlist:
                    # print("--",trigger_word.lower())
                    trigger_wordlist.append(trigger_word)
            if concept.preferred_name.strip().lower() in ngramlist:
                present_umls_terms.append(concept.preferred_name)
        # else:
        #     print(concept)



    trigger_wordlist = list(set(trigger_wordlist))
    metamap_cui = list(set(metamap_cui))
    metamap_umls = list(set(metamap_umls))
    present_umls_terms = list(set(present_umls_terms))

    return trigger_wordlist,metamap_cui,metamap_umls,present_umls_terms

def getTriggeredWords(sentence):

    # sentstemp =nl.sent_tokenize(str(sentence))
    # sents = [ converascii(sent) for sent in sentstemp]
    sents=[sentence]
    trigger_wordlist=[]
    metamap_cui = []
    metamap_umls=[]
    present_umls_terms=[]
    concepts,error = mm.extract_concepts(sents,list(range(len(sents))),composite_phrase=3,word_sense_disambiguation=True)
    ngramlist=[]
    for sent in sents:
        ngramlisttemp = nl.getNgramList(sent.lower(),5)
        ngramlist += ngramlisttemp

    # print(ngramlist)
    for concept in concepts:
        print(concept)
        if isinstance(concept,ConceptMMI):
            metamap_cui.append(concept.cui)
            metamap_umls.append((concept.preferred_name,concept.score))

            # print(concept)
            trigger_words = [trigger.replace("\"", "").split("-tx")[0].lower() for trigger in
                             concept.trigger.replace("[", "").split(",")]
            trigger_words = list(set(trigger_words))
            # print(trigger_word.lower())
            # print(sents[0].lower())
            for trigger_word in trigger_words:
                if trigger_word.strip() in ngramlist:
                    # print("--",trigger_word.lower())
                    trigger_wordlist.append(trigger_word)
            if concept.preferred_name.strip().lower() in ngramlist:
                present_umls_terms.append(concept.preferred_name)
        # else:
        #     print(concept)



    trigger_wordlist = list(set(trigger_wordlist))
    metamap_cui = list(set(metamap_cui))
    metamap_umls = list(set(metamap_umls))
    present_umls_terms = list(set(present_umls_terms))

    return trigger_wordlist,metamap_cui,metamap_umls,present_umls_terms,concepts


def getTriggeredWordsScore(text):
    sents = nl.sent_tokenize(text)
    trigger_words_score={}
    concepts,error = mm.extract_concepts(sents,list(range(len(sents))),composite_phrase=5,word_sense_disambiguation=True)
    if error is not None:
        print(error)
        return trigger_words_score

    for concept in concepts:
        # print(concept)
        # print(concept.trigger)
        if isinstance(concept,ConceptMMI):

            trigger_words = [trigger.replace("\"", "").split("-tx")[0].lower() for trigger in
                             concept.trigger.replace("[", "").split(",") ]
            trigger_words = [ x for x in trigger_words if x not in ['i-']]

            trigger_words = list(set(trigger_words))

            for trigger_word in trigger_words:
                if trigger_word.strip() not in trigger_words_score:
                    trigger_words_score[trigger_word.strip()]=0
                trigger_words_score[trigger_word.strip()] = max(trigger_words_score[trigger_word.strip()] ,float(concept.score))

    return trigger_words_score

def getQETerms(term):

    dict_qe={}
    dict_qe['query'] = term
    # dict_qe['medical_terms']=[]
    dict_qe['triggered_words']=[]
    # dict_qe['variations']=[]
    # dict_qe['embedding_based_terms']=[]

    text = term
    qe_list={}
    trigger_wordlist, metamap_cui, metamap_umls, present_umls_terms, concepts = getTriggeredWords(text)
    for concept in concepts:
        if float(concept.score)>3:
            # print(concept)
            cui = concept.cui
            trigger_words = [trigger.replace("\"", "").split("-tx")[0].lower() for trigger in
                             concept.trigger.replace("[", "").split(",")]
            # print(trigger_words)
            # qe_list += trigger_words

            for triggerword in trigger_words:
                if triggerword not in qe_list:
                    print(concept.semtypes)
                    qe_list[triggerword]= (concept.score,concept.semtypes)
                    print(triggerword)

    # dict_qe['variations'] = list(set(qe_list))

    # ngrams = nl.getNgramList(nl.preprocessText(term),5)

    # qe_embed_terms = [' '.join(nl.segment_str(nl.preprocessText(word))) for (word, score) in model.most_similar(positive=[term], k=20) if score > 0.8]
    # print(qe_embed_terms)
    # qe_embed_terms = [ term for term in qe_embed_terms if term not in ngrams and search_expand_term(term) > 0]
    # print(qe_embed_terms)
    # dict_qe['embed_expansion_terms'] =list(set(qe_embed_terms))

    return json.dumps(qe_list)

def getQEEmbed(keywords,text):

    dict_qe={}

    vec_query_question_narr =  model.get_numpy_text_vector(text)
    qe_output = model.words_for_vector(vec_query_question_narr, k=30)
    qe_embed_terms = [' '.join(nl.segment_str(nl.preprocessText(word))).strip() for (word, score) in qe_output if score > 0.7 ]
    qe_embed_terms = [ term for term in qe_embed_terms if term not in ngrams and search_expand_term(term) > 0]
    dict_qe['embedding_based_optional_terms_bioembed'] = list(set(qe_embed_terms[:5]))

    # vec_query_question_narr =  model2.get_numpy_text_vector(query)
    # qe_output = model2.words_for_vector(vec_query_question_narr, k=30)
    # qe_embed_terms = [' '.join(nl.segment_str(nl.preprocessText(word))).strip() for (word, score) in qe_output if score > 0.7]
    # qe_embed_terms = [ term for term in qe_embed_terms if term not in ngrams]
    # dict_qe['embedding_based_optional_terms_collection'] = list(set(qe_embed_terms[:5]))


    return json.dumps(dict_qe)



if __name__ == '__main__':
    #row['query'] + " "+ row['narrative']
    # a=getQEEmbed('malaria coronavirus')
    b = getQETerms('the study was aimed at investigating the effects of wearing n95 and surgical facemasks with and without nano-functional treatments on thermophysiological responses and the subjective perception of discomfort. method: five healthy male and five healthy female participants performed intermittent exercise on a treadmill while wearing the protective facemasks in a climate chamber controlled at an air temperature of 25 c and a relative humidity of 70%. four types of facemasks including n95 (3m 8210) and surgical facemasks which were treated with nano-functional materials were used in the study. results: (1) the subjects had significantly lower average heart rates when wearing nano-treated and untreated surgical facemasks than when wearing nano-treated and untreated n95 facemasks. (2) the outer surface temperature of both surgical facemasks was significantly higher than that of both n95 facemasks. on the other hand the microclimate and skin temperatures inside the facemask were significantly lower than those in both n95 facemasks. (3) both surgical facemasks had significantly higher absolute humidity outside the surface than both n95 facemasks. the absolute humidity inside the surgical facemask was significantly lower than that inside both n95 facemasks. (4) both surgical facemasks were rated significantly lower for perception of humidity heat breath resistance and overall discomfort than both n95 facemasks. the ratings for other sensations including feeling unfit tight itchy fatigued odorous and salty that were obtained while the subjects were wearing the surgical facemasks were significantly lower than when the subjects were wearing the n95 facemasks. (5) subjective preference for the nano-treated surgical facemasks was the highest. there was significant differences in preference between the nano-treated and untreated surgical facemasks and between the surgical and n95 facemasks. discussion: we discuss how n95 and surgical facemasks induce significantly different temperature and humidity in the microclimates of the facemasks which have profound influences on heart rate and thermal stress and subjective perception of discomfort. ')
    # print(a)
    print(b)



