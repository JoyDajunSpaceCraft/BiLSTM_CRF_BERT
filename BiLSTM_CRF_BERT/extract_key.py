import json
from nltk.stem import WordNetLemmatizer
import operator
import threading
from sklearn.metrics.pairwise import cosine_similarity as simi
from nltk.stem import PorterStemmer
import re
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk
# nltk.download('stopwords')
# pip3 install git+https://github.com/boudinfl/pke.git
# pip3 install -U spacy
# python3 -m spacy download en
# pip3 install scispacy
# pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.4.0/en_core_sci_sm-0.4.0.tar.gz
# pip3 install pyfasttext
# pip3 install sklearn
# [all semantic types] https://metamap.nlm.nih.gov/Docs/SemanticTypes_2018AB.txt
# 27 GB - Model downloaded - https://ftp.ncbi.nlm.nih.gov/pub/lu/Suppl/BioSentVec/BioWordVec_PubMed_MIMICIII_d200.bin
# model = FastText('/home/khushboo/keyphrase/data/BioWordVec_PubMed_MIMICIII_d200.bin')
# model = FastText('/ihome/hdaqing/kmt81/bin/HELPeR/data/bioembed/BioWordVec_PubMed_MIMICIII_d200.bin')
print("load model")


class Stemmer(threading.local):

    def __init__(self):
        # wn.ensure_loaded()
        self.stem = PorterStemmer().stem

stemmer = Stemmer()

def stem(text):
    word_list = text.split(" ")
    for i in range(len(word_list)):
        word_list[i] = stemmer.stem(word_list[i])

    return ' '.join(word_list)


def preprocessText(text,stemming=False, lower=False):

    text = text.replace("\n"," ")
    text = re.sub("[ ]{1,}",r' ',text)

    text = re.sub(r'\W+|\d+', ' ', text.strip())
    tokens = word_tokenize(text)
    tokens = [token.strip().lower() for token in tokens ]
    if lower:
        text = text.lower()
    if stemming:
        tokens = [stem(token.strip()) for token in tokens]

    return " ".join(tokens)


def keyphrase_sim(model,text, keyphrase_list=[]):
    concept_weight = {}
    text = text.lower()
    text = preprocessText(text)
    vec = model.get_numpy_text_vector(text)

    for token in keyphrase_list:

        sim2 = simi(vec.reshape(-1, 200), model.get_numpy_vector(token.lower()).reshape(-1, 200))
        concept_weight[token] = sim2[0][0]
    return concept_weight



lemmatizer = WordNetLemmatizer()

semantic_types_allowed = ['T091', 'T122', 'T019', 'T200', 'T060', 'T203', 'T047', 'T045', 'T028',
                          'T093', 'T059', 'T034', 'T063', 'T114', 'T042', 'T046', 'T121', 'T184', 'T005', 'T127','T061',
                         'T023']


def extract_scispacy(n, model, nlp, text=None):
    concept_list = []
    concept_def = {}
    concept_list_process=[]
    print("sentence divided")
    doc = nlp(text)
    linker = nlp.get_pipe("scispacy_linker")
    for entity in doc.ents:
        if entity._.kb_ents is not None and len(entity._.kb_ents) > 0:
            umls_ent = entity._.kb_ents[0]
            score = umls_ent[1]
            name = str(entity).strip()
            process_name = preprocessText(name,lower=True,stemming=True)
            cui = linker.kb.cui_to_entity[umls_ent[0]]
            #and cui[3][0] in semantic_types_allowed
            if len(name) > 2 and score > 0.8:
                if process_name not in concept_list_process:
                    concept_list.append(name)
                    concept_list_process.append(process_name)
                    concept_def[name] = cui[4]
    print("concept extracted")
    concept_dict = keyphrase_sim(model, text, concept_list)
    print("concept similarity calculated")
    trunc_concept = list(sorted(concept_dict.items(), key=operator.itemgetter(1), reverse=True)[:n+1])
    print("sort and sent the coccepts")
    return trunc_concept, concept_def



import pke

# extracting keywords using topic rank
def extract_Keys_tr(txt,topn=25):
    # Topic Rank Settings
    extractor = pke.unsupervised.TopicRank()
    extractor.load_document(input=txt, language="en", normalization='none')
    extractor.candidate_selection(pos={'NOUN', 'PROPN', 'ADJ'})
    extractor.candidate_weighting(threshold=0.74, method='average')

    results = {}
    result = {}

    for (keyphrase, score) in extractor.get_n_best(n=topn, stemming=False):

        # These are all the keyphrases without any filteration / This might be very noisy and contains many nonn-words

        # original key-phrase
        result['orig_key'] = keyphrase

        # key-phrase description
        # result['desc']= We need a way to provide the description / Either here or in realime

        # lemmarize key-phrase
        result['lem_key'] = lemmatizer.lemmatize(keyphrase)

        # similarity score
        result['score'] = score

        # creating snippet
        results[keyphrase] = result.copy()

    return results

def extract_Keys(txt,model,nlp,topn=25):
    results = {}
    keys, defs = extract_scispacy(n=topn,model= model,nlp=nlp,text=txt)
    for (keyphrase, score) in keys:
        if defs[keyphrase] != None:
            result = {}
            # original key-phrase
            result['orig_key'] = keyphrase
            # keyphrase description
            result['desc'] = defs[keyphrase]

            # lemmarize key-phrase
            result['lem_key'] = lemmatizer.lemmatize(keyphrase)

            # similarity score
            result['score'] = str(score)
            # creating snippet
            results[keyphrase] = result.copy()
    return results



if __name__ == '__main__':
    print("hi")
