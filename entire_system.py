# wikidatabase
import os
import shutil
import json
import spacy 
from utils_db import dict_load_json
from doc_results import Claim
from claim_database_stage_2_3 import get_word_tag_list_from_text
from claim_database_stage_2_3 import hypothesis_evidence_2_index
from wiki_database import WikiDatabaseSqlite
import pickle
import torch
import config
from esim.model import ESIM
import pickle

from utils_doc_results_db import get_tag_2_id_dict_unigrams

path_wiki_pages = os.path.join(config.ROOT, config.DATA_DIR, config.WIKI_PAGES_DIR, 'wiki-pages')
path_wiki_database_dir = os.path.join(config.ROOT, config.DATA_DIR, config.DATABASE_DIR)

wiki_database = WikiDatabaseSqlite(path_wiki_database_dir, path_wiki_pages)

nlp = spacy.load('en', disable=["parser", "ner"])


import six

def check_predicted_evidence_format(instance):
    if 'predicted_evidence' in instance.keys() and len(instance['predicted_evidence']):
        assert all(isinstance(prediction, list)
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page,line) lists"

        assert all(len(prediction) == 2
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page,line) lists"

        assert all(isinstance(prediction[0], six.string_types)
                    for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page<string>,line<int>) lists"

        assert all(isinstance(prediction[1], int)
                   for prediction in instance["predicted_evidence"]), \
            "Predicted evidence must be a list of (page<string>,line<int>) lists"


def is_correct_label(instance):
    return instance["label"].upper() == instance["predicted_label"].upper()


def is_strictly_correct(instance, max_evidence=None):
    #Strict evidence matching is only for NEI class
    check_predicted_evidence_format(instance)

    if instance["label"].upper() != "NOT ENOUGH INFO" and is_correct_label(instance):
        assert 'predicted_evidence' in instance, "Predicted evidence must be provided for strict scoring"

        if max_evidence is None:
            max_evidence = len(instance["predicted_evidence"])


        for evience_group in instance["evidence"]:
            #Filter out the annotation ids. We just want the evidence page and line number
            actual_sentences = [[e[2], e[3]] for e in evience_group]
            #Only return true if an entire group of actual sentences is in the predicted sentences
            if all([actual_sent in instance["predicted_evidence"][:max_evidence] for actual_sent in actual_sentences]):
                return True

    #If the class is NEI, we don't score the evidence retrieval component
    elif instance["label"].upper() == "NOT ENOUGH INFO" and is_correct_label(instance):
        return True

    return False


def evidence_macro_precision(instance, max_evidence=None):
    this_precision = 0.0
    this_precision_hits = 0.0

    if instance["label"].upper() != "NOT ENOUGH INFO":
        all_evi = [[e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None]

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                        instance["predicted_evidence"][:max_evidence]

        for prediction in predicted_evidence:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

        return (this_precision / this_precision_hits) if this_precision_hits > 0 else 1.0, 1.0

    return 0.0, 0.0

def evidence_macro_recall(instance, max_evidence=None):
    # We only want to score F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        # If there's no evidence to predict, return 1
        if len(instance["evidence"]) == 0 or all([len(eg) == 0 for eg in instance]):
           return 1.0, 1.0

        predicted_evidence = instance["predicted_evidence"] if max_evidence is None else \
                                                                        instance["predicted_evidence"][:max_evidence]

        for evidence_group in instance["evidence"]:
            evidence = [[e[2], e[3]] for e in evidence_group]
            if all([item in predicted_evidence for item in evidence]):
                # We only want to score complete groups of evidence. Incomplete groups are worthless.
                return 1.0, 1.0
        return 0.0, 1.0
    return 0.0, 0.0


# Micro is not used. This code is just included to demostrate our model of macro/micro
def evidence_micro_precision(instance):
    this_precision = 0
    this_precision_hits = 0

    # We only want to score Macro F1/Precision/Recall of recalled evidence for NEI claims
    if instance["label"].upper() != "NOT ENOUGH INFO":
        all_evi = [[e[2], e[3]] for eg in instance["evidence"] for e in eg if e[3] is not None]

        for prediction in instance["predicted_evidence"]:
            if prediction in all_evi:
                this_precision += 1.0
            this_precision_hits += 1.0

    return this_precision, this_precision_hits


def fever_score(predictions,actual=None, max_evidence=5):
    correct = 0
    strict = 0

    macro_precision = 0
    macro_precision_hits = 0

    macro_recall = 0
    macro_recall_hits = 0

    for idx,instance in enumerate(predictions):
        assert 'predicted_evidence' in instance.keys(), 'evidence must be provided for the prediction'

        #If it's a blind test set, we need to copy in the values from the actual data
        if 'evidence' not in instance or 'label' not in instance:
            assert actual is not None, 'in blind evaluation mode, actual data must be provided'
            assert len(actual) == len(predictions), 'actual data and predicted data length must match'
            assert 'evidence' in actual[idx].keys(), 'evidence must be provided for the actual evidence'
            instance['evidence'] = actual[idx]['evidence']
            instance['label'] = actual[idx]['label']

        assert 'evidence' in instance.keys(), 'gold evidence must be provided'

        if is_correct_label(instance):
            correct += 1.0

            if is_strictly_correct(instance, max_evidence):
                strict+=1.0

        macro_prec = evidence_macro_precision(instance, max_evidence)
        macro_precision += macro_prec[0]
        macro_precision_hits += macro_prec[1]

        macro_rec = evidence_macro_recall(instance, max_evidence)
        macro_recall += macro_rec[0]
        macro_recall_hits += macro_rec[1]

    total = len(predictions)

    strict_score = strict / total
    acc_score = correct / total

    pr = (macro_precision / macro_precision_hits) if macro_precision_hits > 0 else 1.0
    rec = (macro_recall / macro_recall_hits) if macro_recall_hits > 0 else 0.0

    f1 = 2.0 * pr * rec / (pr + rec)

    return strict_score, acc_score, pr, rec, f1

from utils_db import dict_load_json, dict_save_json
from utils_db import get_file_name_from_variable_list
from tqdm import tqdm

class EntireSystem():
    
    def __init__(self, wiki_database, nlp, path_stage_2_model, path_stage_3_model,
                 path_dir_doc_selected, method_tokenization, path_base_dir,
                 path_word_dict_stage_2, path_word_dict_stage_3,
                 embeddings_settings_sentence_retrieval_list = [],
                 embeddings_settings_label_prediction_list = []):
        # === process inputs === #
        self.path_stage_2_model = path_stage_2_model
        self.path_stage_3_model = path_stage_3_model
        self.nlp = nlp
        self.path_dir_doc_selected = path_dir_doc_selected
        self.method_tokenization = method_tokenization
        self.path_base_dir = path_base_dir
        self.path_word_dict_stage_2 = path_word_dict_stage_2
        self.path_word_dict_stage_3 = path_word_dict_stage_3
        self.embeddings_settings_sentence_retrieval_list = embeddings_settings_sentence_retrieval_list
        self.embeddings_settings_label_prediction_list = embeddings_settings_label_prediction_list
        
        # === paths === #
        self.path_document_retrieval_dir = os.path.join(path_base_dir, get_file_name_from_variable_list(['document_retrieval']))
        self.path_sentence_retrieval_dir = os.path.join(path_base_dir, 'sentence_retrieval')
        self.path_label_prediction_dir = os.path.join(path_base_dir, 'label_prediction')
        
        for embeddings_setting in embeddings_settings_sentence_retrieval_list:
            self.path_sentence_retrieval_dir = get_file_name_from_variable_list([self.path_sentence_retrieval_dir, embeddings_setting])

        for embeddings_setting in embeddings_settings_label_prediction_list:
            self.path_label_prediction_dir = get_file_name_from_variable_list([self.path_label_prediction_dir, embeddings_setting])
        
        if not os.path.isdir(self.path_base_dir):
            os.makedirs(self.path_base_dir)
                       
        self.path_settings = os.path.join(self.path_base_dir, 'settings.json')
        
        if os.path.isfile(self.path_settings):
            self.settings = dict_load_json(self.path_settings)
        else:
            self.settings = {}
        
        if 'nr_claims' not in self.settings:
            self.settings['nr_claims'] = self.nr_files_in_dir(self.path_dir_doc_selected)
            self.save_settings()
            
        self.nr_claims = self.settings['nr_claims']
        self.nr_claims = 19998
#         self.nr_claims = 100
        print('nr claims:', self.nr_claims)
        
        # === process === #
        self.tag_2_id_dict = get_tag_2_id_dict_unigrams()
    
        if not os.path.isdir(self.path_document_retrieval_dir):
            os.makedirs(self.path_document_retrieval_dir)
            self.document_retrieval()
        
        if not os.path.isdir(self.path_sentence_retrieval_dir):
            os.makedirs(self.path_sentence_retrieval_dir)
            self.sentence_retrieval()
            
        if not os.path.isdir(self.path_label_prediction_dir):
            os.makedirs(self.path_label_prediction_dir)
            self.label_prediction()
            
        self.compute_score()
        
    def compute_score(self):
        # STAGE 2
        # F1
        # PRECISION
        # RECALL
        
        # STAGE 3
        # FEVER
        
        list_claims = []
        for claim_nr in tqdm(range(self.nr_claims)):
            path_claim = os.path.join(self.path_label_prediction_dir, str(claim_nr) + '.json')
            claim_dict = dict_load_json(path_claim)
            list_claims.append(claim_dict)
        
        strict_score, acc_score, pr, rec, f1 = fever_score(predictions=list_claims, actual=None, max_evidence=5)
        
        print(strict_score, acc_score, pr, rec, f1)
        self.settings['score_metrics'] = {}
        self.settings['score_metrics']['strict_score'] = strict_score
        self.settings['score_metrics']['acc_score'] = acc_score
        self.settings['score_metrics']['pr'] = pr
        self.settings['score_metrics']['rec'] = rec
        self.settings['score_metrics']['f1'] = f1
        
        self.save_settings()
        
    def label_prediction(self):
        print('- label prediction: initialise')
        word_dict = pickle.load( open( self.path_word_dict_stage_3, "rb" ) )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(self.path_stage_3_model)

        vocab_size = checkpoint["model"]["_word_embedding.weight"].size(0)
        embedding_dim = checkpoint["model"]['_word_embedding.weight'].size(1)
        hidden_size = checkpoint["model"]["_projection.0.weight"].size(0)
        num_classes = checkpoint["model"]["_classification.4.weight"].size(0)
        
        use_oov_flag=0
        if 'oov' in self.embeddings_settings_label_prediction_list:
            use_oov_flag=1
            
        use_pos_tag_flag=0
        if 'pos' in self.embeddings_settings_label_prediction_list:
            use_pos_tag_flag=1
            
        model = ESIM(vocab_size,
                     embedding_dim,
                     hidden_size,
                     num_classes=num_classes,
                     use_pos_tag_flag=use_pos_tag_flag,
                     use_oov_flag=use_oov_flag,
                     device=device).to(device)

        model.load_state_dict(checkpoint["model"])

        model.eval()
        
        print('- label prediction: iterate through claims')
        for claim_nr in tqdm(range(self.nr_claims)):
            path_claim = os.path.join(self.path_sentence_retrieval_dir, str(claim_nr) + '.json')
            claim_dict = dict_load_json(path_claim)
            
            prob_list  = []
            prob_list_supported = []
            prob_list_refuted = []
            for i in range(len(claim_dict['sentence_retrieval']['doc_nr_list'])):
                doc_nr = claim_dict['sentence_retrieval']['doc_nr_list'][i]
                line_nr = claim_dict['sentence_retrieval']['line_nr_list'][i]
                if doc_nr in claim_dict['document_retrieval']:
                    if line_nr in claim_dict['document_retrieval'][doc_nr]:
                        prob = compute_prob_stage_3(model, claim_dict, doc_nr, line_nr, device)
                        prob_list.append(prob)
                        prob_list_supported.append(prob[2])
                        prob_list_refuted.append(prob[1])
                    else:
                        print('line_nr not in list', line_nr)
                else:
                    print('doc_nr not in list', doc_nr)
            if max(prob_list_supported) > 0.5:
                claim_dict['predicted_label'] = 'SUPPORTS'
            elif max(prob_list_refuted) > 0.5:
                claim_dict['predicted_label'] = 'REFUTES'
            else:
                claim_dict['predicted_label'] = 'NOT ENOUGH INFO'
            
            path_save = os.path.join(self.path_label_prediction_dir, str(claim_nr) + '.json')
            self.save_dict(claim_dict, path_save)
                
    def sentence_retrieval(self):
        print('- sentence retrieval: initialise')
        word_dict = pickle.load( open( self.path_word_dict_stage_2, "rb" ) )
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(self.path_stage_2_model)

        vocab_size = checkpoint["model"]["_word_embedding.weight"].size(0)
        embedding_dim = checkpoint["model"]['_word_embedding.weight'].size(1)
        hidden_size = checkpoint["model"]["_projection.0.weight"].size(0)
        num_classes = checkpoint["model"]["_classification.4.weight"].size(0)
        
        use_oov_flag=0
        if 'oov' in self.embeddings_settings_sentence_retrieval_list:
            use_oov_flag=1
            
        use_pos_tag_flag=0
        if 'pos' in self.embeddings_settings_sentence_retrieval_list:
            use_pos_tag_flag=1
            
        model = ESIM(vocab_size,
                     embedding_dim,
                     hidden_size,
                     num_classes=num_classes,
                     use_pos_tag_flag=use_pos_tag_flag,
                     use_oov_flag=use_oov_flag,
                     device=device).to(device)

        model.load_state_dict(checkpoint["model"])

        model.eval()
        
        print('- sentence retrieval: iterate through claims')
        for claim_nr in tqdm(range(self.nr_claims)):
            path_claim = os.path.join(self.path_document_retrieval_dir, str(claim_nr) + '.json')
            claim_dict = dict_load_json(path_claim)
            
            list_prob = []
            list_doc_nr = []
            list_line_nr = []
        
            for doc_nr in claim_dict['document_retrieval']:
                for line_nr in claim_dict['document_retrieval'][doc_nr]:
                    if 'sentence_retrieval' not in claim_dict:
                        claim_dict['sentence_retrieval'] = {}
                    if doc_nr not in claim_dict['sentence_retrieval']:
                        claim_dict['sentence_retrieval'][doc_nr] = {}
                    if line_nr not in claim_dict['sentence_retrieval'][doc_nr]:
                        claim_dict['sentence_retrieval'][doc_nr][line_nr] = {}
                    
                    prob = compute_prob_stage_2(model, claim_dict, doc_nr, line_nr, device)
                    claim_dict['sentence_retrieval'][doc_nr][line_nr]['prob'] = prob
                    
                    list_doc_nr.append(doc_nr)
                    list_line_nr.append(line_nr)
                    list_prob.append(prob)
                    
            sorted_list_doc_nr = sort_list(list_doc_nr, list_prob)[-5:]
            sorted_list_line_nr = sort_list(list_line_nr, list_prob)[-5:]
            sorted_list_prob = sort_list(list_prob, list_prob)[-5:]
            claim_dict['sentence_retrieval']['doc_nr_list'] = sorted_list_doc_nr  
            claim_dict['sentence_retrieval']['line_nr_list'] = sorted_list_line_nr  
            claim_dict['sentence_retrieval']['prob_list'] = sorted_list_prob 
            
            claim_dict['predicted_evidence'] = []
            for i in range(len(sorted_list_doc_nr)):
                doc_nr = sorted_list_doc_nr[i]
                title = wiki_database.get_title_from_id(int(doc_nr))
                line_nr = int(sorted_list_line_nr[i])
                claim_dict['predicted_evidence'].append([title, line_nr])                                   
            
            path_save = os.path.join(self.path_sentence_retrieval_dir, str(claim_nr) + '.json')
            self.save_dict(claim_dict, path_save)
                    
        
    def document_retrieval(self):
#         claim_nr = 12
#         line_nr = 0
#         nr_in_doc_selected_list = 0
        word_dict = pickle.load( open( self.path_word_dict_stage_3, "rb" ) )
    
        for claim_nr in tqdm(range(self.nr_claims)):
            path_claim = os.path.join(self.path_dir_doc_selected, str(claim_nr) + '.json')
            claim_dict = dict_load_json(path_claim)
            claim = Claim(claim_dict)
            claim_text = claim.claim
            # === process word tags and word list === #
            tag_list_claim, word_list_claim = get_word_tag_list_from_text(text_str = claim_text, 
                                                                          nlp = nlp, 
                                                                          method_tokenization_str = method_tokenization)

            for doc_nr in claim_dict['docs_selected']:
                line_list = wiki_database.get_lines_from_id(doc_nr)
                nr_lines = len(line_list)
                for line_nr in range(nr_lines):
                    line_text = line_list[line_nr]

                    # === process word tags and word list === #
                    tag_list_line, word_list_line = get_word_tag_list_from_text(text_str = line_text, 
                                                                                nlp = nlp, 
                                                                                method_tokenization_str = method_tokenization)
                    
                    if 'document_retrieval' not in claim_dict:
                        claim_dict['document_retrieval'] = {}
                    if str(doc_nr) not in claim_dict['document_retrieval']:
                        claim_dict['document_retrieval'][str(doc_nr)] = {}

                    if str(line_nr) not in claim_dict['document_retrieval'][str(doc_nr)]:
                        claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)] = {}

                    if 'claim' not in claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]:
                        claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['claim'] = {}

                    if 'document' not in claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]:
                        claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['document'] = {}

                    claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['claim']['tag_list'] = [17] + tag_str_2_id_list(
                        tag_list_claim, self.tag_2_id_dict) + [17]
                    claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['claim']['word_list'] = word_list_2_id_list(
                        ["_BOS_"] + word_list_claim + ["_EOS_"], word_dict)
                    claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['document']['tag_list'] = [17] + tag_str_2_id_list(
                        tag_list_line, self.tag_2_id_dict) + [17]
                    claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['document']['word_list'] = word_list_2_id_list(
                        ["_BOS_"] + word_list_line + ["_EOS_"], word_dict)

                    ids_document, ids_claim = hypothesis_evidence_2_index(hypothesis = word_list_line,
                                               premise = word_list_claim,
                                               randomise_flag = False)

                    claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['claim']['exact_match_list'] = [0] + ids_claim + [1]
                    claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['document']['exact_match_list'] = [0] + ids_document + [1]

            path_save = os.path.join(self.path_document_retrieval_dir, str(claim_nr) + '.json')
            self.save_dict(claim_dict, path_save)

    def save_settings(self):
        dict_save_json(self.settings, self.path_settings)
    
    def save_dict(self, input_dict, path):
        dict_save_json(input_dict, path)
        
    def load_dict(self, path):
        return dict_load_json(path)
    
    def nr_files_in_dir(self, path_dir):
        list_files = os.listdir(path_dir) # dir is your directory path
        number_files = len(list_files)
        return number_files

def sort_list(list1, list2): 
    zipped_pairs = zip(list2, list1) 
    z = [x for _, x in sorted(zipped_pairs)] 
    return z 

def tag_str_2_id_list(tag_str_list, tag_2_id_dict):
    return [tag_2_id_dict[tag] for tag in tag_str_list]

def word_list_2_id_list(word_list, word_dict):
    id_list = []
    for word in word_list:
        if word in word_dict:
            id_list.append(word_dict[word])
        else:
#             print(word)
            id_list.append(word_dict['_OOV_'])    
    return id_list

def compute_prob_stage_2(model, claim_dict, doc_nr, line_nr, device):
    claims_word_list = claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['claim']['word_list']
    hypothesis_word_list = claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['document']['word_list']
    claim_pos = claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['claim']['tag_list']
    hypothesis_pos = claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['document']['tag_list']
    claim_oov = claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['claim']['exact_match_list']
    hypothesis_oov = claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['document']['exact_match_list']
    
    premises = torch.tensor([claims_word_list]).to(device)
    premises_lengths = torch.tensor([len(claims_word_list)]).to(device)
    hypotheses = torch.tensor([hypothesis_word_list]).to(device)
    hypotheses_lengths = torch.tensor([len(hypothesis_word_list)]).to(device)
    premises_part_of_speech = torch.tensor([claim_pos]).to(device)
    premises_part_of_speech_lengths = torch.tensor([len(claim_pos)]).to(device)
    premises_out_of_vocabulary = torch.tensor([claim_oov]).to(device)
    premises_out_of_vocabulary_lengths = torch.tensor([len(claim_oov)]).to(device)
    hypotheses_part_of_speech = torch.tensor([hypothesis_pos]).to(device)
    hypotheses_part_of_speech_lengths = torch.tensor([len(hypothesis_pos)]).to(device)
    hypotheses_out_of_vocabulary = torch.tensor([hypothesis_oov]).to(device)
    hypotheses_out_of_vocabulary_lengths = torch.tensor([len(hypothesis_oov)]).to(device)
    
    _, probs = model(premises,
                    premises_lengths,
                    hypotheses,
                    hypotheses_lengths, 
                    premises_part_of_speech, 
                    premises_part_of_speech_lengths,
                    premises_out_of_vocabulary,
                    premises_out_of_vocabulary_lengths,
                    hypotheses_part_of_speech,
                    hypotheses_part_of_speech_lengths,
                    hypotheses_out_of_vocabulary,
                    hypotheses_out_of_vocabulary_lengths)
    return float(probs[0][1])

def compute_prob_stage_3(model, claim_dict, doc_nr, line_nr, device):
    claims_word_list = claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['claim']['word_list']
    hypothesis_word_list = claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['document']['word_list']
    claim_pos = claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['claim']['tag_list']
    hypothesis_pos = claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['document']['tag_list']
    claim_oov = claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['claim']['exact_match_list']
    hypothesis_oov = claim_dict['document_retrieval'][str(doc_nr)][str(line_nr)]['document']['exact_match_list']
    
    premises = torch.tensor([claims_word_list]).to(device)
    premises_lengths = torch.tensor([len(claims_word_list)]).to(device)
    hypotheses = torch.tensor([hypothesis_word_list]).to(device)
    hypotheses_lengths = torch.tensor([len(hypothesis_word_list)]).to(device)
    premises_part_of_speech = torch.tensor([claim_pos]).to(device)
    premises_part_of_speech_lengths = torch.tensor([len(claim_pos)]).to(device)
    premises_out_of_vocabulary = torch.tensor([claim_oov]).to(device)
    premises_out_of_vocabulary_lengths = torch.tensor([len(claim_oov)]).to(device)
    hypotheses_part_of_speech = torch.tensor([hypothesis_pos]).to(device)
    hypotheses_part_of_speech_lengths = torch.tensor([len(hypothesis_pos)]).to(device)
    hypotheses_out_of_vocabulary = torch.tensor([hypothesis_oov]).to(device)
    hypotheses_out_of_vocabulary_lengths = torch.tensor([len(hypothesis_oov)]).to(device)
    
    _, probs = model(premises,
                    premises_lengths,
                    hypotheses,
                    hypotheses_lengths, 
                    premises_part_of_speech, 
                    premises_part_of_speech_lengths,
                    premises_out_of_vocabulary,
                    premises_out_of_vocabulary_lengths,
                    hypotheses_part_of_speech,
                    hypotheses_part_of_speech_lengths,
                    hypotheses_out_of_vocabulary,
                    hypotheses_out_of_vocabulary_lengths)
    return [float(score) for score in probs[0]]


nr_selected_documents = 5
path_dir_doc_selected = '/mnt/01_thesis/01_code/fever/_04_results/vocab_title_1_t/thr_0.01_/ex_term_frequency_inverse_document_frequency_title/results_experiment_dev_tf_idf_normalise_' + str(nr_selected_documents)  + '/claims_dev'
method_tokenization = 'tokenize_text_pos'
path_base_dir = 'tmp_' + str(nr_selected_documents) + '_pos_oov'
path_word_dict_stage_2 = os.path.join('/mnt/01_thesis/01_code/ESIM/data/preprocessed/FEVER_stage_2', 'worddict.pkl')
path_word_dict_stage_3 = os.path.join('/mnt/01_thesis/01_code/ESIM/data/preprocessed/FEVER_stage_3', 'worddict.pkl')

path_stage_2_model = '/mnt/01_thesis/01_code/ESIM/data/checkpoints/FEVER_stage_2_pos_oov/best.pth.tar'
path_stage_3_model = '/mnt/01_thesis/01_code/ESIM/data/checkpoints/FEVER_stage_3_pos_oov/best.pth.tar'

embeddings_settings_sentence_retrieval_list = ['oov','pos']
embeddings_settings_label_prediction_list = ['oov','pos']

entire_system = EntireSystem(wiki_database, nlp, path_stage_2_model, path_stage_3_model, 
                 path_dir_doc_selected, method_tokenization, path_base_dir,
                 path_word_dict_stage_2, path_word_dict_stage_3,
                 embeddings_settings_sentence_retrieval_list = embeddings_settings_sentence_retrieval_list,
                 embeddings_settings_label_prediction_list = embeddings_settings_label_prediction_list)
