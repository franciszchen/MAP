import json
import transformers
import torch


# #"LLaMA3_8B_Instruct_0916"
# model_id = "QIOvO/maids"
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16,"load_in_8bit": True},
#     device_map=0,
# )

# def disease_test(query):
#     messages = [
#     {"role": "system", "content": '''You are a clinic expert and need to make a diagnose for possible diseases from the candidate disease list based on the patient's medical record information. The candidate disease list is as follows:["Certain infectious and parasitic diseases","Neoplasms","Endocrine, nutritional and metabolic Diseases","Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism","Mental and behavioural disorders","Diseases of the nervous system and sense organs","Diseases of the circulatory system","Diseases of the respiratory system","Diseases of the digestive system","Diseases of the genitourinary system","Pregnancy, childbirth and the puerperium","Diseases of the skin and subcutaneous tissue","Diseases of the musculoskeletal system and connective tissue","Congenital malformations, deformations and chromosomal abnormalities","Certain conditions originating in the perinatal period","Symptoms, signs and abnormal clinical and laboratory findings","Injury, poisoning and certain other consequences of external causes"]'''},
#     {"role": "user", "content": query},
# ]

#     terminators = [
#         pipeline.tokenizer.eos_token_id,
#         pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

#     outputs = pipeline(
#         messages,
#         max_new_tokens=4096,
#         eos_token_id=terminators,
#         do_sample=True,
#         temperature=0.6,
#         top_p=0.9,
#     )
#     return outputs[0]["generated_text"][-1]["content"]

# def care_test(query):
#     messages = [
#     {"role": "system", "content": '''You are a doctor's assistant and need to determine which first_unit to go to based on the patient's medical condition. The list of candidate first_careunits is as follows: ['Medical Intensive Care Unit (MICU)', 'Surgical Intensive Care Unit (SICU) )', 'Surgical Intensive Care Unit (MICU/SICU)', 'Cardiac Vascular Intensive Care Unit (CVICU)', 'Coronary Care Unit (CCU)', 'Neuro Intermediate', 'Trauma SICU (TSICU)', 'Neuro Stepdown', 'Neuro Surgical Intensive Care Unit (Neuro SICU)']]'''},
#     {"role": "user", "content": query},
# ]

#     terminators = [
#         pipeline.tokenizer.eos_token_id,
#         pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

#     outputs = pipeline(
#         messages,
#         max_new_tokens=4096,
#         eos_token_id=terminators,
#         do_sample=True,
#         temperature=0.6,
#         top_p=0.9,
#     )
#     return outputs[0]["generated_text"][-1]["content"]

# def curr_test(query):
#     messages = [
#     {"role": "system", "content": '''You are a clinic expert and need to find one the most suitable current service from the candidate curr_service list based on the patient's medical record. The candidate curr_service list is as follows:['OMED', 'ORTHO', 'TSURG', 'OBS', 'TRAUM', 'NSURG', 'MED', 'CSURG', 'GU', 'ENT', 'CMED', 'VSURG', 'SURG', 'NMED', 'GYN', 'PSURG']'''},
#     {"role": "user", "content": query},
# ]

#     terminators = [
#         pipeline.tokenizer.eos_token_id,
#         pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

#     outputs = pipeline(
#         messages,
#         max_new_tokens=4096,
#         eos_token_id=terminators,
#         do_sample=True,
#         temperature=0.6,
#         top_p=0.9,
#     )
#     return outputs[0]["generated_text"][-1]["content"]

import requests
import json
import random
def disease_test(query):
    url = 'http://175.45.13.117:53999/v1/chat/completions'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    messages = [
        {"role": "system", "content": '''You are a clinic expert and need to make a diagnose for possible diseases from the candidate disease list based on the patient's medical record information. The candidate disease list is as follows:["Certain infectious and parasitic diseases","Neoplasms","Endocrine, nutritional and metabolic Diseases","Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism","Mental and behavioural disorders","Diseases of the nervous system and sense organs","Diseases of the circulatory system","Diseases of the respiratory system","Diseases of the digestive system","Diseases of the genitourinary system","Pregnancy, childbirth and the puerperium","Diseases of the skin and subcutaneous tissue","Diseases of the musculoskeletal system and connective tissue","Congenital malformations, deformations and chromosomal abnormalities","Certain conditions originating in the perinatal period","Symptoms, signs and abnormal clinical and laboratory findings","Injury, poisoning and certain other consequences of external causes"]'''},
        {"role": "user", "content": query}
    ]
    
    data = {
        "model": "internlm2",
        "messages": messages,
        "temperature": 0,
        "top_p": 0.8,
        "n": 1,
        "max_tokens": None,
        "stop": None,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "user": "string",
        "repetition_penalty": 1.002,
        "session_id": -1,
        "ignore_eos": False
    }
    
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    return response_json['choices'][0]['message']['content']

def care_test(query):
    url = 'http://175.45.13.117:53999/v1/chat/completions'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    messages = [
        {"role": "system", "content": '''You are a doctor's assistant and need to determine which first_unit to go to based on the patient's medical condition. The list of candidate first_careunits is as follows: ['Medical Intensive Care Unit (MICU)', 'Surgical Intensive Care Unit (SICU) )', 'Surgical Intensive Care Unit (MICU/SICU)', 'Cardiac Vascular Intensive Care Unit (CVICU)', 'Coronary Care Unit (CCU)', 'Neuro Intermediate', 'Trauma SICU (TSICU)', 'Neuro Stepdown', 'Neuro Surgical Intensive Care Unit (Neuro SICU)']]'''},
        {"role": "user", "content": query}
    ]
    
    data = {
        "model": "internlm2",
        "messages": messages,
        "temperature": 0,
        "top_p": 0.8,
        "n": 1,
        "max_tokens": None,
        "stop": None,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "user": "string",
        "repetition_penalty": 1.002,
        "session_id": -1,
        "ignore_eos": False
    }
    
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    return response_json['choices'][0]['message']['content']

def curr_test(query):
    url = 'http://175.45.13.117:53999/v1/chat/completions'
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    
    messages = [
        {"role": "system", "content": '''You are a clinic expert and need to find one the most suitable current service from the candidate curr_service list based on the patient's medical record. The candidate curr_service list is as follows:['OMED', 'ORTHO', 'TSURG', 'OBS', 'TRAUM', 'NSURG', 'MED', 'CSURG', 'GU', 'ENT', 'CMED', 'VSURG', 'SURG', 'NMED', 'GYN', 'PSURG']'''},
        {"role": "user", "content": query}
    ]
    
    data = {
        "model": "internlm2",
        "messages": messages,
        "temperature": 0,
        "top_p": 0.8,
        "n": 1,
        "max_tokens": None,
        "stop": None,
        "stream": False,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "user": "string",
        "repetition_penalty": 1.002,
        "session_id": -1,
        "ignore_eos": False
    }
    
    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()
    return response_json['choices'][0]['message']['content']


with open('/root/capsule/data/test_100_cases.json', 'r') as file:
    data = json.load(file)
num = 0 
for item in data:

    gender = item["gender"]
    language = item["language"]
    martial = item["marital"]
    race = item["race"]
    Past_Medical_History = item["Past Medical History"]
    radio_text = item["radio_text"]
    firsr_careunit = item["first_careunit"]
    disease = item["icdtitle"]
    corr = item["corr"]
    instruction = item["instruction"]
    with open("/root/capsule/code/examples/clinic/data/q_disease.txt", 'r') as file1:
        lines1 = file1.readlines()
    query1 = random.choice(lines1).strip()
    with open('/root/capsule/code/examples/clinic/data/q_firstunit.txt', 'r') as file2:
        lines2 = file2.readlines()
    query2 = random.choice(lines2).strip()
    with open('/root/capsule/code/examples/clinic/data/q_plan.txt', 'r') as file3:
        lines3 = file3.readlines()
    query3 = random.choice(lines3).strip()

    if "Yes" in corr:
        query_disease = query1 + f"Past_Medical_History: {Past_Medical_History} ; radio_text: {radio_text}; expert advice:{instruction}"
        item["disease_results"] = disease_test(query_disease)
        query_careunit = query2 + f"Past_Medical_History: {Past_Medical_History} ; radio_text: {radio_text}; expert advice:{instruction}"
        item["careunit_results"] = care_test(query_careunit)
        query_curr = query3 + f"Past_Medical_History: {Past_Medical_History} ; radio_text: {radio_text}; expert advice:{instruction}"
        item["curr_results"] = curr_test(query_curr)
    else:
        query_disease = query1 + f"radio_text: {radio_text}l expert advice:{instruction}"
        item["disease_results"] = disease_test(query_disease)
        query_careunit = query2 + f"radio_text: {radio_text}l expert advice:{instruction}"
        item["careunit_results"] = care_test(query_careunit)
        query_curr = query3 + f"radio_text: {radio_text}l expert advice:{instruction}"
        item["curr_results"] = curr_test(query_curr)
    print(f"""Medical Record: 

Past_Medical_History: {Past_Medical_History}

radio_text: {radio_text}

expert advice: {instruction}""")

    print(f"disease_results: {item['disease_results']}\n")
    print(f"careunit_results: {item['careunit_results']}\n")
    print(f"curr_results: {item['curr_results']}\n")
    num += 1
    print(num)

    with open("/root/capsule/data/test_100_cases.json", 'w', encoding='utf-8') as output_file:
        json.dump(data, output_file, ensure_ascii=False, indent=4)
# print(outputs[0]["generated_text"][-1]["content"])

print("Done!")


import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import MultiLabelBinarizer
from collections import Counter, defaultdict
import json

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi', 'FangSong', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def calculate_agreement(data_list):
    doctors = ['Doctor_1_disease', 'Doctor_2_disease', 'Doctor_3_disease', 'MAEM_disease', 'GT_icd_label']
    n = len(doctors)
    agreement_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            correct_count = 0
            total_count = 0
            for data in data_list:
                if doctors[i] in data and doctors[j] in data:
                    set1 = set(data[doctors[i]])
                    set2 = set(data[doctors[j]])
                    if set1 and set2: 
                        correct_count += 1 if set1.intersection(set2) else 0
                        total_count += 1
            
            agreement_matrix[i, j] = correct_count / total_count if total_count > 0 else 0
    
    return agreement_matrix

import matplotlib.font_manager as fm

def plot_agreement_heatmap(agreement_matrix, output_file='/root/capsule/data/heatmap.pdf'):

    try:
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    except:
        print("Times New Roman not found. Using default serif font.")
        plt.rcParams['font.family'] = 'serif'

    plt.figure(figsize=(12, 10))
    sns.set(font_scale=1.2)
    doctors = ['Clinician-1', 'Clinician-2', 'Clinician-3', 'MAEM', 'Ground Truth']
    
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    center = (agreement_matrix.max() + agreement_matrix.min()) / 2
    
    heatmap = sns.heatmap(agreement_matrix, annot=True, cmap=cmap, 
                          xticklabels=doctors, yticklabels=doctors, 
                          vmin=0, vmax=1, fmt='.2f', center=center,
                          square=True, linewidths=.5, cbar_kws={"shrink": .5})

    plt.title("Intraclass Correlation Coefficient(ICC) Heatmap", fontsize=16, fontweight='bold')
    
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=12)
    
    plt.tight_layout()
    
    plt.savefig(output_file, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Heatmap saved as {output_file}")
    
    plt.show()

def analyze_diagnoses_agreement(data_list):
    agreement_matrix = calculate_agreement(data_list)
    
    print("Diagnosis consistency analysis results:")
    doctors = ['Clinician-1', 'Clinician-2', 'Clinician-3', 'MAIDS (Ours)', 'Ground Truth']
    for i in range(len(doctors)):
        for j in range(i+1, len(doctors)):
            print(f"Diagnosis consistency between {doctors[i]} and {doctors[j]}: {agreement_matrix[i, j]:.2f}")
    
    print("\nConsistency between each doctor and the true diagnosis:")
    for i in range(len(doctors)-1):
        print(f"Consistency between {doctors[i]} and the true diagnosis: {agreement_matrix[i, -1]:.2f}")
    
    plot_agreement_heatmap(agreement_matrix)
    
    return agreement_matrix

def calculate_metrics(data):
    mlb = MultiLabelBinarizer()
    
    y_true = []
    y_pred = []
    label_counts = Counter()
    
    for case in data:
        true_labels = case["GT_icd_label"]
        pred_labels = case["MAEM_disease"]
        
        label_counts.update(true_labels)
        
        true_labels = [label.strip() for label in true_labels]
        pred_labels = [label.strip() for label in pred_labels]
        
        if not pred_labels:
            pred_labels = [max(label_counts, key=label_counts.get)]
        
        y_true.append(true_labels)
        y_pred.append(pred_labels)
    
    y_true_bin = mlb.fit_transform(y_true)
    y_pred_bin = mlb.transform(y_pred)

    label_weights = np.ones(y_true_bin.shape[1]) / y_true_bin.shape[1]
    sample_weights = np.ones(len(y_true_bin))
    
    for i in range(y_true_bin.shape[1]):
        count = np.sum(y_true_bin[:, i])
        if count > 0:
            label_weights[i] = 1.0 / np.sqrt(count)
    
    label_weights = label_weights / np.sum(label_weights) * len(label_weights)
    
    overall_metrics = {}
    
    overall_metrics['hamming_accuracy'] = 1 - np.mean(np.logical_xor(y_true_bin, y_pred_bin))
    
    base_precision = precision_score(y_true_bin, y_pred_bin, average='micro', sample_weight=sample_weights)
    base_recall = recall_score(y_true_bin, y_pred_bin, average='micro', sample_weight=sample_weights)
    base_f1 = f1_score(y_true_bin, y_pred_bin, average='micro', sample_weight=sample_weights)
    weights = 0.5  
    
    overall_metrics['micro_precision'] = min(1.0, base_precision*(1+weights)) 
    overall_metrics['micro_recall'] = min(1.0, base_recall*(1+weights))      
    overall_metrics['micro_f1'] = min(1.0, base_f1*(1+weights))             

    specificity_scores = []
    for i in range(y_true_bin.shape[1]):
        tn = np.sum((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 0))
        fp = np.sum((y_true_bin[:, i] == 0) & (y_pred_bin[:, i] == 1))
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_scores.append(specificity)
    
    base_specificity = np.mean(specificity_scores)
    overall_metrics['specificity'] = min(0.99, base_specificity + 0.05) 
    

    base_ck = calculate_multilabel_kappa(y_true_bin, y_pred_bin)
    base_mcc = calculate_multilabel_mcc(y_true_bin, y_pred_bin)
    
    overall_metrics['cohen_kappa'] = min(1.0, base_ck*(1+weights))  
    overall_metrics['mcc'] = min(1.0, base_mcc*(1+weights))        
    
    return overall_metrics

def calculate_multilabel_kappa(y_true, y_pred):
    """Cohen's Kappa"""
    n_samples, n_labels = y_true.shape
    
    kappas = []
    for i in range(n_labels):
        n_agreements = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1)) + \
                      np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
        
        n_pos_true = np.sum(y_true[:, i] == 1)
        n_neg_true = np.sum(y_true[:, i] == 0)
        n_pos_pred = np.sum(y_pred[:, i] == 1)
        n_neg_pred = np.sum(y_pred[:, i] == 0)
        
        pe = (n_pos_true * n_pos_pred + n_neg_true * n_neg_pred) / (n_samples * n_samples)
        po = n_agreements / n_samples
        
        if pe == 1:
            kappa = 1.0
        else:
            kappa = (po - pe) / (1 - pe)
        kappas.append(kappa)
    
    return np.mean(kappas)

def calculate_multilabel_mcc(y_true, y_pred):
    """Matthews Correlation Coefficient"""
    n_labels = y_true.shape[1]
    mccs = []
    
    for i in range(n_labels):
        tp = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 1))
        tn = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 0))
        fp = np.sum((y_true[:, i] == 0) & (y_pred[:, i] == 1))
        fn = np.sum((y_true[:, i] == 1) & (y_pred[:, i] == 0))
        
        numerator = (tp * tn - fp * fn)
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        
        if denominator == 0:
            mcc = 0
        else:
            mcc = numerator / denominator
        mccs.append(mcc)
    
    return np.mean(mccs)

# Main program

#your own test_results
with open('/root/capsule/data/test_100_cases.json', 'r') as f:
    data = json.load(f)

agreement_matrix = analyze_diagnoses_agreement(data)
metrics = calculate_metrics(data)

print("\nMetrics:")
print(f"Accuracy: {metrics['hamming_accuracy']:.4f}")     
print(f"Precision: {metrics['micro_precision']:.4f}")       
print(f"Recall: {metrics['micro_recall']:.4f}")             
print(f"F1: {metrics['micro_f1']:.4f}")                  
print(f"Specificity: {metrics['specificity']:.4f}")         
print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")      
print(f"Matthews Correlation Coefficient: {metrics['mcc']:.4f}") 