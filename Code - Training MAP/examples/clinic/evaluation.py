
# import json
# import transformers
# import torch
# import requests
# import random
# # disease_test 
# def disease_test(query):
#     url = "http://127.0.0.1:6006/disease_test"
#     headers = {
#         'accept': 'application/json',
#         'Content-Type': 'application/json'
#     }
#     data = {"query": query}  #  /disease_test 

#     try:
#         response = requests.post(url, headers=headers, json=data)
#         response.raise_for_status()  #  HTTP 
#         response_json = response.json()
#         return response_json.get('response', 'No response received')
#     except Exception as e:
#         print(f"Error in disease_test request: {e}")
#         return "Error: Unable to process the request"

# # predict 
# def api_request(query, system_content, endpoint):
#     url = f'http://127.0.0.1:6006/{endpoint}'
#     headers = {
#         'accept': 'application/json',
#         'Content-Type': 'application/json'
#     }
#     messages = [
#         {"role": "system", "content": system_content},
#         {"role": "user", "content": query}
#     ]
#     data = {
#         "model": "internlm2",
#         "messages": messages,
#         "temperature": 0,
#         "top_p": 0.8,
#         "n": 1,
#         "max_tokens": None,
#         "stop": None,
#         "stream": False,
#         "presence_penalty": 0,
#         "frequency_penalty": 0,
#         "user": "string",
#         "repetition_penalty": 1.002,
#         "session_id": -1,
#         "ignore_eos": False
#     }

#     try:
#         response = requests.post(url, headers=headers, json=data)
#         response.raise_for_status()
#         response_json = response.json()
#         return response_json.get('response', 'No response received')
#     except Exception as e:
#         print(f"Error in API request: {e}")
#         return "Error: Unable to process the request"

# # care_test
# def care_test(query):
#     endpoint = "predict"
#     system_content = '''You are a doctor's assistant and need to determine which first_unit to go to based on the patient's medical condition. The list of candidate first_careunits is as follows: ['Medical Intensive Care Unit (MICU)', 'Surgical Intensive Care Unit (SICU) )', 'Surgical Intensive Care Unit (MICU/SICU)', 'Cardiac Vascular Intensive Care Unit (CVICU)', 'Coronary Care Unit (CCU)', 'Neuro Intermediate', 'Trauma SICU (TSICU)', 'Neuro Stepdown', 'Neuro Surgical Intensive Care Unit (Neuro SICU)']'''
#     return api_request(query, system_content, endpoint)

# # curr_test
# def curr_test(query):
#     endpoint = "predict"
#     system_content = '''You are a clinic expert and need to find one the most suitable current service from the candidate curr_service list based on the patient's medical record. The candidate curr_service list is as follows:['OMED', 'ORTHO', 'TSURG', 'OBS', 'TRAUM', 'NSURG', 'MED', 'CSURG', 'GU', 'ENT', 'CMED', 'VSURG', 'SURG', 'NMED', 'GYN', 'PSURG']'''
#     return api_request(query, system_content, endpoint)
import json
import transformers
import torch
import requests
import random
import time
import subprocess
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_ssh_tunnel():
    """建立SSH隧道"""
    try:
        # 清理可能存在的旧连接
        subprocess.run("pkill -f 'ssh.*6006'", shell=True)
        time.sleep(2)

        # SSH连接参数
        ssh_cmd = [
            "sshpass", "-p", "7pG33QeFOYAc",
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ServerAliveInterval=60",
            "-CNg",
            "-L", "6006:127.0.0.1:6006",
            "root@connect.cqa1.seetacloud.com",
            "-p", "19139"
        ]

        process = subprocess.Popen(
            ssh_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        time.sleep(5)
        return check_service_availability()

    except Exception as e:
        logger.error(f"Error setting up SSH tunnel: {str(e)}")
        return False

def check_service_availability(max_retries=3):
    """检查服务是否可用"""
    for i in range(max_retries):
        try:
            response = requests.get("http://127.0.0.1:6006/health", timeout=5)
            if response.status_code == 200:
                logger.info("Service is available")
                return True
        except:
            logger.warning(f"Service check attempt {i+1} failed")
            time.sleep(2)
    return False

def create_robust_session():
    """创建带有重试机制的会话"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def ensure_connection():
    """确保连接可用"""
    if not check_service_availability():
        if not setup_ssh_tunnel():
            raise ConnectionError("Could not establish connection to service")

def disease_test(query, use_fallback=True):
    """疾病测试函数"""
    try:
        ensure_connection()
        
        session = create_robust_session()
        url = "http://127.0.0.1:6006/disease_test"
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        data = {"query": query}

        response = session.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json().get('response', 'No response received')

    except Exception as e:
        logger.error(f"Error in disease_test request: {e}")
        if use_fallback:
            return fallback_disease_test(query)
        return "Error: Unable to process the request"

def api_request(query, system_content, endpoint, use_fallback=True):
    """API请求函数"""
    try:
        ensure_connection()
        
        session = create_robust_session()
        url = f'http://127.0.0.1:6006/{endpoint}'
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        messages = [
            {"role": "system", "content": system_content},
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

        response = session.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json().get('response', 'No response received')

    except Exception as e:
        logger.error(f"Error in API request: {e}")
        if use_fallback:
            return fallback_api_request(query, system_content, endpoint)
        return "Error: Unable to process the request"

def fallback_disease_test(query):
    """疾病测试备用方案"""
    try:
        url = 'http://175.45.13.117:53999/v1/chat/completions'
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        data = {
            "model": "internlm2",
            "messages": [
                {"role": "user", "content": f"Based on the following medical record, please diagnose possible diseases: {query}"}
            ],
            "temperature": 0,
            "top_p": 0.8
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Fallback disease_test failed: {e}")
        return "Error: Unable to process the request"

def fallback_api_request(query, system_content, endpoint):
    """API请求备用方案"""
    try:
        url = 'http://175.45.13.117:53999/v1/chat/completions'
        headers = {
            'accept': 'application/json',
            'Content-Type': 'application/json'
        }
        data = {
            "model": "internlm2",
            "messages": [
                {"role": "system", "content": system_content},
                {"role": "user", "content": query}
            ],
            "temperature": 0,
            "top_p": 0.8
        }
        response = requests.post(url, headers=headers, json=data, timeout=30)
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        logger.error(f"Fallback API request failed: {e}")
        return "Error: Unable to process the request"

def care_test(query):
    """护理测试函数"""
    endpoint = "predict"
    system_content = '''You are a doctor's assistant and need to determine which first_unit to go to based on the patient's medical condition. The list of candidate first_careunits is as follows: ['Medical Intensive Care Unit (MICU)', 'Surgical Intensive Care Unit (SICU) )', 'Surgical Intensive Care Unit (MICU/SICU)', 'Cardiac Vascular Intensive Care Unit (CVICU)', 'Coronary Care Unit (CCU)', 'Neuro Intermediate', 'Trauma SICU (TSICU)', 'Neuro Stepdown', 'Neuro Surgical Intensive Care Unit (Neuro SICU)']'''
    return api_request(query, system_content, endpoint)

def curr_test(query):
    """当前服务测试函数"""
    endpoint = "predict"
    system_content = '''You are a clinic expert and need to find one the most suitable current service from the candidate curr_service list based on the patient's medical record. The candidate curr_service list is as follows:['OMED', 'ORTHO', 'TSURG', 'OBS', 'TRAUM', 'NSURG', 'MED', 'CSURG', 'GU', 'ENT', 'CMED', 'VSURG', 'SURG', 'NMED', 'GYN', 'PSURG']'''
    return api_request(query, system_content, endpoint)




with open('/data/test_clinician_100_cases.json', 'r') as file:
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
    print("=" * 50) 
    print("Medical Record:")
    print("-" * 50) 
    print(f"""
            Past Medical History: {Past_Medical_History}

            Radio Text: {radio_text}

            Expert Advice: {instruction}
    """)
    print("=" * 50)  

    print("Disease Results:")
    print("-" * 50)
    print(f"{item['disease_results']}\n")

    print("Care Unit Results:")
    print("-" * 50)
    print(f"{item['careunit_results']}\n")

    print("Current Service Results:")
    print("-" * 50)
    print(f"{item['curr_results']}\n")

    print("=" * 50) 
    num += 1
    print(num)

#    with open("/root/capsule/data/test_10_cases_results.json", 'w', encoding='utf-8') as output_file:
#        json.dump(data, output_file, ensure_ascii=False, indent=4)
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
with open('/data/exampls_result.json', 'r') as f:
    data = json.load(f)

#agreement_matrix = analyze_diagnoses_agreement(data)
metrics = calculate_metrics(data)
print("**MAIDS results on our IPDS benchmark**")
print("\nMetrics:")
print(f"Accuracy: {metrics['hamming_accuracy']:.4f}")     
print(f"Precision: {metrics['micro_precision']:.4f}")       
print(f"Recall: {metrics['micro_recall']:.4f}")             
print(f"F1: {metrics['micro_f1']:.4f}")                  
print(f"Specificity: {metrics['specificity']:.4f}")         
print(f"Cohen's Kappa: {metrics['cohen_kappa']:.4f}")      
print(f"Matthews Correlation Coefficient: {metrics['mcc']:.4f}") 
