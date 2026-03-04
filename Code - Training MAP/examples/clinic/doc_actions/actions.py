 """
Filename:MAgent.py
Created Date: Tuesday, Oct 27th 2024, 6:52:25 pm
Author: XiaoQi
"""

'''
icustays_units = ['Medical Intensive Care Unit (MICU)',
 'Surgical Intensive Care Unit (SICU)',
 'Surgical Intensive Care Unit (MICU/SICU)',
 'Cardiac Vascular Intensive Care Unit (CVICU)', 'Coronary Care Unit (CCU)',
 'Neuro Intermediate', 'Trauma SICU (TSICU)', 'Neuro Stepdown',
 'Neuro Surgical Intensive Care Unit (Neuro SICU)']
'''
import asyncio
from datetime import datetime
from metagpt.environment import Environment
from metagpt.roles import Role
from metagpt.team import Team
from metagpt.actions import Action, UserRequirement
from pydantic import ConfigDict, Field
from gymnasium import spaces
from metagpt.logs import logger
from typing import Optional
import platform
from typing import Any
import os
os.environ["GIT_PYTHON_REFRESH"] = "quiet"
import fire
import json
import re
from metagpt.schema import Message
import git
import torch
import transformers
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import requests
disease_query_path = "/root/capsule/code/examples/clinic/data/q_disease.txt"
curr_query_path= "/root/capsule/code/examples/clinic/data/q_plan.txt"
sub_memory_path = "/root/capsule/code/examples/clinic/data/sub_memory.json"
dynamic_memory_path = "/root/capsule/code/examples/clinic/data/dynamic_memory.json"
static_memory_path = "/root/capsule/code/examples/clinic/data/static_memory.json"
# model_id = "/home/xusheng_liang/cair/xtuner-main/RAG/LLaMA3_8B_Instruct_10_08"

# model_id = "QIOvO/maids"
# model_id = ""
# pipeline = transformers.pipeline(
#     "text-generation",
#     model=model_id,
#     model_kwargs={"torch_dtype": torch.bfloat16},
#     device_map=0,
# )

def calculate_similarity(text1, text2):
    vectorizer = TfidfVectorizer().fit_transform([text1, text2])
    cosine_sim = cosine_similarity(vectorizer)
    return cosine_sim[0][1]

def Correlation_calculation(record, threshold=0.7):
    if not isinstance(record, str) or not record:
        print("Error: Invalid record.")
        return None

    pmh_match = re.search(r'Past Medical History:(.*?)(?=\n\s*\S+:|\Z)', record, re.DOTALL)
    rr_match = re.search(r'Radiology Report:(.*?)(?=\n\s*\S+:|\Z)', record, re.DOTALL)
    
    if pmh_match and rr_match:
        pmh = pmh_match.group(0).strip()  
        rr = rr_match.group(1).strip()
        
        similarity = calculate_similarity(pmh, rr)
        
        if similarity < threshold:
            new_record = re.sub(r'Past Medical History:.*?(?=\n\s*\S+:|\Z)', '', record, flags=re.DOTALL)
            return new_record.strip()
    return record

# def Trainable_RAG(Query):
#     with open(disease_query_path, 'r') as file1:
#         lines1 = file1.readlines()
#     query1 = random.choice(lines1).strip()
#     with open(curr_query_path, 'r') as file2:
#         lines2 = file2.readlines()
#     query1 = random.choice(lines1).strip()
#     query2 = random.choice(lines2).strip()
#     Disease_query = Query + "\n" + query1
#     messages = [
#     {"role": "system", "content": '''You are a clinic expert and need to make a diagnose for possible diseases from the candidate disease list based on the patient's medical record information. The candidate disease list is as follows:["Certain infectious and parasitic diseases","Neoplasms","Endocrine, nutritional and metabolic Diseases","Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism","Mental and behavioural disorders","Diseases of the nervous system and sense organs","Diseases of the circulatory system","Diseases of the respiratory system","Diseases of the digestive system","Diseases of the genitourinary system","Pregnancy, childbirth and the puerperium","Diseases of the skin and subcutaneous tissue","Diseases of the musculoskeletal system and connective tissue","Congenital malformations, deformations and chromosomal abnormalities","Certain conditions originating in the perinatal period","Symptoms, signs and abnormal clinical and laboratory findings","Injury, poisoning and certain other consequences of external causes"]'''},
#     {"role": "user", "content": Disease_query},
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
#     Curr_query = Query + "\n" + query2
#     messages_curr = [
#     {"role": "system", "content": '''You are a clinic expert and need to find one the most suitable current service from the candidate curr_service list based on the patient's medical record. The candidate curr_service list is as follows:['OMED', 'ORTHO', 'TSURG', 'OBS', 'TRAUM', 'NSURG', 'MED', 'CSURG', 'GU', 'ENT', 'CMED', 'VSURG', 'SURG', 'NMED', 'GYN', 'PSURG']'''},
#     {"role": "user", "content": Curr_query},
# ]

#     terminators_curr = [
#         pipeline.tokenizer.eos_token_id,
#         pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
#     ]

#     outputs_curr = pipeline(
#         messages_curr,
#         max_new_tokens=4096,
#         eos_token_id=terminators,
#         do_sample=True,
#         temperature=0.6,
#         top_p=0.9,
#     )
#     Disease = "Diagnostic results:  " + outputs[0]["generated_text"][-1]["content"]
#     Curr = "Current_service results:  " + outputs_curr[0]["generated_text"][-1]["content"]

#     return Disease,Curr


def cares_copilot(query):

    url = 'http://175.45.13.117:53999/v1/chat/completions'
    headers = {
    'accept': 'application/json',
    'Content-Type': 'application/json'
    }

    data = {
        "model": "internlm2",
        "messages": [
            {"role": "user", "content": f'Please pla{query}'}
        ],
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

# def Trainable_RAG(Query):
#     with open(disease_query_path, 'r') as file1:
#         lines1 = file1.readlines()
#     query1 = random.choice(lines1).strip()
#     with open(curr_query_path, 'r') as file2:
#         lines2 = file2.readlines()
#     query1 = random.choice(lines1).strip()
#     query2 = random.choice(lines2).strip()
#     Disease_query = Query + "\n" + query1

#     q1 = f'''You are a clinic expert and need to make a diagnose for possible diseases from the candidate disease list based on the patient's medical record information. The candidate disease list is as follows:["Certain infectious and parasitic diseases","Neoplasms","Endocrine, nutritional and metabolic Diseases","Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism","Mental and behavioural disorders","Diseases of the nervous system and sense organs","Diseases of the circulatory system","Diseases of the respiratory system","Diseases of the digestive system","Diseases of the genitourinary system","Pregnancy, childbirth and the puerperium","Diseases of the skin and subcutaneous tissue","Diseases of the musculoskeletal system and connective tissue","Congenital malformations, deformations and chromosomal abnormalities","Certain conditions originating in the perinatal period","Symptoms, signs and abnormal clinical and laboratory findings","Injury, poisoning and certain other consequences of external causes"] \n\n query:{Disease_query}'''
#     disease = cares_copilot(q1)


#     Curr_query = Query + "\n" + query2

#     q2 = f'''You are a clinic expert and need to find one the most suitable current service from the candidate curr_service list based on the patient's medical record. The candidate curr_service list is as follows:['OMED', 'ORTHO', 'TSURG', 'OBS', 'TRAUM', 'NSURG', 'MED', 'CSURG', 'GU', 'ENT', 'CMED', 'VSURG', 'SURG', 'NMED', 'GYN', 'PSURG'] \n\n query:{Curr_query}'''
#     curr = cares_copilot(q2)
#     Disease = "Diagnostic results:  " + disease
#     Curr = "Current_service results:  " + curr

#     return Disease,Curr

import requests
import random


disease_query_path = "/root/capsule/code/examples/clinic/data/q_disease.txt"
curr_query_path = "/root/capsule/code/examples/clinic/data/q_plan.txt"

import requests
import random
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import requests
import random
import time
import subprocess
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

        # 使用subprocess.Popen启动SSH隧道
        process = subprocess.Popen(
            ssh_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        # 等待隧道建立
        time.sleep(3)

        # 检查端口是否已经打开
        check_cmd = "netstat -tulnp | grep :6006"
        result = subprocess.run(check_cmd, shell=True)
        
        if result.returncode == 0:
            print("SSH tunnel established successfully")
            return True
        else:
            print("Failed to establish SSH tunnel")
            return False

    except Exception as e:
        print(f"Error setting up SSH tunnel: {str(e)}")
        return False

def wait_for_service(timeout=30):
    """等待服务可用"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get("http://127.0.0.1:6006/health", timeout=5)
            if response.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False

def Trainable_RAG(Query):
    # 确保SSH隧道已建立
    if not setup_ssh_tunnel():
        # 如果无法建立SSH隧道，使用备用API
        return fallback_RAG(Query)
    
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[500, 502, 503, 504]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    
    try:
        # 读取示例查询
        with open(disease_query_path, 'r') as file1:
            lines1 = file1.readlines()
        query1 = random.choice(lines1).strip()

        with open(curr_query_path, 'r') as file2:
            lines2 = file2.readlines()
        query2 = random.choice(lines2).strip()

        Disease_query = Query + "\n" + query1
        Curr_query = Query + "\n" + query2

        # 尝试使用本地服务
        try:
            # Disease prediction
            disease_response = session.post(
                "http://127.0.0.1:6006/disease_test",
                json={"query": Disease_query},
                timeout=30
            )
            disease_result = disease_response.json().get("response")

            # Current service prediction
            curr_response = session.post(
                "http://127.0.0.1:6006/predict",
                json={
                    "messages": [
                        {
                            "role": "system",
                            "content": '''You are a clinic expert and need to find one the most suitable current service from the candidate curr_service list based on the patient's medical record. The candidate curr_service list is as follows:['OMED', 'ORTHO', 'TSURG', 'OBS', 'TRAUM', 'NSURG', 'MED', 'CSURG', 'GU', 'ENT', 'CMED', 'VSURG', 'SURG', 'NMED', 'GYN', 'PSURG']'''
                        },
                        {"role": "user", "content": Curr_query}
                    ]
                },
                timeout=30
            )
            curr_result = curr_response.json().get("response")

        except Exception as e:
            print(f"Local service failed: {str(e)}")
            return fallback_RAG(Query)

        return (
            f"Diagnostic results: {disease_result}",
            f"Current_service results: {curr_result}"
        )

    except Exception as e:
        print(f"Error in Trainable_RAG: {str(e)}")
        return fallback_RAG(Query)

def fallback_RAG(Query):
    """备用方案：使用cares_copilot API"""
    try:
        # 使用直接API调用
        disease_query = f'''You are a clinic expert and need to make a diagnose for possible diseases from the candidate disease list based on the patient's medical record information. The candidate disease list is as follows:["Certain infectious and parasitic diseases","Neoplasms","Endocrine, nutritional and metabolic Diseases","Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism","Mental and behavioural disorders","Diseases of the nervous system and sense organs","Diseases of the circulatory system","Diseases of the respiratory system","Diseases of the digestive system","Diseases of the genitourinary system","Pregnancy, childbirth and the puerperium","Diseases of the skin and subcutaneous tissue","Diseases of the musculoskeletal system and connective tissue","Congenital malformations, deformations and chromosomal abnormalities","Certain conditions originating in the perinatal period","Symptoms, signs and abnormal clinical and laboratory findings","Injury, poisoning and certain other consequences of external causes"] \n\n query:{Query}'''
        
        curr_query = f'''You are a clinic expert and need to find one the most suitable current service from the candidate curr_service list based on the patient's medical record. The candidate curr_service list is as follows:['OMED', 'ORTHO', 'TSURG', 'OBS', 'TRAUM', 'NSURG', 'MED', 'CSURG', 'GU', 'ENT', 'CMED', 'VSURG', 'SURG', 'NMED', 'GYN', 'PSURG'] \n\n query:{Query}'''
        
        disease_result = cares_copilot(disease_query)
        curr_result = cares_copilot(curr_query)
        
        return (
            f"Diagnostic results: {disease_result}",
            f"Current_service results: {curr_result}"
        )
    except Exception as e:
        print(f"Fallback RAG failed: {str(e)}")
        return "Error in Disease query", "Error in Curr_service query"


# def Trainable_RAG(Query):
#     try:

#         with open(disease_query_path, 'r') as file1:
#             lines1 = file1.readlines()
#         if not lines1:
#             raise ValueError("disease_query_path file is empty")  
#         query1 = random.choice(lines1).strip()

#         with open(curr_query_path, 'r') as file2:
#             lines2 = file2.readlines()
#         if not lines2:
#             raise ValueError("curr_query_path file is empty") 
#         query2 = random.choice(lines2).strip()


#         Disease_query = Query + "\n" + query1
#         disease_response = requests.post(
#             "http://127.0.0.1:6006/disease_test",
#             json={"query": Disease_query},
#             timeout=30
#         )


#         if disease_response.status_code == 200:

#             disease_result = disease_response.json().get("response", "No response from disease_test")
#         else:
#             disease_result = f"Error: {disease_response.status_code}, {disease_response.text}"

#         Curr_query = Query + "\n" + query2
#         messages_curr = [
#             {"role": "system", "content": '''You are a clinic expert and need to find one the most suitable current service from the candidate curr_service list based on the patient's medical record. The candidate curr_service list is as follows:['OMED', 'ORTHO', 'TSURG', 'OBS', 'TRAUM', 'NSURG', 'MED', 'CSURG', 'GU', 'ENT', 'CMED', 'VSURG', 'SURG', 'NMED', 'GYN', 'PSURG']'''},
#             {"role": "user", "content": Curr_query}
#         ]
#         predict_response = requests.post(
#             "http://127.0.0.1:6006/predict",
#             json={"messages": messages_curr},
#             timeout=30
#         )
#         print("Predict Full Response:", predict_response.text) 

#         if predict_response.status_code == 200:
 
#             curr_service_result = predict_response.json().get("response", "No response from predict")
#         else:
#             curr_service_result = f"Error: {predict_response.status_code}, {predict_response.text}"


#         Disease = "Diagnostic results:  " + disease_result
#         Curr = "Current_service results:  " + curr_service_result

#         return Disease, Curr

#     except Exception as e:
#         print(f"Error in Trainable_RAG: {e}")
#         return "Error in Disease query", "Error in Curr_service query"

class ClinicAssignment(Action):

    PROMPT_TEMPLATE: str = """
    ## Role definition
    Suppose you are a ICU Clinic Assistant, you need to assign one or more diagnostic department for the patient according given a patient's medical record.

    ## PATIENT'S PATIENT'S MEDICAL RECORD
    {context}
    
    ## Duty
    Now it's your turn,you need to assign one or more diagnostic department for the patient according given a patient's medical record{context} and give reasons,
    You need to pay more attention to the patient's imaging report to make a decision and the diagnostic department you provide must be belong to the following departments:
    "Medical Intensive Care Unit","Surgical Intensive Care Unit","Cardiac Vascular Intensive Care Unit","Coronary Care Unit","Neuro Intermediate","Trauma SICU","Neuro Stepdown","Neuro Surgical Intensive Care Unit","Surgical Intensive Care Unit (MICU/SICU)"

    return ```Diagnostic Department: ```  
    return ```Reasons:```  
    """
    name: str = "Plan"

    async def run(self, context: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context)
        logger.info(prompt)
        rsp = await self._aask(prompt)

        list = []
        list.append(rsp)

        with open(sub_memory_path, "w") as f:
            json.dump(list, f, indent=2)  

        return rsp

class MICU_Joint_Consultation(Action):

    PROMPT_TEMPLATE_INTERN: str = """
    ## ROLE DEFINITION
    Suppose you are {name} in a MICU Diagnostic Doctor Team which responsible for treating patients with serious symptoms due to medical diseases, you are collaborating with  {collaboratorA_name}to diagnose the patient based on patient's medical record.
    You are a young MICU doctor who focus on the diagnosis and treatment of internal medicine diseases with little clinical experience and are pone to discosure.
    ##  PATIENT'S MEDICAL RECORD
    {context}
    ## DUTY
    Now it's your turn, you need to provide preliminary disease diagnosis and current service plans based on the electronic medical records of emergency patients and guidance. Pay attention to the guidance，it is important to help you make a right diagnosis. In addition, you need to pay close attention to the recommendations and guidance of {collaboratorA_name}. If your diagnosis is wrong, you need to re-diagnose and propose a treatment plan based on the recommendations.
    The language should be refined and relevant to the disease diagnosis.

    return ```Diagnostic results: ```
    return ```Current_service results: ```

    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.
    """

    PROMPT_TEMPLATE_CHIEF: str = """
    ## Role definition
    Suppose you are {name} in the MICU diagnostic team, responsible for treating patients with severe symptoms due to illness. You are working with {collaboratorA_name}, mainly guiding interns to help her make the correct diagnosis and treatment plan.

    ## Patient's medical record
    {context}
    ## Responsibilities
    Now it is your turn. You should pay close attention to the latest diagnosis and treatment plan of your collaborator {collaboratorA_name}. If there is an error in his diagnostic results, you need to correct his mistakes or problems  and give advice and guidance based on Patient's medical record,.

    Return ```Is it true?: ```
    Return ```Guidance: ```

    Avoid continuous questions and answers, wait for the other party to respond before asking the next round of questions.
"""

    name: str = "Cooperation"


    async def run(self, context: str, name: str, collaboratorA_name: str):
        PMH_module = Correlation_calculation(str(context[0]))
        retrieval_result = Trainable_RAG(PMH_module)
        
        if name == "Doctor_Intern":
            prompt = self.PROMPT_TEMPLATE_INTERN.format(context=str(context[0]) + str(retrieval_result), name=name, collaboratorA_name=collaboratorA_name)
            rsp = await self._aask(prompt)

        elif name == "Doctor_Chief":
            prompt = self.PROMPT_TEMPLATE_CHIEF.format(context=context, name=name, collaboratorA_name=collaboratorA_name)
            rsp = await self._aask(prompt)

        return rsp



class SICU_Joint_Consultation(Action):

    PROMPT_TEMPLATE_INTERN: str = """
    ## ROLE DEFINITION
    Suppose you are {name} in a MICU Diagnostic Doctor Team which responsible for treating patients with serious symptoms due to medical diseases, you are collaborating with  {collaboratorA_name}to diagnose the patient based on patient's medical record.
    You are a young MICU doctor who focus on the diagnosis and treatment of internal medicine diseases with little clinical experience and are pone to discosure.
    ##  PATIENT'S MEDICAL RECORD
    {context}
    ## DUTY
    Now it's your turn, you need to provide preliminary disease diagnosis and current service plans based on the electronic medical records of emergency patients and guidance. Pay attention to the guidance，it is important to help you make a right diagnosis. In addition, you need to pay close attention to the recommendations and guidance of {collaboratorA_name}. If your diagnosis is wrong, you need to re-diagnose and propose a treatment plan based on the recommendations.
    The language should be refined and relevant to the disease diagnosis.
    
    return ```Diagnostic results: ```
    return ```Current_service results: ```

    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.
    """

    PROMPT_TEMPLATE_CHIEF: str = """
    ## Role definition
    Suppose you are {name} in the MICU diagnostic team, responsible for treating patients with severe symptoms due to illness. You are working with {collaboratorA_name}, mainly guiding interns to help her make the correct diagnosis and treatment plan.

    ## Patient's medical record
    {context}
    ## Responsibilities
    Now it is your turn. You should pay close attention to the latest diagnosis and treatment plan of your collaborator {collaboratorA_name}. If there is an error in his diagnostic results, you need to correct his mistakes or problems  and give advice and guidance based on Patient's medical record,.

    Return ```Is it true?: ```
    Return ```Guidance: ```

    Avoid continuous questions and answers, wait for the other party to respond before asking the next round of questions.
"""

    name: str = "Cooperation"


    async def run(self, context: str, name: str, collaboratorA_name: str):
        PMH_module = Correlation_calculation(str(context[0]))
        retrieval_result = Trainable_RAG(PMH_module)
        
        if name == "Doctor_Intern":
            prompt = self.PROMPT_TEMPLATE_INTERN.format(context=str(context[0]) + str(retrieval_result), name=name, collaboratorA_name=collaboratorA_name)
            rsp = await self._aask(prompt)
        elif name == "Doctor_Chief":
            prompt = self.PROMPT_TEMPLATE_CHIEF.format(context=context, name=name, collaboratorA_name=collaboratorA_name)

        rsp = await self._aask(prompt)

        return rsp

#MICU/SICU
class MICU_SICU_Joint_Consultation(Action):

    PROMPT_TEMPLATE_INTERN: str = """
    ## ROLE DEFINITION
    Suppose you are {name} in a MICU Diagnostic Doctor Team which responsible for treating patients with serious symptoms due to medical diseases, you are collaborating with  {collaboratorA_name}to diagnose the patient based on patient's medical record.
    You are a young MICU doctor who focus on the diagnosis and treatment of internal medicine diseases with little clinical experience and are pone to discosure.
    ##  PATIENT'S MEDICAL RECORD
    {context}
        ## DUTY
    Now it's your turn, you need to provide preliminary disease diagnosis and current service plans based on the electronic medical records of emergency patients and guidance. Pay attention to the guidance，it is important to help you make a right diagnosis. In addition, you need to pay close attention to the recommendations and guidance of {collaboratorA_name}. If your diagnosis is wrong, you need to re-diagnose and propose a treatment plan based on the recommendations.
    The language should be refined and relevant to the disease diagnosis.
    
    return ```Diagnostic results: ```
    return ```Current_service results: ```

    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.
    """

    PROMPT_TEMPLATE_CHIEF: str = """
    ## Role definition
    Suppose you are {name} in the MICU diagnostic team, responsible for treating patients with severe symptoms due to illness. You are working with {collaboratorA_name}, mainly guiding interns to help her make the correct diagnosis and treatment plan.

    ## Patient's medical record
    {context}
    ## Responsibilities
    Now it is your turn. You should pay close attention to the latest diagnosis and treatment plan of your collaborator {collaboratorA_name}. If there is an error in his diagnostic results, you need to correct his mistakes or problems  and give advice and guidance based on Patient's medical record,.

    Return ```Is it true?: ```
    Return ```Guidance: ```

    Avoid continuous questions and answers, wait for the other party to respond before asking the next round of questions.
"""

    name: str = "Cooperation"


    async def run(self, context: str, name: str, collaboratorA_name: str):
        PMH_module = Correlation_calculation(str(context[0]))
        retrieval_result = Trainable_RAG(PMH_module)
        
        if name == "Doctor_Intern":
            prompt = self.PROMPT_TEMPLATE_INTERN.format(context=str(context[0]) + str(retrieval_result), name=name, collaboratorA_name=collaboratorA_name)
            rsp = await self._aask(prompt)
        elif name == "Doctor_Chief":
            prompt = self.PROMPT_TEMPLATE_CHIEF.format(context=context, name=name, collaboratorA_name=collaboratorA_name)

        rsp = await self._aask(prompt)

        return rsp
    
#CVICU
class CVICU_Joint_Consultation(Action):

    PROMPT_TEMPLATE_INTERN: str = """
    ## ROLE DEFINITION
    Suppose you are {name} in a MICU Diagnostic Doctor Team which responsible for treating patients with serious symptoms due to medical diseases, you are collaborating with  {collaboratorA_name}to diagnose the patient based on patient's medical record.
    You are a young MICU doctor who focus on the diagnosis and treatment of internal medicine diseases with little clinical experience and are pone to discosure.
    ##  PATIENT'S MEDICAL RECORD
    {context}
        ## DUTY
    Now it's your turn, you need to provide preliminary disease diagnosis and current service plans based on the electronic medical records of emergency patients and guidance. Pay attention to the guidance，it is important to help you make a right diagnosis. In addition, you need to pay close attention to the recommendations and guidance of {collaboratorA_name}. If your diagnosis is wrong, you need to re-diagnose and propose a treatment plan based on the recommendations.
    The language should be refined and relevant to the disease diagnosis.
    
    return ```Diagnostic results: ```
    return ```Current_service results: ```

    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.
    """

    PROMPT_TEMPLATE_CHIEF: str = """
    ## Role definition
    Suppose you are {name} in the MICU diagnostic team, responsible for treating patients with severe symptoms due to illness. You are working with {collaboratorA_name}, mainly guiding interns to help her make the correct diagnosis and treatment plan.

    ## Patient's medical record
    {context}
    ## Responsibilities
    Now it is your turn. You should pay close attention to the latest diagnosis and treatment plan of your collaborator {collaboratorA_name}. If there is an error in his diagnostic results, you need to correct his mistakes or problems  and give advice and guidance based on Patient's medical record,.

    Return ```Is it true?: ```
    Return ```Guidance: ```

    Avoid continuous questions and answers, wait for the other party to respond before asking the next round of questions.
"""

    name: str = "Cooperation"


    async def run(self, context: str, name: str, collaboratorA_name: str):
        PMH_module = Correlation_calculation(str(context[0]))
        retrieval_result = Trainable_RAG(PMH_module)
        
        if name == "Doctor_Intern":
            prompt = self.PROMPT_TEMPLATE_INTERN.format(context=str(context[0]) + str(retrieval_result), name=name, collaboratorA_name=collaboratorA_name)
            rsp = await self._aask(prompt)
        elif name == "Doctor_Chief":
            prompt = self.PROMPT_TEMPLATE_CHIEF.format(context=context, name=name, collaboratorA_name=collaboratorA_name)

        rsp = await self._aask(prompt)

        return rsp


#CCU
class CCU_Joint_Consultation(Action):

    PROMPT_TEMPLATE_INTERN: str = """
    ## ROLE DEFINITION
    Suppose you are {name} in a MICU Diagnostic Doctor Team which responsible for treating patients with serious symptoms due to medical diseases, you are collaborating with  {collaboratorA_name}to diagnose the patient based on patient's medical record.
    You are a young MICU doctor who focus on the diagnosis and treatment of internal medicine diseases with little clinical experience and are pone to discosure.
    ##  PATIENT'S MEDICAL RECORD
    {context}
        ## DUTY
    Now it's your turn, you need to provide preliminary disease diagnosis and current service plans based on the electronic medical records of emergency patients and guidance. Pay attention to the guidance，it is important to help you make a right diagnosis. In addition, you need to pay close attention to the recommendations and guidance of {collaboratorA_name}. If your diagnosis is wrong, you need to re-diagnose and propose a treatment plan based on the recommendations.
    The language should be refined and relevant to the disease diagnosis.
    
    return ```Diagnostic results: ```
    return ```Current_service results: ```

    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.
    """

    PROMPT_TEMPLATE_CHIEF: str = """
    ## Role definition
    Suppose you are {name} in the MICU diagnostic team, responsible for treating patients with severe symptoms due to illness. You are working with {collaboratorA_name}, mainly guiding interns to help her make the correct diagnosis and treatment plan.

    ## Patient's medical record
    {context}
    ## Responsibilities
    Now it is your turn. You should pay close attention to the latest diagnosis and treatment plan of your collaborator {collaboratorA_name}. If there is an error in his diagnostic results, you need to correct his mistakes or problems  and give advice and guidance based on Patient's medical record,.

    Return ```Is it true?: ```
    Return ```Guidance: ```

    Avoid continuous questions and answers, wait for the other party to respond before asking the next round of questions.
"""

    name: str = "Cooperation"


    async def run(self, context: str, name: str, collaboratorA_name: str):
        PMH_module = Correlation_calculation(str(context[0]))
        retrieval_result = Trainable_RAG(PMH_module)
        
        if name == "Doctor_Intern":
            prompt = self.PROMPT_TEMPLATE_INTERN.format(context=str(context[0]) + str(retrieval_result), name=name, collaboratorA_name=collaboratorA_name)
            rsp = await self._aask(prompt)
        elif name == "Doctor_Chief":
            prompt = self.PROMPT_TEMPLATE_CHIEF.format(context=context, name=name, collaboratorA_name=collaboratorA_name)
            rsp = await self._aask(prompt)

        return rsp

#Neuro Intermediate
class Neuro_Intermediate_Joint_Consultation(Action):

    PROMPT_TEMPLATE_INTERN: str = """
    ## ROLE DEFINITION
    Suppose you are {name} in a MICU Diagnostic Doctor Team which responsible for treating patients with serious symptoms due to medical diseases, you are collaborating with  {collaboratorA_name}to diagnose the patient based on patient's medical record.
    You are a young MICU doctor who focus on the diagnosis and treatment of internal medicine diseases with little clinical experience and are pone to discosure.
    ##  PATIENT'S MEDICAL RECORD
    {context}
        ## DUTY
    Now it's your turn, you need to provide preliminary disease diagnosis and current service plans based on the electronic medical records of emergency patients and guidance. Pay attention to the guidance，it is important to help you make a right diagnosis. In addition, you need to pay close attention to the recommendations and guidance of {collaboratorA_name}. If your diagnosis is wrong, you need to re-diagnose and propose a treatment plan based on the recommendations.
    The language should be refined and relevant to the disease diagnosis.
    
    return ```Diagnostic results: ```
    return ```Current_service results: ```

    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.
    """

    PROMPT_TEMPLATE_CHIEF: str = """
    ## Role definition
    Suppose you are {name} in the MICU diagnostic team, responsible for treating patients with severe symptoms due to illness. You are working with {collaboratorA_name}, mainly guiding interns to help her make the correct diagnosis and treatment plan.

    ## Patient's medical record
    {context}
    ## Responsibilities
    Now it is your turn. You should pay close attention to the latest diagnosis and treatment plan of your collaborator {collaboratorA_name}. If there is an error in his diagnostic results, you need to correct his mistakes or problems  and give advice and guidance based on Patient's medical record,.

    Return ```Is it true?: ```
    Return ```Guidance: ```

    Avoid continuous questions and answers, wait for the other party to respond before asking the next round of questions.
"""

    name: str = "Cooperation"


    async def run(self, context: str, name: str, collaboratorA_name: str):
        PMH_module = Correlation_calculation(str(context[0]))
        retrieval_result = Trainable_RAG(PMH_module)
        
        if name == "Doctor_Intern":
            prompt = self.PROMPT_TEMPLATE_INTERN.format(context=str(context[0]) + str(retrieval_result), name=name, collaboratorA_name=collaboratorA_name)
            rsp = await self._aask(prompt)
        elif name == "Doctor_Chief":
            prompt = self.PROMPT_TEMPLATE_CHIEF.format(context=context, name=name, collaboratorA_name=collaboratorA_name)
            rsp = await self._aask(prompt)

        return rsp


#TSICU
class TSICU_Joint_Consultation(Action):

    PROMPT_TEMPLATE_INTERN: str = """
    ## ROLE DEFINITION
    Suppose you are {name} in a MICU Diagnostic Doctor Team which responsible for treating patients with serious symptoms due to medical diseases, you are collaborating with  {collaboratorA_name}to diagnose the patient based on patient's medical record.
    You are a young MICU doctor who focus on the diagnosis and treatment of internal medicine diseases with little clinical experience and are pone to discosure.
    ##  PATIENT'S MEDICAL RECORD
    {context}
        ## DUTY
    Now it's your turn, you need to provide preliminary disease diagnosis and current service plans based on the electronic medical records of emergency patients and guidance. Pay attention to the guidance，it is important to help you make a right diagnosis. In addition, you need to pay close attention to the recommendations and guidance of {collaboratorA_name}. If your diagnosis is wrong, you need to re-diagnose and propose a treatment plan based on the recommendations.
    The language should be refined and relevant to the disease diagnosis.
    
    return ```Diagnostic results: ```
    return ```Current_service results: ```

    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.
    """

    PROMPT_TEMPLATE_CHIEF: str = """
    ## Role definition
    Suppose you are {name} in the MICU diagnostic team, responsible for treating patients with severe symptoms due to illness. You are working with {collaboratorA_name}, mainly guiding interns to help her make the correct diagnosis and treatment plan.

    ## Patient's medical record
    {context}
    ## Responsibilities
    Now it is your turn. You should pay close attention to the latest diagnosis and treatment plan of your collaborator {collaboratorA_name}. If there is an error in his diagnostic results, you need to correct his mistakes or problems  and give advice and guidance based on Patient's medical record,.

    Return ```Is it true?: ```
    Return ```Guidance: ```

    Avoid continuous questions and answers, wait for the other party to respond before asking the next round of questions.
"""

    name: str = "Cooperation"


    async def run(self, context: str, name: str, collaboratorA_name: str):
        PMH_module = Correlation_calculation(str(context[0]))
        retrieval_result = Trainable_RAG(PMH_module)
        
        if name == "Doctor_Intern":
            prompt = self.PROMPT_TEMPLATE_INTERN.format(context=str(context[0]) + str(retrieval_result), name=name, collaboratorA_name=collaboratorA_name)
            rsp = await self._aask(prompt)
        elif name == "Doctor_Chief":
            prompt = self.PROMPT_TEMPLATE_CHIEF.format(context=context, name=name, collaboratorA_name=collaboratorA_name)
            rsp = await self._aask(prompt)

        return rsp


#Neuro Stepdowm
class Neuro_Stepdown_Joint_Consultation(Action):

    PROMPT_TEMPLATE_INTERN: str = """
    ## ROLE DEFINITION
    Suppose you are {name} in a MICU Diagnostic Doctor Team which responsible for treating patients with serious symptoms due to medical diseases, you are collaborating with  {collaboratorA_name}to diagnose the patient based on patient's medical record.
    You are a young MICU doctor who focus on the diagnosis and treatment of internal medicine diseases with little clinical experience and are pone to discosure.
    ##  PATIENT'S MEDICAL RECORD
    {context}
        ## DUTY
    Now it's your turn, you need to provide preliminary disease diagnosis and current service plans based on the electronic medical records of emergency patients and guidance. Pay attention to the guidance，it is important to help you make a right diagnosis. In addition, you need to pay close attention to the recommendations and guidance of {collaboratorA_name}. If your diagnosis is wrong, you need to re-diagnose and propose a treatment plan based on the recommendations.
    The language should be refined and relevant to the disease diagnosis.
    
    return ```Diagnostic results: ```
    return ```Current_service results: ```

    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.
    """

    PROMPT_TEMPLATE_CHIEF: str = """
    ## Role definition
    Suppose you are {name} in the MICU diagnostic team, responsible for treating patients with severe symptoms due to illness. You are working with {collaboratorA_name}, mainly guiding interns to help her make the correct diagnosis and treatment plan.

    ## Patient's medical record
    {context}
    ## Responsibilities
    Now it is your turn. You should pay close attention to the latest diagnosis and treatment plan of your collaborator {collaboratorA_name}. If there is an error in his diagnostic results, you need to correct his mistakes or problems  and give advice and guidance based on Patient's medical record,.

    Return ```Is it true?: ```
    Return ```Guidance: ```

    Avoid continuous questions and answers, wait for the other party to respond before asking the next round of questions.
"""

    name: str = "Cooperation"


    async def run(self, context: str, name: str, collaboratorA_name: str):
        PMH_module = Correlation_calculation(str(context[0]))
        retrieval_result = Trainable_RAG(PMH_module)
        
        if name == "Doctor_Intern":
            prompt = self.PROMPT_TEMPLATE_INTERN.format(context=str(context[0]) + str(retrieval_result), name=name, collaboratorA_name=collaboratorA_name)
            rsp = await self._aask(prompt)
        elif name == "Doctor_Chief":
            prompt = self.PROMPT_TEMPLATE_CHIEF.format(context=context, name=name, collaboratorA_name=collaboratorA_name)
            rsp = await self._aask(prompt)

        return rsp

#Neuro SICU
class Neuro_SICU_Joint_Consultation(Action):

    PROMPT_TEMPLATE_INTERN: str = """
    ## ROLE DEFINITION
    Suppose you are {name} in a MICU Diagnostic Doctor Team which responsible for treating patients with serious symptoms due to medical diseases, you are collaborating with  {collaboratorA_name}to diagnose the patient based on patient's medical record.
    You are a young MICU doctor who focus on the diagnosis and treatment of internal medicine diseases with little clinical experience and are pone to discosure.
    ##  PATIENT'S MEDICAL RECORD
    {context}
        ## DUTY
    Now it's your turn, you need to provide preliminary disease diagnosis and current service plans based on the electronic medical records of emergency patients and guidance. Pay attention to the guidance，it is important to help you make a right diagnosis. In addition, you need to pay close attention to the recommendations and guidance of {collaboratorA_name}. If your diagnosis is wrong, you need to re-diagnose and propose a treatment plan based on the recommendations.
    The language should be refined and relevant to the disease diagnosis.
    
    return ```Diagnostic results: ```
    return ```Current_service results: ```

    Avoid continuous questions and answers and wait for the other party to respond before moving on to the next round of questions.
    """

    PROMPT_TEMPLATE_CHIEF: str = """
    ## Role definition
    Suppose you are {name} in the MICU diagnostic team, responsible for treating patients with severe symptoms due to illness. You are working with {collaboratorA_name}, mainly guiding interns to help her make the correct diagnosis and treatment plan.

    ## Patient's medical record
    {context}
    ## Responsibilities
    Now it is your turn. You should pay close attention to the latest diagnosis and treatment plan of your collaborator {collaboratorA_name}. If there is an error in his diagnostic results, you need to correct his mistakes or problems  and give advice and guidance based on Patient's medical record,.

    Return ```Is it true?: ```
    Return ```Guidance: ```

    Avoid continuous questions and answers, wait for the other party to respond before asking the next round of questions.
"""

    name: str = "Cooperation"


    async def run(self, context: str, name: str, collaboratorA_name: str):
        PMH_module = Correlation_calculation(str(context[0]))
        retrieval_result = Trainable_RAG(PMH_module)
        
        if name == "Doctor_Intern":
            prompt = self.PROMPT_TEMPLATE_INTERN.format(context=str(context[0]) + str(retrieval_result), name=name, collaboratorA_name=collaboratorA_name)
            rsp = await self._aask(prompt)
        elif name == "Doctor_Chief":
            prompt = self.PROMPT_TEMPLATE_CHIEF.format(context=context, name=name, collaboratorA_name=collaboratorA_name)
            rsp = await self._aask(prompt)

        return rsp


class Summary(Action):

    PROMPT_TEMPLATE: str = """
    ## ROLE DEFINITION
    Suppose you are Chief Physician, You need to determine the final summary based on the diagnosis and current service  discussed by several doctors on the bridge.
    ## HISTORY
    {context}
    ## DUTY
    Now it's your turn, You need to make final summary based on the diagnosis and treatment plans discussed by several doctors based on HISTORY.
    The summary outline should follow the following aspects:

    return ```Final Diagnostic Result: ```
    return ```Final Current Service: ```
    """

    name: str = "Review"

    async def run(self, context: str):
        prompt = self.PROMPT_TEMPLATE.format(context=context)
        logger.info(prompt)

        rsp = await self._aask(prompt)

        return rsp

__all__ = [
    'ClinicAssignment',
    'MICU_Joint_Consultation',
    'SICU_Joint_Consultation',
    'MICU_SICU_Joint_Consultation',
    'CVICU_Joint_Consultation',
    'CCU_Joint_Consultation',
    'Neuro_Intermediate_Joint_Consultation',
    'TSICU_Joint_Consultation',
    'Neuro_Stepdown_Joint_Consultation',
    'Neuro_SICU_Joint_Consultation',
    'Summary'
]
