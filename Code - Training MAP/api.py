# # import requests
# # def test_api():
# #     #  disease_test 
# #     try:
# #         response = requests.post(
# #             "http://127.0.0.1:6006/disease_test",
# #             json={"query": "Patient has fever and cough"},
# #             timeout=30
# #         )
# #         print("Disease Test Response:", response.json())
# #     except Exception as e:
# #         print(f"Disease test error: {e}")

# #     #  predict 
# #     try:
# #         messages = [
# #             {
# #                 "role": "system",
# #                 "content": "You are a medical diagnostician"
# #             },
# #             {
# #                 "role": "user",
# #                 "content": "I've been feeling dizzy lately."
# #             }
# #         ]
# #         response = requests.post(
# #             "http://127.0.0.1:6006/predict",
# #             json={"messages": messages},
# #             timeout=30
# #         )
# #         print("Predict Response:", response.json())
# #     except Exception as e:
# #         print(f"Predict error: {e}")

# # # test_api()

# # import requests
# # import random

# # # 
# # disease_query_path = "/root/capsule/code/examples/clinic/data/q_disease.txt"
# # curr_query_path = "/root/capsule/code/examples/clinic/data/q_plan.txt"

# # # Trainable_RAG 
# # def Trainable_RAG(Query):
# #     try:
# #         # 
# #         with open(disease_query_path, 'r') as file1:
# #             lines1 = file1.readlines()
# #         if not lines1:
# #             raise ValueError("disease_query_path file is empty")  #
# #         query1 = random.choice(lines1).strip()

# #         with open(curr_query_path, 'r') as file2:
# #             lines2 = file2.readlines()
# #         if not lines2:
# #             raise ValueError("curr_query_path file is empty")  # 
# #         query2 = random.choice(lines2).strip()

# #         # 
# #         Disease_query = Query + "\n" + query1
# #         disease_response = requests.post(
# #             "http://127.0.0.1:6006/disease_test",
# #             json={"query": Disease_query},
# #             timeout=30
# #         )
# #         print("Disease Test Full Response:", disease_response.text)  # 

# #         if disease_response.status_code == 200:
# #             # 
# #             disease_result = disease_response.json().get("response", "No response from disease_test")
# #         else:
# #             disease_result = f"Error: {disease_response.status_code}, {disease_response.text}"

# #         # 
# #         Curr_query = Query + "\n" + query2
# #         messages_curr = [
# #             {"role": "system", "content": '''You are a clinic expert...'''},
# #             {"role": "user", "content": Curr_query}
# #         ]
# #         predict_response = requests.post(
# #             "http://127.0.0.1:6006/predict",
# #             json={"messages": messages_curr},
# #             timeout=30
# #         )
# #         print("Predict Full Response:", predict_response.text)  # 

# #         if predict_response.status_code == 200:
# #             # 
# #             curr_service_result = predict_response.json().get("response", "No response from predict")
# #         else:
# #             curr_service_result = f"Error: {predict_response.status_code}, {predict_response.text}"

# #         # 
# #         Disease = "Diagnostic results:  " + disease_result
# #         Curr = "Current_service results:  " + curr_service_result

# #         return Disease, Curr

# #     except Exception as e:
# #         print(f"Error in Trainable_RAG: {e}")
# #         return "Error in Disease query", "Error in Curr_service query"


# # # 
# # if __name__ == "__main__":
# #     # 
# #     with open(disease_query_path, 'w') as f:
# #         f.write("Patient has fever and cough\nPatient feels shortness of breath\n")
# #     with open(curr_query_path, 'w') as f:
# #         f.write("OMED\nORTHO\n")

# #     # 
# #     Query = "Patient is experiencing severe headache and nausea"

# #     # 
# #     Disease, Curr = Trainable_RAG(Query)
# #     print(Disease)
# #     print(Curr)

# import requests
# import requests

# # 
# def api_request(query, system_content, endpoint):
#     url = f'http://127.0.0.1:6006/{endpoint}'  # 
#     headers = {
#         'accept': 'application/json',
#         'Content-Type': 'application/json'
#     }
    
#     # 
#     messages = [
#         {"role": "system", "content": system_content},
#         {"role": "user", "content": query}
#     ]
    
#     # 
#     data = {
#         "model": "internlm2",  # 
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
#         # 
#         response = requests.post(url, headers=headers, json=data)
#         response.raise_for_status()  # 
#         response_json = response.json()
#         # 
#         return response_json.get('response', 'No response received')
#     except Exception as e:
#         print(f"Error in API request: {e}")
#         return "Error: Unable to process the request"

# # 
# def disease_test(query):
#     endpoint = "disease_test"  # 
#     system_content = '''You are a clinic expert and need to make a diagnose for possible diseases from the candidate disease list based on the patient's medical record information. The candidate disease list is as follows:["Certain infectious and parasitic diseases","Neoplasms","Endocrine, nutritional and metabolic Diseases","Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism","Mental and behavioural disorders","Diseases of the nervous system and sense organs","Diseases of the circulatory system","Diseases of the respiratory system","Diseases of the digestive system","Diseases of the genitourinary system","Pregnancy, childbirth and the puerperium","Diseases of the skin and subcutaneous tissue","Diseases of the musculoskeletal system and connective tissue","Congenital malformations, deformations and chromosomal abnormalities","Certain conditions originating in the perinatal period","Symptoms, signs and abnormal clinical and laboratory findings","Injury, poisoning and certain other consequences of external causes"]'''
#     return api_request(query, system_content, endpoint)

# # 
# def care_test(query):
#     endpoint = "predict"  #
#     system_content = '''You are a doctor's assistant and need to determine which first_unit to go to based on the patient's medical condition. The list of candidate first_careunits is as follows: ['Medical Intensive Care Unit (MICU)', 'Surgical Intensive Care Unit (SICU) )', 'Surgical Intensive Care Unit (MICU/SICU)', 'Cardiac Vascular Intensive Care Unit (CVICU)', 'Coronary Care Unit (CCU)', 'Neuro Intermediate', 'Trauma SICU (TSICU)', 'Neuro Stepdown', 'Neuro Surgical Intensive Care Unit (Neuro SICU)']'''
#     return api_request(query, system_content, endpoint)

# # 
# def curr_test(query):
#     endpoint = "predict"  # 
#     system_content = '''You are a clinic expert and need to find one the most suitable current service from the candidate curr_service list based on the patient's medical record. The candidate curr_service list is as follows:['OMED', 'ORTHO', 'TSURG', 'OBS', 'TRAUM', 'NSURG', 'MED', 'CSURG', 'GU', 'ENT', 'CMED', 'VSURG', 'SURG', 'NMED', 'GYN', 'PSURG']'''
#     return api_request(query, system_content, endpoint)

# if __name__ == "__main__":
#     #  disease_test
#     disease_query = "Patient has severe headache and fever."
#     print("Disease Test Result:", disease_test(disease_query))

#     #  care_test
#     care_query = "Patient is experiencing shortness of breath."
#     print("Care Test Result:", care_test(care_query))

#     #  curr_test
#     curr_query = "Patient needs urgent orthopedic care."
#     print("Current Service Test Result:", curr_test(curr_query))

import requests

# disease_test 
def disease_test(query):
    url = "http://127.0.0.1:6006/disease_test"
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/json'
    }
    data = {"query": query}  # 

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # 
        response_json = response.json()
        return response_json.get('response', 'No response received')
    except Exception as e:
        print(f"Error in disease_test request: {e}")
        return "Error: Unable to process the request"

# predict 
def api_request(query, system_content, endpoint):
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

    try:
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()
        response_json = response.json()
        return response_json.get('response', 'No response received')
    except Exception as e:
        print(f"Error in API request: {e}")
        return "Error: Unable to process the request"

# care_test 
def care_test(query):
    endpoint = "predict"
    system_content = '''You are a doctor's assistant and need to determine which first_unit to go to based on the patient's medical condition. The list of candidate first_careunits is as follows: ['Medical Intensive Care Unit (MICU)', 'Surgical Intensive Care Unit (SICU) )', 'Surgical Intensive Care Unit (MICU/SICU)', 'Cardiac Vascular Intensive Care Unit (CVICU)', 'Coronary Care Unit (CCU)', 'Neuro Intermediate', 'Trauma SICU (TSICU)', 'Neuro Stepdown', 'Neuro Surgical Intensive Care Unit (Neuro SICU)']'''
    return api_request(query, system_content, endpoint)

# curr_test 
def curr_test(query):
    endpoint = "predict"
    system_content = '''You are a clinic expert and need to find one the most suitable current service from the candidate curr_service list based on the patient's medical record. The candidate curr_service list is as follows:['OMED', 'ORTHO', 'TSURG', 'OBS', 'TRAUM', 'NSURG', 'MED', 'CSURG', 'GU', 'ENT', 'CMED', 'VSURG', 'SURG', 'NMED', 'GYN', 'PSURG']'''
    return api_request(query, system_content, endpoint)

if __name__ == "__main__":
    #  disease_test
    disease_query = "Patient has severe headache and fever."
    print("Disease Test Result:", disease_test(disease_query))

    #  care_test
    care_query = "Patient is experiencing shortness of breath."
    print("Care Test Result:", care_test(care_query))

    #  curr_test
    curr_query = "Patient needs urgent orthopedic care."
    print("Current Service Test Result:", curr_test(curr_query))