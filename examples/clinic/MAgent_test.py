"""
Filename:MAgent.py
Created Date: Tuesday, Oct 27th 2024, 6:52:25 pm
Author: XiaoQi
"""
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
import sys
from pathlib import Path
from doc_actions.actions import *
from doc_roles.roles import *

CURRENT_DIR = Path(__file__).parent.absolute()
sys.path.append(str(CURRENT_DIR))
#Set memory_base path
disease_query_path = "./data/q_disease.txt"
curr_query_path= "./clinic/data/q_plan.txt"
sub_memory_path = "./clinic/data/sub_memory.json"
dynamic_memory_path = "./data/dynamic_memory.json"
static_memory_path = "./data/static_memory.json"
model_id = "/home/xusheng_liang/cair/xtuner-main/RAG/LLaMA3_8B_Instruct_10_08"
#Set up the patient record in the following format:
#     template = f'''
# Medical Record:
# {personal_info}
    
# Radiology Report:
# {radiology_report}
    
# Past Medical History:
# {medical_history}
# '''

#example:
record = '''
Medical Reocrd:
    Language:English
    Gender:Male
    Marital status:Single
    Race:White
    Radiology Report:EXAMINATION:  Chest radiographINDICATION:  Dyspnea.TECHNIQUE:  AP and lateral views of the chest.COMPARISON:  ___FINDINGS: Heart size is mildly enlarged.  There is mild unfolding of the thoracic aorta.Cardiomediastinal silhouette and hilar contours are otherwise unremarkable. There is mild bibasilar atelectasis.  Lungs are otherwise clear.  Pleuralsurfaces are clear without effusion or pneumothorax.  Focus of air seen underthe right hemidiaphragm, likely represents colonic interposition.IMPRESSION: No acute cardiopulmonary abnormality.
    Past Medical History:Cardiac History  -Pericarditis, as above.-Hypertension.-Dyslipidemia.   Other PMH  -Rheumatoid arthritis. -Remote traumatic DVT.-Cholecystectomy.-Appendectomy.-Tonsillectomy.-Left wrist reconstruction.-Right rotator cuff reconstruction. 
'''

async def Assignment_department(record: str, investment: float = 3.0, n_round: int = 1):

    clinicdoctor = ClinicDoctor(name="ClinicDoctor", profile="Assign")

    team = Team()
    team.hire([clinicdoctor])
    team.invest(investment)   
    team.run_project(record, send_to="ClinicDoctor")  
    await team.run(n_round=n_round)

class MICU_Doctor(Role):
    name: str = ""
    profile: str = ""
    collaboratorA_name: str = ""
    store: Optional[object] = Field(default=None, exclude=True)  # must inplement tools.SearchInterface

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([MICU_Joint_Consultation])
        self._watch([MICU_Joint_Consultation])

    async def _observe(self) -> int:
        await super()._observe()

        if self.name == "Doctor_Intern":
            self.rc.news =   [msg for msg in self.rc.news if msg.send_to == "Doctor_Chief" or "Doctor_Wang"]
        elif self.name == "Doctor_Chief":
            self.rc.news =   [msg for msg in self.rc.news if msg.send_to == "Doctor_Intern" or "Doctor_Wang"]

        return len(self.rc.news)

    async def _act(self) -> Message:
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")
        todo = self.rc.todo  
        memories = self.get_memories()
        rsp = await todo.run(context=memories, name=self.name, collaboratorA_name=self.collaboratorA_name)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.collaboratorA_name,
        )
        file_path = dynamic_memory_path
        with open(file_path, "r") as json_file:

            existing_data = json.load(json_file)
        # list.append(msg.to_dict())
        existing_data.append(msg.to_dict())

        with open(file_path, "w") as f:
            json.dump(existing_data, f, indent=2)                
        self.rc.memory.add(msg)

        return msg

async def MICU(record: str, investment: float = 3.0, n_round: int = 20):

    Doctor_Intern = MICU_Doctor(name="Doctor_Intern", profile="Doctor_Intern", collaboratorA_name="Doctor_Chief")
    Doctor_Chief = MICU_Doctor(name="Doctor_Chief", profile="Doctor_Chief", collaboratorA_name="Doctor_Intern")

    team = Team()
    team.hire([Doctor_Intern, Doctor_Chief])
    team.invest(investment)

    team.run_project(record, send_to="Doctor_Intern")  # send debate topic to Biden and let him speak first
    print("MICU joint consultation begin:")
    await team.run(n_round=n_round)
    print("MICU joint consultation end:")

async def SICU(record: str, investment: float = 3.0, n_round: int = 20):

    Doctor_Intern = SICU_Doctor(name="Doctor_Intern", profile="Doctor_Intern", collaboratorA_name="Doctor_Chief")
    Doctor_Chief = SICU_Doctor(name="Doctor_Chief", profile="Doctor_Chief", collaboratorA_name="Doctor_Intern")

    team = Team()
    team.hire([Doctor_Intern, Doctor_Chief])
    team.invest(investment)

    team.run_project(record, send_to="Doctor_Intern") 
    print("SICU joint consultation begin:")
    await team.run(n_round=n_round)
    print("SICU Medicine joint consultation end:")

async def MICU_SICU(record: str, investment: float = 3.0, n_round: int = 20):

    Doctor_Intern = MICU_SICU_Doctor(name="Doctor_Intern", profile="Doctor_Intern", collaboratorA_name="Doctor_Chief",collaboratorB_name="Doctor_Wang")
    Doctor_Chief = MICU_SICU_Doctor(name="Doctor_Chief", profile="Doctor_Chief", collaboratorA_name="Doctor_Intern",collaboratorB_name="Doctor_Wang")

    
    team = Team()
    team.hire([Doctor_Intern, Doctor_Chief])
    team.invest(investment)

    team.run_project(record, send_to="Doctor_Intern")  # send debate topic to Biden and let him speak first
    print("MICU&SICU joint consultation begin:")
    await team.run(n_round=n_round)
    print("MICU&SICU Medicine joint consultation end:")

async def CVICU(record: str, investment: float = 3.0, n_round: int = 20):

    Doctor_Intern = CVICU_Doctor(name="Doctor_Intern", profile="Doctor_Intern", collaboratorA_name="Doctor_Chief")
    Doctor_Chief = CVICU_Doctor(name="Doctor_Chief", profile="Doctor_Chief", collaboratorA_name="Doctor_Intern")

    team = Team()
    team.hire([Doctor_Intern, Doctor_Chief])
    team.invest(investment)

    team.run_project(record, send_to="Doctor_Intern")  # send debate topic to Biden and let him speak first
    print("CVICU joint consultation begin:")
    await team.run(n_round=n_round)
    print("CVICU Medicine joint consultation end:")

async def CCU(record: str, investment: float = 3.0, n_round: int = 20):

    Doctor_Intern = CCU_Doctor(name="Doctor_Intern", profile="Doctor_Intern", collaboratorA_name="Doctor_Chief")
    Doctor_Chief = CCU_Doctor(name="Doctor_Chief", profile="Doctor_Chief", collaboratorA_name="Doctor_Intern")

    team = Team()
    team.hire([Doctor_Intern, Doctor_Chief])
    team.invest(investment)

    team.run_project(record, send_to="Doctor_Intern")  # send debate topic to Biden and let him speak first
    print("CCU joint consultation begin:")
    await team.run(n_round=n_round)
    print("CCU joint consultation end:")

async def Neuro_Intermediate(record: str, investment: float = 3.0, n_round: int = 20):

    Doctor_Intern = Neuro_Intermediate_Doctor(name="Doctor_Intern", profile="Doctor_Intern", collaboratorA_name="Doctor_Chief")
    Doctor_Chief = Neuro_Intermediate_Doctor(name="Doctor_Chief", profile="Doctor_Chief", collaboratorA_name="Doctor_Intern")

    team = Team()
    team.hire([Doctor_Intern, Doctor_Chief])
    team.invest(investment)

    team.run_project(record, send_to="Doctor_Intern")  # send debate topic to Biden and let him speak first
    print("Neuro Intermediate joint consultation begin:")
    await team.run(n_round=n_round)
    print("Neuro Intermediate Medicine joint consultation end:")


async def TSICU(record: str, investment: float = 3.0, n_round: int = 20):

    Doctor_Intern = TSICU_Doctor(name="Doctor_Intern", profile="Doctor_Intern", collaboratorA_name="Doctor_Chief")
    Doctor_Chief = TSICU_Doctor(name="Doctor_Chief", profile="Doctor_Chief", collaboratorA_name="Doctor_Intern")

    team = Team()
    team.hire([Doctor_Intern, Doctor_Chief])
    team.invest(investment)

    team.run_project(record, send_to="Doctor_Intern")  # send debate topic to Biden and let him speak first
    print("TSICU joint consultation begin:")
    await team.run(n_round=n_round)
    print("TSICU Medicine joint consultation end:")

async def Neuro_Stepdown(record: str, investment: float = 3.0, n_round: int = 20):

    Doctor_Intern = Neuro_Stepdown_Doctor(name="Doctor_Intern", profile="Doctor_Intern", collaboratorA_name="Doctor_Chief")
    Doctor_Chief = Neuro_Stepdown_Doctor(name="Doctor_Chief", profile="Doctor_Chief", collaboratorA_name="Doctor_Intern")

    team = Team()
    team.hire([Doctor_Intern, Doctor_Chief])
    team.invest(investment)

    team.run_project(record, send_to="Doctor_Intern")  # send debate topic to Biden and let him speak first
    print("Neuro_Stepdown joint consultation begin:")
    await team.run(n_round=n_round)
    print("Neuro_Stepdown Medicine joint consultation end:")

async def Neuro_SICU(record: str, investment: float = 3.0, n_round: int = 20):

    Doctor_Intern = Neuro_SICU_Doctor(name="Doctor_Intern", profile="Doctor_Intern", collaboratorA_name="Doctor_Chief",collaboratorB_name="Doctor_Wang")
    Doctor_Chief = Neuro_SICU_Doctor(name="Doctor_Chief", profile="Doctor_Chief", collaboratorA_name="Doctor_Intern",collaboratorB_name="Doctor_Wang")
    
    team = Team()
    team.hire([Doctor_Intern, Doctor_Chief])
    team.invest(investment)

    team.run_project(record, send_to="Doctor_Intern")  # send debate topic to Biden and let him speak first
    print("Neuro_SICU joint consultation begin:")
    await team.run(n_round=n_round)
    print("Neuro_SICU Medicine joint consultation end:")

async def FinalDecison(record: str, investment: float = str, n_round: int = 1):

    chief_physician = Chief_Physician(name="ChiefPhysician", profile="decision")

    team = Team()
    team.hire([chief_physician])
    team.invest(investment)   
    team.run_project(record, send_to="ChiefPhysician")  
    await team.run(n_round=n_round)

def department():

    department_mapping = {
        "MICU": ["Medical Intensive Care Unit", "Medical Intensive Care Unit (MICU)", "MICU"],
        "SICU": ["Surgical Intensive Care Unit", "Surgical Intensive Care Unit (SICU)", "SICU"],
        "MICU/SICU": ["MICU/SICU", "Surgical Intensive Care Unit (MICU/SICU)"],
        "CVICU": ["Cardiac Vascular Intensive Care Unit", "Cardiac Vascular Intensive Care Unit (CVICU)", "CVICU"],
        "CCU": ["Coronary Care Unit", "CCU"],
        "Neuro Intermediate": ["Neuro Intermediate"],
        "TSICU": ["Trauma SICU", "TSICU"],
        "Neuro Stepdown": ["Neuro Stepdown"],
        "Neuro SICU": ["Neuro Surgical Intensive Care Unit", "Neuro SICU"]
    }
    
    try:
        with open(sub_memory_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        if not data:
            print("Warning: No data found in memory file")
            return []
            
        text = data[-1]
        detected_departments = []
        
        for standard_name, variations in department_mapping.items():
            for variant in variations:
                if variant in text:
                    detected_departments.append(standard_name)
                    break 
                    
        if not detected_departments:
            print("Warning: No valid departments found in text")
            
        return detected_departments
        
    except FileNotFoundError:
        print(f"Error: Memory file not found at {sub_memory_path}")
        return []
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in memory file")
        return []
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return []

async def clinic_system(record:str, investment:int=3.0, n_round:int=2):
    print("Prepare assignment department:")
    await Assignment_department(record, investment, 3)
    department_list = department()
    print(f"department:{department_list}")

    for item in department_list:
        if item == "Medical Intensive Care Unit" or "Medical Intensive Care Unit (MICU)":
            await MICU(record,investment, n_round)
        elif item == "Surgical Intensive Care Unit" or "Surgical Intensive Care Unit (SICU)":
            await SICU(record, investment, n_round)
        elif item == "Surgical Intensive Care Unit (MICU/SICU)":
            await MICU_SICU(record, investment, n_round)
        elif item == "Cardiac Vascular Intensive Care Unit" or "Cardiac Vascular Intensive Care Unit (CVICU)":
            await CVICU(record, investment, n_round)
        elif item == "Coronary Care Unit" or "CCU":      
            await CCU(record, investment, n_round)  
        elif item == "Neuro Intermediate":    
            await Neuro_Intermediate(record, investment, n_round)
        elif item == "Trauma SICU" or "TSICU":     
            await TSICU(record, investment, n_round)
        elif item == "Neuro Stepdown":     
            await Neuro_Stepdown(record, investment, n_round)
        elif item == "Neuro Surgical Intensive Care Unit" or "Neuro SICU":    
            await Neuro_SICU(record, investment, n_round)
        else:
            print("No department has been selected yet!")
    file_path = dynamic_memory_path

    with open(file_path, 'r') as file:
        summary = json.load(file)
    summary = summary[-1]["role"] +"'s final diagnostic:" + summary[-1]["content"]
    await FinalDecison(str(summary), investment, 1)

    list = []
    with open(file_path, "w") as json_file:
        json.dump(list, json_file)

def main(record:str = record,investment: float = 3.0, n_round: int = 3):
    if platform.system() == "Windows":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(clinic_system(record, investment, n_round),)


if __name__ == "__main__":
    fire.Fire(main)  




