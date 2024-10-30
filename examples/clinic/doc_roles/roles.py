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
# os.environ["CUDA_VISIBLE_DEVICES"] = "3"
from doc_actions.actions import *

disease_query_path = "./data/q_disease.txt"
curr_query_path= "./clinic/data/q_plan.txt"
sub_memory_path = "./clinic/data/sub_memory.json"
dynamic_memory_path = "./data/dynamic_memory.json"
static_memory_path = "./data/static_memory.json"

record = '''
Medical Reocrd:
    Language:English
    Gender:Male
    Marital status:Single
    Race:White
    Radiology Report:EXAMINATION:  Chest radiographINDICATION:  Dyspnea.TECHNIQUE:  AP and lateral views of the chest.COMPARISON:  ___FINDINGS: Heart size is mildly enlarged.  There is mild unfolding of the thoracic aorta.Cardiomediastinal silhouette and hilar contours are otherwise unremarkable. There is mild bibasilar atelectasis.  Lungs are otherwise clear.  Pleuralsurfaces are clear without effusion or pneumothorax.  Focus of air seen underthe right hemidiaphragm, likely represents colonic interposition.IMPRESSION: No acute cardiopulmonary abnormality.
    Past Medical History:Cardiac History  -Pericarditis, as above.-Hypertension.-Dyslipidemia.   Other PMH  -Rheumatoid arthritis. -Remote traumatic DVT.-Cholecystectomy.-Appendectomy.-Tonsillectomy.-Left wrist reconstruction.-Right rotator cuff reconstruction. 
'''

class ClinicDoctor(Role):
    name: str = "PlanMaker"
    profile: str = "Plan"

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([ClinicAssignment])
        self._watch([UserRequirement])

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

class SICU_Doctor(Role):
    name: str = ""
    profile: str = ""
    collaboratorA_name: str = ""
    collaboratorB_name: str = ""
    store: Optional[object] = Field(default=None, exclude=True)  # must inplement tools.SearchInterface

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([SICU_Joint_Consultation])
        self._watch([SICU_Joint_Consultation])

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

class MICU_SICU_Doctor(Role):
    name: str = ""
    profile: str = ""
    collaboratorA_name: str = ""
    collaboratorB_name: str = ""
    store: Optional[object] = Field(default=None, exclude=True)  # must inplement tools.SearchInterface

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([MICU_SICU_Joint_Consultation])
        self._watch([MICU_SICU_Joint_Consultation])

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

class CVICU_Doctor(Role):
    name: str = ""
    profile: str = ""
    collaboratorA_name: str = ""
    store: Optional[object] = Field(default=None, exclude=True)  # must inplement tools.SearchInterface

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([CVICU_Joint_Consultation])
        self._watch([CVICU_Joint_Consultation])

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

class CCU_Doctor(Role):
    name: str = ""
    profile: str = ""
    collaboratorA_name: str = ""
    store: Optional[object] = Field(default=None, exclude=True)  # must inplement tools.SearchInterface

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([CCU_Joint_Consultation])
        self._watch([CCU_Joint_Consultation])

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

class TSICU_Doctor(Role):
    name: str = ""
    profile: str = ""
    collaboratorA_name: str = ""
    store: Optional[object] = Field(default=None, exclude=True)  # must inplement tools.SearchInterface

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([TSICU_Joint_Consultation])
        self._watch([TSICU_Joint_Consultation])

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

class Neuro_Intermediate_Doctor(Role):
    name: str = ""
    profile: str = ""
    collaboratorA_name: str = ""
    store: Optional[object] = Field(default=None, exclude=True)  # must inplement tools.SearchInterface

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([Neuro_Intermediate_Joint_Consultation])
        self._watch([Neuro_Intermediate_Joint_Consultation])

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

class Neuro_Stepdown_Doctor(Role):
    name: str = ""
    profile: str = ""
    collaboratorA_name: str = ""
    store: Optional[object] = Field(default=None, exclude=True)  # must inplement tools.SearchInterface

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([Neuro_Stepdown_Joint_Consultation])
        self._watch([Neuro_Stepdown_Joint_Consultation])

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

class Neuro_SICU_Doctor(Role):
    name: str = ""
    profile: str = ""
    collaboratorA_name: str = ""
    store: Optional[object] = Field(default=None, exclude=True)  # must inplement tools.SearchInterface

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([Neuro_SICU_Joint_Consultation])
        self._watch([Neuro_SICU_Joint_Consultation])

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

class Chief_Physician(Role):
    name: str = ""
    profile: str = ""
    collaborator_name: str = ""

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.set_actions([Summary])
        self._watch([UserRequirement])


    async def _act(self) -> Message:

        todo = self.rc.todo  
        # 读取JSON文件
        with open(dynamic_memory_path, 'r') as file:
            data1 = json.load(file)
        # result = [f"{item['role']}:{item['content']}" for item in data]
        

        # 打印结果,每句前加上序号
        # for i, line in enumerate(Doctor_Chief_lines, start=1):
        #     print(f"{i}. {line}")
        with open(sub_memory_path, 'r') as file:
            data = json.load(file)    
        for i in data:
            text = i
            # text = "Diagnostic Department: Gastroenterology, Internal Medicine\n\n"
            start = text.find("Diagnostic Department:") + len("Diagnostic Department:")
            end = text.find("\n")
            departments = text[start:end].strip()
            department_list = [dept.strip() for dept in departments.split(",")]
            # print(f" 777777777777777{department_list[1]}")
        # for item in department_list:
        # 打印结果,每句前加上序号
        times = len(department_list)
        context = ""
        Doctor_Chief_lines = [line["content"] for line in data1[-5:] if line["role"] == "Doctor_Chief" and line["content"]]
        # for i, line in enumerate(Doctor_Chief_lines, start=0):
        #     # print(line)
        #     result = f"{department_list[i]}'s final results of the consultation are as follows:{line}"
        # # 将列表元素连接为一个字符串
        #     context = ''.join(result)
    
        rsp = await todo.run(context=context)

        msg = Message(
            content=rsp,
            role=self.profile,
            cause_by=type(todo),
            sent_from=self.name,
            send_to=self.collaborator_name,
        )
        self.rc.memory.add(msg)

        with open(static_memory_path, 'r') as f:
            content = json.load(f)   
        new_item = {
        "unique_id": int(datetime.now().timestamp()),
        "medical_record":record,
        "department":department_list,
        "final_diagnostic_result":rsp.split("```Final Diagnostic Result:")[1].split("```Final Current Service:")[0].strip(),
        "final_tcurrent_service":rsp.split("```Final Current Service:")[1].strip()
    }
        content.append(new_item)
        with open(static_memory_path, "w") as f:
            json.dump(content, f, indent=2)  
               
        return msg


__all__ = [
    'ClinicDoctor',
    'MICU_Doctor',
    'SICU_Doctor',
    'MICU_SICU_Doctor',
    'CVICU_Doctor',
    'CCU_Doctor',
    'TSICU_Doctor',
    'Neuro_Intermediate_Doctor',
    'Neuro_Stepdown_Doctor',
    'Neuro_SICU_Doctor',
    'Chief_Physician'
]
