from dataclasses import dataclass
import inspect
import importlib
import importlib.util
import json
import os
import sys

from pathlib import Path
from time import process_time, sleep
from typing import Iterable, List

from chortest.cfsm import (
    CFSM,
    CommunicatingSystem,
    InTransitionLabel,
    OutTransitionLabel,
    State,
    TransitionLabel,
)

from chortest.parsing import Parsers
from chortest.session import TestMachine
from jinja2 import Template
from shutil import copyfile


@dataclass
class MetaType:
    pass

@dataclass()
class MetaSend(MetaType):
    target_part: str
    msg_type: str
    part_name: str
    target_state: str

    def __str__(self) -> str:
        return f"Send[{self.target_part}, {self.msg_type}, 'T_{self.part_name}_{self.target_state}']"

        
@dataclass()
class MetaRecv(MetaType):
    target_part: str
    msg_type: str
    part_name: str
    target_state: str

    def __str__(self) -> str:
        return f"Recv[{self.target_part}, {self.msg_type}, 'T_{self.part_name}_{self.target_state}']"

@dataclass()
class MetaEps(MetaType):
    def __str__(self) -> str:
        return "Eps"

@dataclass()
class MetaChoose(MetaType):
    target_part: str
    msg_type: str
    part_name: str
    target_state: str
    rec_type: MetaType

    def __str__(self) -> str:
        return f"Choose[{self.target_part}, {self.msg_type}, 'T_{self.part_name}_{self.target_state}', {self.rec_type}]"

@dataclass()
class MetaOffer(MetaType):
    target_part: str
    msg_type: str
    part_name: str
    target_state: str
    rec_type: MetaType

    def __str__(self) -> str:
        return f"Offer[{self.target_part}, {self.msg_type}, 'T_{self.part_name}_{self.target_state}', {self.rec_type}]"

@dataclass
class TypeDefinition:
    participant: str
    state: str
    type_expression: MetaType

    def output_method(self) -> str:
        match self.type_expression:
            case MetaSend(target_part, msg_type, part_name, target_state):
                return f"""
    async def _{self.state}_{msg_type.lower()}_send_{target_state}(self):
        assert isinstance(self.protocol, Send)
        assert self.state == {self.state}
        _msg = self.data.draw(st.builds({msg_type}))
        await self.protocol.send(_msg)
        self.state = {target_state}
        """
            case MetaRecv(target_part, msg_type, part_name, target_state):
                return f"""
    async def _{self.state}_{msg_type.lower()}_recv_{target_state}(self):
        assert isinstance(self.protocol, Recv)
        assert self.state == {self.state}
        self.protocol, _msg = await self.protocol.recv()
        assert isinstance(_msg, {msg_type})
        self.state = {target_state}
        """
            case MetaEps():
                return f"""
    async def _{self.state}_eps(self):
        assert isinstance(self.protocol, Eps)
        assert self.state == {self.state}
        self.protocol.close()
        """
            case MetaChoose(target_part, msg_type, part_name, target_state, rec_type):
                return f"""
    async def _{self.state}_{msg_type.lower()}_choice_pick_{target_state}(self):
        assert isinstance(self.protocol, Choose)
        _msg = self.data.draw(st.builds({msg_type}))
        await self.protocol.pick(_msg)
        self.state = {target_state}

    async def _{self.state}_{msg_type.lower()}_choice_skip_{self.state}(self):
        assert isinstance(self.protocol, Choose)
        self.protocol.skip()
{TypeDefinition(self.participant, self.state, rec_type).output_method()}
        """
            case MetaOffer(target_part, msg_type, part_name, target_state, rec_type):
                return f"""
    async def _{self.state}_{msg_type.lower()}_offer_recv_{target_state}(self):
        assert isinstance(self.protocol, Offer)
        assert self.state == {self.state}
        reply = await self.protocol.offer()
        match reply:
            case Left(c, _msg):
                assert isinstance(_msg, {msg_type})
        self.state = {target_state}

    async def _{self.state}_{msg_type.lower()}_offer_skip_{self.state}(self):
        assert isinstance(self.protocol, Offer)
        assert self.state == {self.state}
        reply = await self.protocol.offer()
        match reply:
            case Right(c):
                pass
{TypeDefinition(self.participant, self.state, rec_type).output_method()}
        """
            case _:
                raise Exception("Unknown type")


def gen_type_rec(
        state: str, 
        m: CFSM, 
        p_name: str, 
        trs: list[tuple[TransitionLabel, str]] | None = None
    ) -> MetaType:

    if trs is None:
        trs = list(m.transitions[state].items())
    if len(trs) == 0:
        return MetaEps()
    elif len(trs) == 1:
        k,ts = trs[0]
        if isinstance(k, OutTransitionLabel):
            return MetaSend(k.B, k.m, p_name, ts)
        else:
            return MetaRecv(k.A, k.m, p_name, ts)
    else:
        # trs = sorted(trs, key=lambda x: x[0].m)
        tr, target_state = trs[0]
        rec_type = gen_type_rec(state, m, p_name, trs[1:])
        if isinstance(tr, OutTransitionLabel):
            return MetaChoose(tr.B, tr.m, p_name, target_state, rec_type)
        else:
            return MetaOffer(tr.A, tr.m, p_name, target_state, rec_type)

def get_type_defs(cfsm: CommunicatingSystem) -> List[TypeDefinition]:
    ret = []
    for p in cfsm.machines.keys():
        for state in sorted(cfsm.machines[p].states):
            type_expression = gen_type_rec(state, cfsm.machines[p], p)
            ret.append(TypeDefinition(p, state, type_expression))
    return ret


def types_to_string(types: List[TypeDefinition]) -> str:
    ret = ""
    current_participant = ""
    for type_def in types:
        if type_def.participant != current_participant:
            current_participant = type_def.participant
            ret += f"# Types for participant {current_participant}\n"
        ret += f"T_{type_def.participant}_{type_def.state} = {str(type_def.type_expression)}\n"
    return ret

def is_output(t: TransitionLabel) -> bool:
    return isinstance(t, OutTransitionLabel)

def is_input(t: TransitionLabel) -> bool:
    return isinstance(t, InTransitionLabel)

def accept_states(cfsm: CommunicatingSystem) -> Iterable[State]:
    for p,m in cfsm.machines.items():
        for s in m.default_oracle():
            yield s

@dataclass()
class TemplateInfo():
    name: str
    always_replace: bool = False

def main(model_path: str, watch: bool = False):
    model_name = Path(model_path).stem
    # check that the model is an .fsa file 
    if not model_path.endswith(".fsa"):
        raise Exception(f"Model file {model_path} must be an .fsa file")
    cfsm: CommunicatingSystem = Parsers.parseFSA(str(model_path))
    while True: 
        try:
            output_path = generate_concrete_tests(model_name, model_path, cfsm)
            if not watch:
                print(f"Generated tests in {str(output_path)}")
                break
            else:
                print(f'\033[KModels output at {str(output_path)}. \r', end='')
            sleep(1.)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")
            sleep(1.)

def generate_concrete_tests(model_name:str, model_path: str, cfsm: CommunicatingSystem) -> Path:
    output_path = Path(model_path).parent / f"{model_name}_code"
    output_path.mkdir(parents=True, exist_ok=True)

    # measure start time
    start_time: float = process_time()

    all_types: List[TypeDefinition] = get_type_defs(cfsm)
    kwargs = {
                "cfsm": cfsm, 
                "messages": cfsm.all_message_types(), 
                "participants": cfsm.all_participant_types(),
                "allTypesStr": types_to_string(all_types),
                "allTypes": all_types,
                "is_output": is_output,
                "is_input": is_input,
                "accept_states": accept_states,
                "output_method": TypeDefinition.output_method,
                "model_name": model_name
            }

    templates = [
                TemplateInfo('model_implementations', False),
                TemplateInfo('model_session_types', True),
                TemplateInfo('model_tests', True),
                TemplateInfo('model_test_machines', True),
                TemplateInfo('model_types', False),
            ]

    base_path: Path = Path(__file__).parent / Path('templates/codegen')

    conftest_template = Template((base_path / 'conftest.py.jinja2').read_text())
    conftest_output = conftest_template.render(model_name=model_name)
    (output_path / 'conftest.py').write_text(str(conftest_output))
    (output_path / '__init__.py').write_text('')

    for template_info in templates:
        template = Template((base_path / f'{template_info.name}.jinja2').read_text())

        output = template.render(**kwargs)
        output_filename = f'{model_name}_{"_".join(template_info.name.split("_")[1:])}.py'
                
        if not template_info.always_replace and (output_path / output_filename).exists():
            continue
        (output_path / output_filename).write_text(str(output))

    total_time = process_time() - start_time

    # import the generated test_machines file
    sys.path.insert(0, str(output_path.parent))
    module = importlib.import_module(f'{model_name}_code.{model_name}_test_machines')

    # iterate through subclasses of TestMachine from the generated file
    counts = {}
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj):
            if issubclass(obj, TestMachine) and obj != TestMachine:
                counts[f'{name.lower()}_transitions'] = len([m for m in dir(obj) if m.startswith('_') and not m.startswith('__')])

    counts['total_transitions'] = sum(counts.values())
                
    assert len(counts) > 0 , "No test machines were generated."

    logfile_path = output_path / f'{model_name}_log.json'
    log_content = {
        "model_name": model_name,
        "model_path": model_path,
        "total_generation_time": total_time,
        "number_of_transitions": counts,
    }

    with open(logfile_path, 'w') as f:
        json.dump(log_content, f, indent=4)

    copyfile(model_path, output_path / f'{model_name}.fsa')
    
    return output_path