import asyncio as aio
import logging as log
import json
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    ForwardRef,
    Generic,
    List,
    Literal,
    Mapping,
    Type,
    TypeVar,
    cast,
    get_args,
)

from chortest.cfsm import CFSM, CommunicatingSystem
from chortest.common import TransitionLabel
from hypothesis.errors import StopTest
import hypothesis as hyp
import hypothesis.strategies as st

DEADLINE = None

class AllTasksCancelledException(Exception):
    pass


class MessageType():
    pass


class ParticipantType:
    pass


def get_type_args(kls) -> tuple[Type, ...]:
    return get_args(kls.__orig_class__)


class Transport(ABC):
    @abstractmethod
    async def send(self, part_id: str, msg: MessageType):
        ...

    @abstractmethod
    async def recv(self, part_id: str) -> MessageType:
        ...

    @abstractmethod
    async def peek(self, part_id: str) -> MessageType:
        ...

    @abstractmethod
    def close(self):
        ...

    @abstractmethod
    def msg_count(self) -> int:
        ...


class LocalProtocol:
    transport: Transport

    def __init__(self, tr: Transport) -> None:
        self.transport = tr
    
    def is_closed(self) -> bool:
        return True if self.transport is None else False

    def _cast_to(self, t_prot: type['LocalProtocol'] | ForwardRef) -> 'LocalProtocol':
        if isinstance(t_prot, ForwardRef):
            t_prot = t_prot._evaluate(LocalProtocol.type_globals, locals(), frozenset()) # type: ignore
        if t_prot is Eps:
            self.__class__ = Eps
        else:
            self.__class__ = t_prot.__origin__  # type: ignore
        self.__orig_class__ = t_prot  # type: ignore
        return self

    def _can_recv(self) -> bool:
        if self.is_closed():
            return False
        return self.transport.msg_count() > 0


class Eps(LocalProtocol):
    def close(self):
        self.transport.close()


_TMsg = TypeVar("_TMsg", bound=MessageType)  # What to send or recv
_TMsg_L = TypeVar("_TMsg_L", bound=MessageType)  # For offer

_TProt = TypeVar("_TProt", bound=LocalProtocol)  # Rest of the protocol
_TL = TypeVar("_TL", bound=LocalProtocol)
_TR = TypeVar("_TR", bound=LocalProtocol)


@dataclass
class Left(Generic[_TL, _TMsg_L]):
    prot: _TL
    msg: _TMsg_L

@dataclass
class Right(Generic[_TR]):
    prot: _TR

_TSender = TypeVar("_TSender", bound=ParticipantType)
_TReceiver = TypeVar("_TReceiver", bound=ParticipantType)


class Recv(LocalProtocol, Generic[_TSender, _TMsg, _TProt]):
    async def recv(self) -> tuple[_TProt, _TMsg]:
        t_sender, t_msg, t_prot  = get_type_args(self)

        res: MessageType = await self.transport.recv(t_sender.__name__)
        # assert type(res) == t_msg, "Type mismatch"

        return cast(_TProt, self._cast_to(t_prot)), cast(_TMsg, res)

    def _get_acceptable_msg_types_recursively(self) -> tuple[MessageType]:
        t_args = get_args(self.__orig_class__)
        return (t_args[1],)


class Send(LocalProtocol, Generic[_TReceiver, _TMsg, _TProt]):
    async def send(self, val: _TMsg):
        t_args = get_args(self.__orig_class__)
        t_recv, t_msg, t_prot = t_args[0], t_args[1], t_args[2]

        # assert type(val) == t_msg, "Type mismatch"
        await self.transport.send(t_recv.__name__, val)

        return cast(_TProt, self._cast_to(t_prot))


class Offer(LocalProtocol, Generic[_TSender, _TMsg_L, _TL, _TR]):
    """
    Passive choice. Tries to recv a message of type `TMsg_L` from
    `_TSender`. If it succeeds, it casts itself to `TL`. Otherwise,
    it casts itself to `TR`.
    """

    async def offer(self) -> Left[_TL, _TMsg_L]|Right[_TR]:
        t_sender, t_msg, tl_cls, tr_cls = get_args(self.__orig_class__) # type: ignore

        msg = await self.transport.peek(t_sender.__name__)

        if isinstance(msg, t_msg):
            return Left(cast(_TL, self._cast_to(tl_cls)), cast(_TMsg_L, msg))
        else:
            return Right(cast(_TR, self._cast_to(tr_cls)))

def get_offered_messages(offer_proto) -> List[Type[MessageType]]:
    messages = []

    if not issubclass(offer_proto.__origin__, Offer):
        raise ValueError("Expected Offer protocol type")

    part_t, msg_t, l_t, r_t = get_args(offer_proto)
    messages.append(msg_t.__name__)
    # If the protocol is Offer, traverse it recursively
    if issubclass(r_t.__origin__, Offer):
        messages += get_offered_messages(r_t)
    elif issubclass(r_t.__origin__, Recv):
        messages += [r_t.__args__[1].__name__]

    return messages


class Choose(LocalProtocol, Generic[_TReceiver, _TMsg, _TL, _TR]):
    # Active choice. This allows the other end of the channel
    # to select one of two options for continuing the protocol:
    # either `TL` or `TR`.
    async def pick(self, val: _TMsg) -> _TL:
        t_recv, t_msg, tl_cls, _ = get_args(self.__orig_class__)  # type: ignore

        assert type(val) == t_msg
        await self.transport.send(t_recv.__name__, val)

        return cast(_TL, self._cast_to(tl_cls))

    def skip(self) -> _TR:
        t_recv, t_msg, tl_cls, tr_cls = get_args(self.__orig_class__)  # type: ignore

        return cast(_TR, self._cast_to(tr_cls))


ParticipantTypeName = str
MessageTypeName = str

@dataclass
class GlobalInfo:
    # Class variables
    channels: Mapping[
        tuple[ParticipantTypeName, ParticipantTypeName],
        aio.Queue[MessageType]
    ] = field(default_factory=lambda: defaultdict(aio.Queue))
    message_counter: Mapping[
        tuple[ParticipantTypeName, MessageTypeName], 
        Dict[MessageTypeName, int]
    ] = field(default_factory= lambda: defaultdict(lambda: defaultdict(int)))
    lock: aio.Lock = field(default_factory=aio.Lock)
    channel_count: int = 0

class AsyncioTransport(Transport):
    # Instance variables
    part_name: str
    closed: bool
    recv_cache: Dict[ParticipantTypeName, MessageType]

    # Global
    global_info: GlobalInfo

    def __init__(self, global_info: GlobalInfo, part_name: str) -> None:
        self.channels = global_info.channels
        self.part_name = part_name
        self.closed = False
        self.global_info = global_info
        self.recv_cache = {}

        global_info.channel_count += 1

        super().__init__()

    async def send(self, target_part: str, msg: MessageType):
        msg_name = type(msg).__name__

        log.info(f"[TRACE] {self.part_name}-{target_part}!{msg_name} ({msg})")

        q_key = (self.part_name, target_part)
            
        async with self.global_info.lock:
            mc = self.global_info.message_counter
            mc[q_key][msg_name] = mc[q_key].get(msg_name, 0) + 1
            assert mc[q_key][msg_name] > 0
            q = self.channels[q_key]
            await q.put(msg)

    async def recv(self, sender_part: str) -> MessageType | None:
        # check cache
        if sender_part in self.recv_cache:
            ms = self.recv_cache[sender_part]
            del self.recv_cache[sender_part]
            return ms

        assert sender_part != self.part_name
        q_key = (sender_part, self.part_name)
        try:
            msg: MessageType = await self.channels[q_key].get() # TODO: Should this await on all queues?
            msg_name = type(msg).__name__
            log.info(f"[TRACE] {sender_part}-{self.part_name}?{msg_name} ({msg})")
            async with self.global_info.lock:
                mc = self.global_info.message_counter
                assert mc[q_key][msg_name] >= 0
                mc[q_key][msg_name] -= 1
                if mc[q_key][msg_name] == 0:
                    del mc[q_key][msg_name]
                return msg
        except Exception as e:
            # log.error(f"Unexpected error: {e}")
            # pass
            raise e


    async def peek(self, sender_part: str) -> MessageType:
        if sender_part in self.recv_cache:
            return self.recv_cache[sender_part]

        keys: set[tuple[str, str]] = set(
            k for k in self.channels.keys() 
            if k[1] == self.part_name
        )
        keys.add((sender_part, self.part_name))
        keys_list = list(keys)

        msg = None
        queue_list = [
            aio.create_task(self.channels[k].get()) for k in keys_list
        ]

        try:
            for completed_task in aio.as_completed(queue_list):
                msg = await completed_task

                #cancel all other tasks
                for task in queue_list:
                    if not task.done():
                        task.cancel()
                        await task
        except aio.CancelledError:
            pass
        finally:
            if msg:
                # if sender_part not in self.recv_cache:
                self.recv_cache[sender_part] = msg
                return msg
            raise AllTasksCancelledException("No messages available. Check for deadlocks.")

    def can_recv_from(self, sender_part: Type[ParticipantType]) -> bool:
        q_key = (sender_part.__name__, self.part_name)
        counter = self.global_info.message_counter[q_key]
        if not self.closed:
            return (self.channels[q_key].qsize() > 0 
            or any([counter[msg_type] > 0 for msg_type in counter]))
        return False

    def can_recv(self, sender_part: Type[ParticipantType], msg_type: Type[MessageType]) -> bool:
        key = (sender_part.__name__, self.part_name)
        counter = self.global_info.message_counter[key]
        if not self.closed:
            return counter[msg_type.__name__] > 0
        return False

    def msg_count(self) -> int:
        if self.closed:
            return 0
        keys = [k for k in self.channels.keys() if k[1] == self.part_name]
        return sum([sum(self.global_info.message_counter[k].values()) for k in keys])

    def close(self):
        log.info(f"Transport for {self.part_name} closed.")
        assert self.global_info.channel_count > 0
        self.global_info.channel_count -= 1
        if self.global_info.channel_count == 0:
            log.info("All transports closed. Exiting.")
            del self.channels
        self.closed = True

    def msg_types_awaiting_recv(self) -> list[MessageTypeName]:
        keys = [k for k in self.channels.keys() if k[1] == self.part_name]
        return [msg_type for k in keys for msg_type in self.global_info.message_counter[k].keys() if self.global_info.message_counter[k][msg_type] > 0]

    @staticmethod
    def make_for(*part_names: str) -> list["AsyncioTransport"]:
        g_info = GlobalInfo()
        return [AsyncioTransport(g_info, p) for p in part_names]


class TestMachine():
    __test__= False

    part_type: Type[ParticipantType]
    state: int
    received_messages: dict[int, MessageType]
    protocol: LocalProtocol

    def __init__(self, init_state: int, protocol: LocalProtocol) -> None:
        self.received_messages = {}
        self.state = init_state
        self.protocol = protocol
    
    def method_name(self, model:CommunicatingSystem, transition: 'Transition') -> str:
        m: CFSM = model[transition.machine.part_type.__name__]
        trs = {tr for tr in m.transitions[str(transition.machine.state)] if tr.m == transition.message.__name__}
        assert len(trs) == 1, f"Transition {transition} not found in model or more than one transition found."
        tr_label = trs.pop()

        cur_state = self.state
        target_state = (
            m.transitions[str(self.state)][tr_label] 
            if 'skip' not in transition.action 
            else str(self.state)
        )

        message_type_name = transition.message.__name__
        fname = f"_{cur_state}_{message_type_name.lower()}_{transition.action}_{target_state}"

        return fname



ImplementationMap = Dict[Type[ParticipantType], TestMachine]

@dataclass
class Transition():
    machine: TestMachine
    A: Type[ParticipantType]
    B: Type[ParticipantType]
    action: Literal['send'] | Literal['recv'] | Literal['choice_pick'] | Literal['choice_skip'] | Literal['offer_recv'] | Literal['offer_skip']
    message: Type[MessageType]

    def __repr__(self) -> str:
        act = '!' if self.action in ['send', 'choice_pick'] else '?' if self.action in ['recv', 'offer_recv'] else '*'
        return f"{self.A.__name__}-{self.B.__name__}{act}{self.message.__name__}" + (' (skipped)' if 'skip' in self.action else '')


@dataclass
class ModelStats():
    messages: set[str]
    transitions: dict[str, set[str]]
    states: dict[str, set[str]]

    message_coverage: list[float]
    transition_coverage: list[float]
    state_coverage: list[float]
    timesteps: list[int]
    current_timestep: int

    missing_transitions: list[str]


    model: CommunicatingSystem

    def __init__(self) -> None:
        self.message_coverage = []
        self.transition_coverage = []
        self.state_coverage = []
        self.timesteps = []

        self.missing_transitions = []
        self.current_timestep = 0

        self.reset()

    def reset_timestep(self):
        self.current_timestep = 0

    def set_model(self, model: CommunicatingSystem):
        self.model = model

    def reset(self):
        self.messages = set()
        self.transitions = dict()
        self.states = dict()

    def add_message(self, msg_type: Type[MessageType]):
        self.messages.add(msg_type.__name__)
    
    def add_transition(self, cut_name: str, transition: Transition):
        if not cut_name in self.transitions:
            self.transitions[cut_name] = set()
        self.transitions[cut_name].add(repr(transition))

    def add_state(self, machine: TestMachine):
        m_name = machine.part_type.__name__
        if not m_name in self.states:
            self.states[m_name] = set()
        self.states[m_name].add(str(machine.state))

    def calc_message_coverage(self) -> float:
        return len(self.messages) / len(self.model.all_message_types())

    def calc_transition_coverage(self) -> tuple[float, set[str]]:
        all_set = set()
        covered_set = set()
        for part in self.model.machines.keys():
            all_set = all_set.union(list(map(lambda x: repr(x[1]), (self.model[part].all_transitions()))))
            covered_set = covered_set.union({str(tr) for tr in self.transitions[part]} if part in self.transitions else set())
        uncovered = all_set - covered_set
        return len(covered_set) / len(all_set), uncovered 

    def calc_state_coverage(self) -> float:
        # Add the initial states
        for m_name in self.states:
            self.states[m_name].add(self.model.machines[m_name].initial)

        return sum([len(self.states[m_name]) for m_name in self.states]) / sum([len(self.model[m_name].states) for m_name in self.model.machines])

    def calc_coverages(self):
        t_cov_ratio, trs_cov = self.calc_transition_coverage()
        self.message_coverage.append(self.calc_message_coverage())
        self.transition_coverage.append(t_cov_ratio)
        self.state_coverage.append(self.calc_state_coverage())
        
        self.missing_transitions = list(trs_cov)

        self.timesteps.append(self.current_timestep)
        self.current_timestep += 1

    def cover(self, cut_name: str, tr: Transition):
        self.add_message(tr.message)
        self.add_transition(cut_name, tr)
        self.add_state(tr.machine)

    def save(self, path):
        with open(path, 'w') as f:
            t_cov_ratio, trs_cov = self.calc_transition_coverage()
            json.dump({
                'messages': list(self.messages),
                'transitions': {k: list(v) for k, v in self.transitions.items()},
                'states': {k: list(v) for k, v in self.states.items()},
                'message_coverage': self.message_coverage,
                'transition_coverage': self.transition_coverage,
                'state_coverage': self.state_coverage,
                'transitions_missing': list(trs_cov),
                'timesteps' : list(self.timesteps)
            }, f, indent=4)

cov_info = ModelStats()

def accepting(machines) -> bool:
    return all(type(m.protocol) is Eps for m in machines.values())

def active_transitions(model: CommunicatingSystem, machines: ImplementationMap, CUT: Type[ParticipantType], cut_task: aio.Task):
    
    for participant in machines:
        m: TestMachine = machines[participant]
        if isinstance(m.protocol, Send):
            target, msg_type, _ = get_type_args(m.protocol)
            yield Transition(m, m.part_type, target, "send", msg_type)
        elif isinstance(m.protocol, Recv):
            sender, msg_type, _ = get_type_args(m.protocol)
            assert isinstance(m.protocol.transport, AsyncioTransport)
            if m.protocol.transport.can_recv(sender, msg_type):
                yield Transition(m, sender, m.part_type, "recv", msg_type)
        elif isinstance(m.protocol, Choose):
            target, msg_type, TL, TR = get_type_args(m.protocol)
            yield Transition(m, m.part_type, target, "choice_pick", msg_type)
            yield Transition(m, m.part_type, target, "choice_skip", msg_type)
        elif isinstance(m.protocol, Offer):
            sender, msg_type, left_type, right_type = get_type_args(m.protocol)
            assert issubclass(right_type.__origin__, LocalProtocol)
            assert isinstance(m.protocol.transport, AsyncioTransport)
            if m.protocol.transport.can_recv(sender, msg_type):
                yield Transition(m, sender, m.part_type, "offer_recv", msg_type)
            else:
                ready_msgs = set(m.protocol.transport.msg_types_awaiting_recv())
                if ready_msgs:
                    offered = set(get_offered_messages(m.protocol.__orig_class__))
                    if offered.intersection(ready_msgs):
                        yield Transition(m, sender, m.part_type, "offer_skip", msg_type)

def subject(tr: Transition) -> Type[ParticipantType]:
    if tr.action == 'send':
        return tr.A
    elif tr.action == 'recv':
        return tr.B
    else:
        raise ValueError('Invalid action')

async def run_test_loop(model, machines, cut, cut_task, data: st.DataObject, model_stats: ModelStats, verbose : bool = False, ignore_cut_exceptions: bool = True):
    try:
        model_stats.set_model(model)
        
        # init the active transitions
        await aio.sleep(0)
        active: list[Transition] = list(active_transitions(model, machines, cut, cut_task))

        # If the cut is the initiator of the protocol
        # then we need to wait for the cut to be ready
        if len(active) == 0:
            await aio.sleep(0)
            active = list(active_transitions(model, machines, cut, cut_task))
            assert len(active) > 0, "No active transitions found"

        if verbose:
            print('\n---------------------------------')
            print(f'Starting test loop for CUT {cut.__name__}')
            print('---------------------------------')

        i = 1
        # while there are active transitions
        while True:

            # draw from the active transitions
            if data.conjecture_data.frozen:
                break
                
            tr: Transition = data.draw(st.sampled_from(active))
            # print(f"{i}:{repr(tr)}")
            i+=1

            # dispatch the transition to the correct test machine
            fname = tr.machine.method_name(model, tr)
            assert hasattr(tr.machine, fname)
            f = tr.machine.__getattribute__(fname)
            try:
                test_task = aio.create_task(f(), name=fname)
                await test_task
            except aio.CancelledError:
                return
            except StopTest as stop:
                raise stop

            if 'skip' not in tr.action:
                hyp.event(f"Taken transition {repr(tr)}")
                model_stats.cover(cut.__name__, tr)

            # at each step, check if the model state is an accepting state
            # if it is, then we're done  
            if accepting(machines):
                break

            await aio.sleep(0)
            # update the active transitions
            active = list(active_transitions(model, machines, cut, cut_task))

            if len(active) == 0:
                # try:
                #     await aio.wait_for(cut_task, timeout=0.1)
                # except aio.TimeoutError:
                #     pass
                # TODO: Find out why this sleep is needed 
                # and if it can be removed
                await aio.sleep(0.01) 

                active = list(active_transitions(model, machines, cut, cut_task))
                if len(active) == 0:
                    break

        if not cut_task.done():
            await aio.wait_for(cut_task, timeout=DEADLINE)
            # await cut_task

        # check that this state was indeed final
        if cut_task.done():
            if not accepting(machines):
                raise aio.InvalidStateError("Test done, but machines are not in an accepting state")
            if cut_task.exception() and not ignore_cut_exceptions:
                raise aio.InvalidStateError("CUT raised an error.") from cut_task.exception()
            
    except Exception as e:
        if cut_task.exception() and not ignore_cut_exceptions:
            raise e from cut_task.exception()
        elif not accepting(machines):
            raise aio.InvalidStateError("Test done, but machines are not in an accepting state")
    finally:
        try:
            # TODO: check if this is always executed
            model_stats.calc_coverages() 

            if not cut_task.done():
                cut_task.cancel()
                await cut_task
            for t in aio.all_tasks():
                if t is not aio.current_task():
                    t.cancel()
                    await t
        except aio.CancelledError:
            pass
