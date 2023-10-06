import random
from typing import Any, Generator, Union
from chortest.cfsm import CFSM, CommunicatingSystem
from chortest.gchor import *

from chortest.common import (
    InTransitionLabel,
    LocalMutationTypes,
    OutTransitionLabel,
    Participant,
    State,
)
from chortest.gchor import GChor


class LocalMutator:
    @staticmethod
    def mutate_randomly(
        cs: CommunicatingSystem,
        mut_type: LocalMutationTypes = LocalMutationTypes.REMOVE_RANDOM_OUTPUT,
        seed: int = 1,
        target_p: Participant = None,
        pick_random_mutation: bool = False,
    ) -> LocalMutationTypes:
        """
        Mutates a specific CFSM according to the chosen type of mutation.

        Returns the type of mutation selected.
        """

        # initialize the random seed
        r = random.Random(seed)

        # if not participant is specified, randomly choose one
        if target_p is None:
            target_p = r.choice(cs.participants)
        m: CFSM = cs.machines[target_p]

        if pick_random_mutation:
            mut_type = random.choice(list(LocalMutationTypes)[1:])  # Exclude none

        # if mut_type == MutationTypes.SWAP_RANDOM_CONSECUTIVE_TRANSITIONS:
        # cs: CommunicatingSystem.mutate_swap_random_consecutive_transitions(m)
        elif mut_type == LocalMutationTypes.REMOVE_RANDOM_OUTPUT:
            LocalMutator.mutate_remove_random_output(m)
        elif mut_type == LocalMutationTypes.CHANGE_RANDOM_TRANSITION_MESSAGE_TYPE:
            LocalMutator.mutate_change_random_transition_message_type(m)
        elif mut_type == LocalMutationTypes.SWAP_INTERACTION_TYPE:
            LocalMutator.mutate_swap_interaction_type(m)

        return mut_type

    @staticmethod
    def mutate_change_random_transition_message_type(m: CFSM):
        """
        Mutates by changing a random transition's type to another type
        randomly drawn from the same gchor.
        """

        trs = list(
            filter(lambda tr: type(tr[1]) is OutTransitionLabel, m.all_transitions())
        )

        random_q, random_tr, old_q1 = random.choice(trs)

        new_tr = type(random_tr)(
            random_tr.A,
            random_tr.B,
            random_tr.m
            + "wrong",  # TODO: Use another message type from the system, take into account the case where there is no other message type, since the mutation would not introduce an error. Possibly, use a different mutation. throw Exc and repeat the mutation. Try up to n times to mutate.
        )

        del m.transitions[random_q][random_tr]
        m.transitions[random_q][new_tr] = old_q1

    @staticmethod
    def mutate_swap_interaction_type(m: CFSM):
        """
        Mutates by changing the transition type of a random transition
        (from output to input)
        """

        def has_single_output_transition(q: State) -> bool:
            return (
                len(m.transitions[q]) == 1
                and type(next(m.transitions[q].__iter__())) == OutTransitionLabel
            )

        filtered_states = list(
            filter(has_single_output_transition, m.transitions.keys())
        )

        q0 = random.choice(list(filtered_states))
        tr = random.choice(list(m.transitions[q0]))

        q1 = m.transitions[q0][tr]

        assert type(tr) == OutTransitionLabel
        new_tr = InTransitionLabel(tr.B, tr.A, tr.m)

        del m.transitions[q0][tr]
        m.transitions[q0][new_tr] = q1

    @staticmethod
    def mutate_remove_random_output(m: CFSM):
        """
        # TODO: (check if refining this removes errors in all the parallel cases)
        Mutates by removing a random output from the CFSM.
        Removes only outputs if they're not part of an internal
        choice or mixed state (that is, if they're the only output
        from a given state).
        """

        def has_single_output_transition(q: State):
            return len(m.transitions[q]) == 1 and next(m.transitions[q].__iter__())

        filtered_states = list(
            filter(has_single_output_transition, m.transitions.keys())
        )
        # TODO: Filter transitions to include only those
        # that will break the protocol. That is, those that
        # which are not part of an internal or mixed choice.
        q0 = random.choice(list(filtered_states))

        # pick that single transition
        tr = random.choice(list(m.transitions[q0]))

        del m.transitions[q0][tr]

    # def mutate_swap_random_consecutive_transitions(cs: CommunicatingSystem, m: CFSM):
    #     """
    #     # TODO: Refine (Two consecutive outputs won't introduce errors with the bag semantics)
    #     """
    #     # pick a random node
    #     q0 = choice(list(m.states))

    #     # pick a random transition from there
    #     tr1 = choice(list(m.transitions[q0]))
    #     q1 = choice(list(m.transitions[q0][tr1]))

    #     # qA -1-> qB -2-> qC
    #     #          \ -3-> qD
    #     # qB -3-> qD
    #     # TODO: To guarantee that an error is introduced
    #     # check additionally that the message types differ.
    #     # pick a random transition from the next node
    #     tr2 = choice(list(m.transitions[q1]))
    #     q2 = m.transitions[q1][tr2]

    #     # swap those transitions
    #     tr1.A, tr1.B, tr1.m, tr2.A, tr2.B, tr2.m = (
    #         tr2.A,
    #         tr2.B,
    #         tr2.m,
    #         tr1.A,
    #         tr1.B,
    #         tr1.m,
    #     )  # Assumes msg is a string

    @staticmethod
    def mutate_systematically(
        cs: CommunicatingSystem, p: Participant
    ) -> Generator[tuple[CommunicatingSystem,dict], None, None]:
        """
        Mutates the CFSM corresponding to the given participant.

        Returns an iterator generating copies of the original communicating system.
        """

        # Try to apply all operations
        yield from LocalMutator.mutate_systematically_by_removing_an_output(cs, p)
        yield from LocalMutator.mutate_systematically_by_changing_a_message_type(cs, p)
        yield from LocalMutator.mutate_systematically_by_converting_an_output_to_an_input(
            cs, p
        )
        yield from LocalMutator.mutate_systematically_by_removing_a_state(
            cs, p
        )
        yield from LocalMutator.mutate_systematically_by_repeating_a_transition(
            cs, p
        )

    @staticmethod
    def mutate_systematically_by_removing_an_output(
        cs: CommunicatingSystem, p: Participant
    ) -> Generator[Any, None, None]:
        m = cs.machines[p]
        for q in m.transitions:
            trs = m.transitions[q]
            # check that there's only one output transition
            for tr in trs:
                if len(trs) == 1 and type(list(trs)[0]) == OutTransitionLabel:
                    new_cs = cs.copy()
                    new_m = new_cs.machines[p]
                    del new_m.transitions[q][tr]
                    yield new_cs, {
                        'type': 'remove_output',
                        'change': (tr),
                    }

    @staticmethod
    def mutate_systematically_by_changing_a_message_type(
        cs: CommunicatingSystem, p: Participant
    ) -> Generator[Any, None, None]:
        m = cs.machines[p]
        for q0 in m.transitions:
            trs = m.transitions[q0]
            for tr in trs:
                new_cs = cs.copy()
                new_m = new_cs.machines[p]
                old_q1 = new_m.transitions[q0][tr]
                del new_m.transitions[q0][tr]
                new_tr = type(tr)(
                    tr.A, tr.B, tr.m + "wrong"
                )  # TODO: Pick from other messages in the gc.
                new_m.transitions[q0][new_tr] = old_q1
                yield new_cs, {
                    'type': 'change_message_type',
                    'change': (tr),
                }

    @staticmethod
    def mutate_systematically_by_converting_an_output_to_an_input(
        cs: CommunicatingSystem, p: Participant
    ) -> Generator[Any, None, None]:
        new_tr: Union[InTransitionLabel, OutTransitionLabel]
        m = cs.machines[p]
        for q in m.transitions:
            trs = m.transitions[q]
            for tr in trs:
                if (
                    type(tr) == OutTransitionLabel
                    and all(
                        type(tr2) == InTransitionLabel
                        for tr2 in trs
                        if tr2 != tr 
                    )
                ) or type(tr) == InTransitionLabel:
                    old_q1 = m.transitions[q][tr]

                    new_cs = cs.copy()
                    new_m = new_cs.machines[p]
                    del new_m.transitions[q][tr]
                    opposite_type: Any = (
                        InTransitionLabel
                        if type(tr) == OutTransitionLabel
                        else OutTransitionLabel
                    )
                    new_tr = opposite_type(tr.B, tr.A, tr.m)
                    new_m.transitions[q][new_tr] = old_q1
                    yield new_cs, {
                        'type': 'swap_transition_type',
                        'change': (tr),
                    }

    @staticmethod
    def mutate_systematically_by_removing_a_state(
        cs: CommunicatingSystem, p: Participant
    ) -> Generator[Any, None, None]:
        m = cs.machines[p]
        for q in m.transitions:
            new_cs = cs.copy()
            new_m = new_cs.machines[p]
            del new_m.transitions[q]
            yield new_cs, {
                'type': 'remove_state',
                'change': q,
            }

    @staticmethod
    def mutate_systematically_by_repeating_a_transition(
        cs: CommunicatingSystem, p: Participant
    ) -> Generator[Any, None, None]:
        m = cs.machines[p]
        for q0 in m.transitions:
            trs = m.transitions[q0]
            for tr in trs:
                new_cs = cs.copy()
                new_m = new_cs.machines[p]
                old_q1 = new_m.transitions[q0][tr]
                del new_m.transitions[q0][tr]
                new_q1 = "q" + str(len(new_m.transitions) + 1)
                new_m.transitions[q0][tr] = new_q1
                new_m.transitions[new_q1] = {tr: old_q1}
                yield new_cs, {
                    'type': 'repeat_transition',
                    'change': (tr),
                }
        

