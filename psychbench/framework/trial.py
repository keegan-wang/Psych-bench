"""Run a single trial: agents respond in order, environment gates visibility."""
from __future__ import annotations

from .agent import BaseAgent
from .environment import Environment
from .types import AgentResponse, Stimulus, TrialContext, TrialResult


def run_trial(
    stimulus: Stimulus,
    agents: list[BaseAgent],
    environment: Environment,
) -> TrialResult:
    environment.begin_trial(stimulus)
    ordered = sorted(agents, key=lambda a: a.position)
    responses: list[AgentResponse] = []
    for agent in ordered:
        prior = environment.visible_prior_responses(
            agent.agent_id, agent.position
        )
        ctx = TrialContext(
            stimulus=stimulus,
            agent_position=agent.position,
            agent_id=agent.agent_id,
            prior_responses=prior,
        )
        resp = agent.respond(ctx)
        responses.append(resp)
        if resp.parsed_answer is not None:
            environment.record_response(
                agent.agent_id, agent.position, resp.parsed_answer,
            )
    return TrialResult(
        trial_index=stimulus.trial_index,
        is_critical=stimulus.is_critical,
        stimulus=stimulus,
        responses=responses,
        conformed=None,
        naive_answer=None,
        confederate_answer=None,
        correct_answer=stimulus.correct_label,
    )
