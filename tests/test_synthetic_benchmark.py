import pytest

from sp500_forecastability.synthetic_benchmark import (
    BenchmarkConfig,
    EvidenceIntervention,
    ProvenanceVisibility,
    Scenario,
    evaluate_selective_router,
    generate_episode,
    generate_future_leakage_attempt,
    generate_mechanism_heldout_split,
    generate_paired_episodes,
    generate_parameterized_episode,
    intervene_rule_agent,
    provenance_risk,
    route_selectively,
)


def test_paired_corruption_preserves_outcome_but_flips_consensus() -> None:
    clean, corrupted = generate_paired_episodes(Scenario.SHARED_CORRUPTION, seed=7)

    assert clean.outcome_action == corrupted.outcome_action
    assert clean.consensus_action == clean.outcome_action
    assert corrupted.consensus_action != corrupted.outcome_action
    assert corrupted.correlated_consensus
    assert corrupted.harmful_false_consensus


def test_shared_clean_is_correlated_but_not_harmful() -> None:
    episode = generate_episode(Scenario.SHARED_CLEAN, seed=4)

    assert episode.correlated_consensus
    assert not episode.harmful_false_consensus


def test_independent_evidence_has_lower_pre_outcome_provenance_risk() -> None:
    independent = generate_episode(Scenario.INDEPENDENT_CLEAN, seed=3)
    shared = generate_episode(Scenario.SHARED_CORRUPTION, seed=3)

    independent_risk = provenance_risk(independent.observation)
    shared_risk = provenance_risk(shared.observation)

    assert independent_risk.source_concentration == 1 / 3
    assert shared_risk.source_concentration == 1.0
    assert independent_risk.score < shared_risk.score


def test_stale_evidence_is_as_of_valid_but_high_risk() -> None:
    episode = generate_episode(Scenario.STALE_EVIDENCE, seed=5)
    risk = provenance_risk(episode.observation)

    assert episode.harmful_false_consensus
    assert risk.stale_fraction == 1.0
    assert risk.temporal_violation_fraction == 0.0
    assert risk.score > 0.8


def test_selective_router_abstains_from_shared_corruption() -> None:
    independent = generate_episode(Scenario.INDEPENDENT_CLEAN, seed=9)
    corrupted = generate_episode(Scenario.SHARED_CORRUPTION, seed=9)

    independent_route = route_selectively(independent.observation)
    corrupted_route = route_selectively(corrupted.observation)

    assert not independent_route.abstained
    assert corrupted_route.abstained
    assert corrupted_route.action == "cash"


def test_source_quality_keeps_shared_clean_but_rejects_shared_corruption() -> None:
    shared_clean = generate_episode(Scenario.SHARED_CLEAN, seed=6)
    shared_corruption = generate_episode(Scenario.SHARED_CORRUPTION, seed=6)

    clean_route = route_selectively(shared_clean.observation)
    corrupted_route = route_selectively(shared_corruption.observation)

    assert not clean_route.abstained
    assert corrupted_route.abstained
    assert clean_route.risk.source_quality_risk < corrupted_route.risk.source_quality_risk


def test_partial_corruption_creates_a_two_to_one_harmful_consensus() -> None:
    episode = generate_episode(Scenario.PARTIAL_CORRUPTION, seed=8)
    route = route_selectively(episode.observation)

    assert episode.agreement == 2 / 3
    assert episode.correlated_consensus
    assert episode.harmful_false_consensus
    assert route.abstained


def test_future_leakage_attempt_is_rejected_before_routing() -> None:
    attempt = generate_future_leakage_attempt(seed=2)

    assert attempt.rejected_by_contract()


def test_rule_agent_interventions_change_or_abstain_without_outcome() -> None:
    episode = generate_episode(Scenario.INDEPENDENT_CLEAN, seed=1)

    reversed_result = intervene_rule_agent(
        episode, "trend_01", EvidenceIntervention.REVERSE
    )
    removed_result = intervene_rule_agent(episode, "trend_01", EvidenceIntervention.REMOVE)

    assert reversed_result.action_changed
    assert not reversed_result.abstained
    assert removed_result.counterfactual_action is None
    assert removed_result.abstained


def test_parameterized_episode_supports_more_agents_and_partial_corruption() -> None:
    config = BenchmarkConfig(
        agent_count=5,
        corruption_strength=0.6,
        source_quality_noise=0.0,
        provenance_visibility=ProvenanceVisibility.FULL,
    )
    episode = generate_parameterized_episode(Scenario.PARTIAL_CORRUPTION, seed=10, config=config)

    assert len(episode.observation.decisions) == 5
    assert episode.agreement == 3 / 5
    assert episode.correlated_consensus
    assert episode.visible_correlated_consensus
    assert episode.harmful_false_consensus


def test_hidden_provenance_breaks_visible_source_overlap_but_not_truth_label() -> None:
    hidden = BenchmarkConfig(
        agent_count=5,
        corruption_strength=1.0,
        source_quality_noise=0.0,
        provenance_visibility=ProvenanceVisibility.HIDDEN,
    )
    aliased = BenchmarkConfig(
        agent_count=5,
        corruption_strength=1.0,
        source_quality_noise=0.0,
        provenance_visibility=ProvenanceVisibility.ALIASED,
    )

    hidden_episode = generate_parameterized_episode(Scenario.SHARED_CORRUPTION, seed=11, config=hidden)
    aliased_episode = generate_parameterized_episode(
        Scenario.SHARED_CORRUPTION, seed=11, config=aliased
    )

    assert hidden_episode.correlated_consensus
    assert not hidden_episode.visible_correlated_consensus
    assert aliased_episode.visible_correlated_consensus


def test_noisy_source_quality_is_reproducible_and_bounded() -> None:
    config = BenchmarkConfig(agent_count=5, source_quality_noise=0.2)

    first = generate_parameterized_episode(Scenario.SHARED_CORRUPTION, seed=12, config=config)
    second = generate_parameterized_episode(Scenario.SHARED_CORRUPTION, seed=12, config=config)

    assert dict(first.observation.source_quality) == dict(second.observation.source_quality)
    assert all(0.0 <= value <= 1.0 for value in first.observation.source_quality.values())


def test_mechanism_heldout_split_never_mixes_corruption_mechanisms() -> None:
    split = generate_mechanism_heldout_split(
        seeds=(1, 2),
        train_scenarios=(Scenario.INDEPENDENT_CLEAN, Scenario.SHARED_CORRUPTION),
        held_out_scenarios=(Scenario.STALE_EVIDENCE, Scenario.PARTIAL_CORRUPTION),
        control_scenarios=(Scenario.SHARED_CLEAN,),
        config=BenchmarkConfig(agent_count=5),
    )

    assert len(split.train) == 6
    assert len(split.test) == 6
    assert {episode.scenario for episode in split.train} == {
        *split.control_scenarios,
        *split.train_scenarios,
    }
    assert {episode.scenario for episode in split.test} == {
        *split.control_scenarios,
        *split.held_out_scenarios,
    }


def test_parameterized_config_rejects_even_agent_counts() -> None:
    with pytest.raises(ValueError, match="odd integer"):
        BenchmarkConfig(agent_count=4)


def test_outcome_time_summary_reports_coverage_and_harmful_interception() -> None:
    config = BenchmarkConfig(
        agent_count=5,
        corruption_strength=1.0,
        source_quality_noise=0.0,
        provenance_visibility=ProvenanceVisibility.FULL,
    )
    episodes = (
        generate_parameterized_episode(Scenario.INDEPENDENT_CLEAN, seed=20, config=config),
        generate_parameterized_episode(Scenario.SHARED_CORRUPTION, seed=21, config=config),
    )

    summary = evaluate_selective_router(episodes)

    assert summary.total_count == 2
    assert summary.routed_count == 1
    assert summary.abstained_count == 1
    assert summary.routed_error_rate == 0.0
    assert summary.harmful_consensus_interception_rate == 1.0
