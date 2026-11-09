from pathlib import Path

import pytest

from vendorkb.taxonomy import load
from vendorkb.topics import detect_topic

SHIPPED = Path(__file__).parent.parent.parent / "taxonomy" / "taxonomy.yaml"


@pytest.fixture(scope="module")
def T() -> list[load.Topic]:
    """The shipped taxonomy's topics (just the ransomware protection kit today)."""
    return load.validate(load.read_doc(SHIPPED)).topics


def _topic(topic_id: str, triggers: tuple[str, ...], cues: tuple[str, ...]) -> load.Topic:
    aspect = load.Aspect("one", "One", "the one aspect", cues, ("security.rbac",))
    return load.Topic(topic_id, topic_id, triggers, (aspect,))


@pytest.mark.parametrize("question", [
    "How does Veeam protect backups against ransomware?",
    "How do Druva and Cohesity compare for cyber-resilience?",  # hyphen = space
    "How does Rubrik handle cyber threats?",  # plural of a trigger
    "Which products help after cyberattacks?",
    "Which products help after cyber attacks?",
    "What does Cohesity offer for RANSOMWARE   recovery?",  # case, runs of whitespace
])
def test_a_broad_question_triggers_the_topic(T, question):
    topic = detect_topic(question, T)
    assert topic is not None and topic.id == "ransomware_protection"


@pytest.mark.parametrize("question", [
    "How does immutability protect against ransomware?",
    "Does Veeam support YARA rules for ransomware?",
    "What is the retention period?",  # no trigger
    "Is air-gapped backup available in Cohesity for ransomware?",  # cue "air gap" + hyphen rule
    "Does Commvault offer anti-virus scanning against ransomware?",
    "Which vendors quarantine infected restore points after a ransomware attack?",  # stem
    "Does Rubrik have anomaly detection for ransomware?",  # stem
    "Is multi-factor authentication required in the console for ransomware defence?",
    "Do Veeam immutable repositories stop cyberattacks?",
])
def test_naming_an_aspect_keeps_todays_behaviour(T, question):
    assert detect_topic(question, T) is None


@pytest.mark.parametrize("question", [
    "Does Veeam support Linux hardened repositories against ransomware?",
    "Which vendors offer isolated copies of backups against ransomware?",
    "Does Rubrik use indicators of compromise to find ransomware?",
    "Does Veeam send alerts to our SIEM during a ransomware attack?",
    "Does Cohesity have audit logging against ransomware?",
    "Does Veeam detect abnormal deletions for ransomware?",
    "Can Commvault spot unusual file changes indicating ransomware?",
    "Can Cohesity scan backups for malware after ransomware?",
    "Does Veeam detect malware in backups to protect against ransomware?",
    "Does Veeam scan restore points before recovery from ransomware?",
    "Does Veeam support Secure Restore for ransomware?",
    "Does Commvault require multi-party approval to delete backups after ransomware?",
    "Does Commvault encrypt backups against ransomware?",
    "Does Veeam support encryption to protect against ransomware?",
    "Does Rubrik offer an isolated recovery environment for ransomware?",
    "Does Dell use retention lock against ransomware?",
    "Does NetApp keep offline copies against ransomware?",
    "Does Veeam require two-person approval to delete backups after ransomware?",
    "Does Veeam encrypt data in transit and at rest to protect against ransomware?",
    "Does Veeam support AES-256 against ransomware?",
    "Can Rubrik restore to the last known good snapshot after ransomware?",
    "Can Commvault use machine learning to detect ransomware?",
    "Does Veeam forward ransomware alerts to Splunk?",
    "Does Veeam send ransomware events to Microsoft Sentinel?",
    "Does Cohesity DataLock stop ransomware from deleting backups?",
    "Does NetApp SnapLock protect backups from ransomware?",
    "Does Pure Storage Safe Mode protect against ransomware?",
])
def test_aspect_specific_questions_do_not_expand(T, question):
    assert detect_topic(question, T) is None


@pytest.mark.parametrize("question", [
    "How does Veeam protect backups against ransomware?",
    "How does Commvault protect backups against ransomware?",
    "What ransomware protection does Rubrik offer?",
    "How does Cohesity help me recover from a ransomware attack?",
    "How do Veeam Backup & Replication and Azure Backup each protect backups against"
    " ransomware?",
    "Compare Commvault and Rubrik on ransomware protection.",
    "How do Druva and Cohesity compare for cyber resilience?",
    "How can I protect my backups against ransomware?",
    "Which vendors protect backups against ransomware?",
    "Which backup vendors offer ransomware protection?",
    "How do I recover after ransomware encrypted my files?",
    "How does Rubrik handle cyber threats?",
    "Which products help after cyberattacks?",
    "How does Veeam help at restore time after a ransomware attack?",
    "What happens in transition to a new repository after ransomware?",
])
def test_broad_questions_expand(T, question):
    topic = detect_topic(question, T)
    assert topic is not None and topic.id == "ransomware_protection"


def test_incidental_words_do_not_count_as_naming_an_aspect(T):
    assert detect_topic("How do I recover after ransomware encrypted my files?", T) is not None
    assert detect_topic("Is my ransomware backup copy safe?", T) is not None


def test_a_trigger_is_a_whole_phrase():
    topics = [_topic("t", ("cyber attack", "ransomware"), ())]
    assert detect_topic("Protection against ransomwares", topics) is not None  # plural suffix
    assert detect_topic("Protection against ransomwarexyz", topics) is None
    assert detect_topic("Protection against notransomware", topics) is None
    assert detect_topic("Protection against a cyber", topics) is None  # part of a trigger


def test_a_cue_matches_at_a_word_start_only():
    topics = [_topic("t", ("ransomware",), ("worm", "immutab"))]
    assert detect_topic("ransomware and worm storage", topics) is None
    assert detect_topic("ransomware and immutable storage", topics) is None
    assert detect_topic("ransomware and an unimmutable storage", topics) is not None


def test_phrases_are_matched_literally_and_hyphens_agree_on_both_sides():
    topics = [_topic("t", ("c++ attack", "cyber-resilience"), ("anti-virus", "a.b"))]
    assert detect_topic("Which c++ attack protections exist?", topics) is not None
    assert detect_topic("Which cyber resilience features exist?", topics) is not None
    assert detect_topic("Which cyber resilience features exist with anti virus?", topics) is None
    # "antivirus" is not the cue "anti-virus": the match is on the words, not the letters
    assert detect_topic("Which cyber-resilience features exist with antivirus?", topics) \
        is not None
    assert detect_topic("Which c++ attack protections exist for a.b?", topics) is None
    assert detect_topic("Which c++ attack protections exist for axb?", topics) is not None


def test_the_first_topic_that_matches_wins():
    first = _topic("a_topic", ("ransomware",), ())
    second = _topic("b_topic", ("ransomware", "cyber attack"), ())
    assert detect_topic("ransomware and cyber attack", [first, second]) is first
    assert detect_topic("ransomware and cyber attack", [second, first]) is second
    # a topic whose cue is named does not fire, so the next one can
    blocked = _topic("a_topic", ("ransomware",), ("ransomware",))
    assert detect_topic("ransomware", [blocked, second]) is second


def test_no_topics_or_an_empty_question_detects_nothing(T):
    assert detect_topic("How does Veeam protect against ransomware?", []) is None
    assert detect_topic("", T) is None


def test_a_phrase_that_normalises_to_nothing_never_matches():
    topics = [_topic("t", ("ransomware", "-", " "), ("-", "  ", "\u2013"))]
    assert detect_topic("ransomware backups", topics) is topics[0]  # an empty cue blocks nothing
    assert detect_topic("backups", [_topic("t", ("-", ""), ())]) is None  # nor fires on anything


@pytest.mark.parametrize("dash", ["\u2010", "\u2011", "\u2013", "\u2014"])
def test_typographic_dashes_count_as_hyphens(dash):
    topics = [_topic("t", ("cyber resilience",), ("anti virus",))]
    assert detect_topic(f"Which products give cyber{dash}resilience?", topics) is not None
    assert detect_topic(f"Which products give cyber resilience, anti{dash}virus?", topics) is None
