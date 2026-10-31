from litdatamatcher.text import (
    infer_population,
    infer_required_variables,
    lexical_similarity,
    normalize_text,
    split_sections,
    split_sentences,
)


def test_normalize_text_collapses_nonbreaking_space_and_whitespace():
    assert normalize_text("  gut\u00a0microbiome\n recovery  ") == "gut microbiome recovery"


def test_sentence_splitter_handles_abbreviations_and_punctuation():
    text = "Dr. Smith works at MIT. What?! Really?! Yes!"

    sentences = split_sentences(text)

    assert sentences[0] == "Dr. Smith works at MIT."
    assert sentences[1:] == ["What?!", "Really?!", "Yes!"]


def test_sentence_splitter_preserves_common_scientific_abbreviations():
    text = "Smith et al. tested E. coli responses. These findings were stable."

    sentences = split_sentences(text)

    assert sentences == [
        "Smith et al. tested E. coli responses.",
        "These findings were stable.",
    ]


def test_section_splitter_uses_headings_conservatively():
    text = """
    Introduction
    We introduce the problem.
    Results are discussed informally in this sentence.
    Methods
    We sequenced samples.
    Future Work
    Future studies should examine longitudinal response.
    """

    sections = split_sections(text)

    assert "We introduce the problem." in sections["introduction"]
    assert "Results are discussed informally" in sections["introduction"]
    assert "We sequenced samples." in sections["methods"]
    assert "Future studies should examine" in sections["future"]


def test_section_splitter_excludes_reference_sections():
    text = """
    Discussion
    Future studies should examine longitudinal response.
    References
    Ldl cholesterol lowering in type 2 diabetes: what is the optimum approach?
    """

    sections = split_sections(text)

    assert "Future studies should examine" in sections["discussion"]
    assert "optimum approach" not in " ".join(sections.values())


def test_variable_inference_for_microbiome_question():
    variables = infer_required_variables(
        "Future studies should examine whether antibiotic exposure changes gut microbiome "
        "composition over longitudinal timepoints in IBD patients."
    )

    assert "antibiotic_exposure" in variables
    assert "microbiome_composition" in variables
    assert "timepoint" in variables
    assert "disease_activity" in variables


def test_population_inference_prefers_specific_groups():
    assert infer_population("human infant patient cohort") == "infant"
    assert infer_population("pediatric patient cohort") == "pediatric"
    assert infer_population("humanized mouse microbiome model") == "mouse"


def test_lexical_similarity_returns_zero_for_stopword_only_text():
    assert lexical_similarity("the and of", "with or to") == 0.0
