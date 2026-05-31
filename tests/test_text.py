from litdatamatcher.text import infer_required_variables, split_sections, split_sentences


def test_sentence_splitter_handles_abbreviations_and_punctuation():
    text = "Dr. Smith works at MIT. What?! Really?! Yes!"

    sentences = split_sentences(text)

    assert sentences[0] == "Dr. Smith works at MIT."
    assert sentences[1:] == ["What?!", "Really?!", "Yes!"]


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


def test_variable_inference_for_microbiome_question():
    variables = infer_required_variables(
        "Future studies should examine whether antibiotic exposure changes gut microbiome "
        "composition over longitudinal timepoints in IBD patients."
    )

    assert "antibiotic_exposure" in variables
    assert "microbiome_composition" in variables
    assert "timepoint" in variables
    assert "disease_activity" in variables
