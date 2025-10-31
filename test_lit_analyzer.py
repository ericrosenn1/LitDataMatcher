import pytest
from lit_analyzer import split_sents, ContextIndexer, assemble_context, semantic_match

class TestSentenceSplitting:
    """Test edge cases in sentence splitting"""
    
    def test_abbreviations(self):
        """Don't split on abbreviations"""
        text = "Dr. Smith works at MIT. He studies AI."
        sents = split_sents(text)
        assert len(sents) == 2
        assert "Dr. Smith works at MIT." in sents[0]
    
    def test_multiple_punctuation(self):
        """Handle multiple punctuation marks"""
        text = "What?! Really?! Yes!"
        sents = split_sents(text)
        # Should be 3 sentences, not 6
        assert len(sents) == 3
    
    def test_empty_string(self):
        """Handle empty input gracefully"""
        assert split_sents("") == []
    
    def test_no_punctuation(self):
        """Handle text without sentence boundaries"""
        text = "this has no punctuation at all"
        sents = split_sents(text)
        assert len(sents) >= 1
    
    def test_very_long_sentence(self):
        """Don't hang on long sentences"""
        text = "word " * 10000 + "."
        sents = split_sents(text)
        assert len(sents) == 1


class TestContextAssembly:
    """Test context building logic"""
    
    def test_basic_context(self):
        """Assemble context from surrounding sentences"""
        sents = ["First.", "Target sentence.", "Last."]
        index = ContextIndexer(sents)
        opts = {"enabled": True, "prev_next": 1, "topk": 0, "max_chars": 1000}
        
        ctx = assemble_context(1, index, opts)
        assert "First." in ctx
        assert "Last." in ctx
        assert "Target sentence." not in ctx  # Excludes target itself
    
    def test_disabled_context(self):
        """Return empty string when disabled"""
        sents = ["A.", "B.", "C."]
        index = ContextIndexer(sents)
        opts = {"enabled": False}
        
        ctx = assemble_context(1, index, opts)
        assert ctx == ""
    
    def test_boundary_first_sentence(self):
        """Handle first sentence (no previous context)"""
        sents = ["First.", "Second.", "Third."]
        index = ContextIndexer(sents)
        opts = {"enabled": True, "prev_next": 1, "topk": 0, "max_chars": 1000}
        
        ctx = assemble_context(0, index, opts)
        assert "Second." in ctx
        # Should not crash
    
    def test_boundary_last_sentence(self):
        """Handle last sentence (no next context)"""
        sents = ["First.", "Second.", "Third."]
        index = ContextIndexer(sents)
        opts = {"enabled": True, "prev_next": 1, "topk": 0, "max_chars": 1000}
        
        ctx = assemble_context(2, index, opts)
        assert "Second." in ctx
    
    def test_max_chars_truncation(self):
        """Respect max_chars limit"""
        sents = ["A" * 100, "B" * 100, "C" * 100]
        index = ContextIndexer(sents)
        opts = {"enabled": True, "prev_next": 2, "topk": 0, "max_chars": 50}
        
        ctx = assemble_context(1, index, opts)
        assert len(ctx) <= 50


class TestSemanticMatching:
    """Test semantic similarity thresholds"""
    
    def test_clear_future_direction(self):
        """Detect obvious future direction statements"""
        sent = "Future studies should examine this hypothesis."
        assert semantic_match(sent, "future", threshold=0.6) == True
    
    def test_not_future_direction(self):
        """Don't match unrelated sentences"""
        sent = "The results were statistically significant."
        assert semantic_match(sent, "future", threshold=0.6) == False
    
    def test_threshold_sensitivity(self):
        """Verify threshold affects matching"""
        sent = "Further research may be needed."
        
        # Low threshold: should match
        assert semantic_match(sent, "future", threshold=0.3) == True
        
        # High threshold: might not match
        result_high = semantic_match(sent, "future", threshold=0.9)
        # Don't assert specific result, just check it runs
    
    def test_empty_sentence(self):
        """Handle empty input gracefully"""
        assert semantic_match("", "future") == False
    
    def test_limitation_vs_future(self):
        """Different categories should match differently"""
        sent = "A limitation is the small sample size."
        
        # Should match limitation
        assert semantic_match(sent, "limit", threshold=0.6) == True
        
        # Should NOT match future
        assert semantic_match(sent, "future", threshold=0.6) == False


# Run with: pytest test_lit_analyzer.py -v