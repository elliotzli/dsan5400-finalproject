import pytest
import os
import sys
sys.path.append(os.getcwd())

from autocomment.utils.generate_reviews import GPT2Generator

Generate = GPT2Generator()

def test_generate_reviews_forGPT():
    assert Generate.generate_reviews_forGPT("this commodity is very good,") == "this commodity is very good, i love it."
    assert Generate.generate_reviews_forGPT("I enjoyed using this product,") == "I enjoyed using this product, it is very good."
    assert Generate.generate_reviews_forGPT("not satisfied with the quality of") == "not satisfied with the quality of the product."

def test_clean_text():
    assert Generate.clean_text("this commodity is very good,") == "commodity good love"
    assert Generate.clean_text("I enjoyed using this product,") == "enjoyed using product good"
    assert Generate.clean_text("not satisfied with the quality of") == "satisfied quality product"
