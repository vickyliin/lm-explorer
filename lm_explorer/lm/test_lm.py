def test_lm_factory():
    from . import LanguageModel
    fullname = 'gpt2/117M'
    model1 = LanguageModel(fullname)
    assert type(model1).__name__ == 'GPT2LanguageModel'
    model2 = LanguageModel(fullname)
    assert model1 == model2
