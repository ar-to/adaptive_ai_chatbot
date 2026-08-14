import pytest


def test_emotion_model_is_pinned(app_module):
    assert app_module.EMOTION_MODEL == "j-hartmann/emotion-english-distilroberta-base"


@pytest.mark.parametrize(
    "text,expected_label",
    [
        ("I just won the lottery, I'm overjoyed!", "joy"),
        ("I'm so scared, something is chasing me in the dark.", "fear"),
        ("My dog passed away this morning and I can't stop crying.", "sadness"),
    ],
)
def test_predict_emotion_labels(app_module, text, expected_label):
    label, score, breakdown = app_module.predict_emotion(text)

    assert label == expected_label
    assert score > 0.5
    assert breakdown
