import pytest


def test_sentiment_model_is_pinned(app_module):
    assert app_module.SENTIMENT_MODEL == "cardiffnlp/twitter-roberta-base-sentiment-latest"


@pytest.mark.parametrize(
    "text,expected_label",
    [
        ("I absolutely love this, best day of my life!", "positive"),
        ("This is terrible, I hate it and want a refund.", "negative"),
        ("The meeting is scheduled for 3pm on Tuesday.", "neutral"),
    ],
)
def test_predict_sentiment_labels(app_module, text, expected_label):
    label, score, breakdown = app_module.predict_sentiment(text)

    assert label == expected_label
    assert score > 0.5
    assert breakdown
