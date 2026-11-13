import sys; sys.path.insert(0, "model")
from prompts import build_messages

def test_build_messages_shape():
    msgs = build_messages("hi")
    assert msgs[0]["role"] == "system"
    assert msgs[-1] == {"role": "user", "content": "hi"}