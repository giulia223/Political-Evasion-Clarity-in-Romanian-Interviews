"""T3.2: neighbour turns and no label leakage."""
import pandas as pd

from polero.context import build_texts, neighbour_questions


def _interview():
    return pd.DataFrame({
        "datapointID": [10, 11, 12, 13, 14],
        "url": ["u"] * 5,
        "question_order": [1, 2, 4, 7, 8],  # gaps on purpose
        "interview_question": ["q1", "q2", "q4", "q7", "q8"],
        "interview_answer": ["a"] * 5,
        "president": ["P"] * 5, "context": ["Interview"] * 5, "date": ["2010-01-01"] * 5,
    })


def test_neighbours_with_gaps():
    df = _interview()
    test_part, train_part = df.iloc[[1, 3]], df.iloc[[0, 2, 4]]
    n = neighbour_questions([train_part, test_part])
    assert n[10] == ("", "q2") and n[11] == ("q1", "q4") and n[13] == ("q4", "q8") and n[14] == ("q7", "")


def test_no_label_columns_needed():
    df = _interview()  # no clarity/evasion columns at all: labels can't be read
    n = neighbour_questions([df])
    q, a = build_texts(df, {"context": "prev_next_q", "meta_prefix": True}, n)
    assert "q1" in q[1] and "q4" in q[1] and "Întrebare: q2" in q[1] and "[P | Interview | 2010]" in q[1]
    q0, _ = build_texts(df, {}, None)
    assert q0 == ["q1", "q2", "q4", "q7", "q8"]
