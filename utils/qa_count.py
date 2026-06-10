"""題組 qa_count（出題數上限）共用常數與正規化。"""

QA_COUNT_MIN = 1
QA_COUNT_MAX = 20
QA_COUNT_DEFAULT = 1


def normalize_qa_count(value: object) -> int:
    """將 DB 或外部值正規化為 1–20；無效時回傳預設 1。"""
    try:
        n = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        n = QA_COUNT_DEFAULT
    return max(QA_COUNT_MIN, min(QA_COUNT_MAX, n))
