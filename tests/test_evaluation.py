from pathlib import Path

from evaluation.metrics import chunk_recall, document_hit, keyword_coverage
from evaluation.report import build_report, write_report
from evaluation.run import load_questions


def test_keyword_coverage():
    coverage = keyword_coverage(
        ["劳动合同", "试用期", "劳动合同解除"],
        "本回答包含劳动合同和试用期。",
    )

    assert coverage == 2 / 3


def test_chunk_recall():
    sources = [
        {"chunk_index": 2},
        {"chunk_index": 3},
        {"chunk_index": 8},
    ]

    recall = chunk_recall([2, 3, 4], sources)

    assert recall == 2 / 3


def test_document_hit():
    sources = [
        {
            "source": "中国劳动法.pdf",
            "metadata": {
                "source": "中国劳动法.pdf",
            },
        }
    ]

    assert document_hit(["中国劳动法.pdf"], sources) is True
    assert document_hit(["Android.txt"], sources) is False


def test_report_generation(tmp_path: Path):
    report = build_report(
        [
            {
                "id": "q1",
                "pass": True,
                "chunk_recall": 1.0,
                "keyword_coverage": 0.8,
                "latency_ms": 100.0,
            },
            {
                "id": "q2",
                "pass": False,
                "chunk_recall": 0.5,
                "keyword_coverage": 0.2,
                "latency_ms": 300.0,
            },
        ]
    )
    output_path = tmp_path / "report.json"
    write_report(report, output_path)

    assert report["total"] == 2
    assert report["pass"] == 1
    assert report["fail"] == 1
    assert report["average_latency_ms"] == 200.0
    assert output_path.exists()


def test_yaml_loader(tmp_path: Path):
    dataset_path = tmp_path / "questions.yaml"
    dataset_path.write_text(
        """
- id: labor_law_001
  question: 劳动法第二章说什么
  expected_documents:
    - 中国劳动法.pdf
  expected_chunks:
    - 2
    - 3
  expected_keywords:
    - 劳动合同
    - 试用期
  knowledge_base_id: 4
  top_k: 6
  score_threshold: 0.5
""".strip(),
        encoding="utf-8",
    )

    questions = load_questions(dataset_path)

    assert len(questions) == 1
    assert questions[0].id == "labor_law_001"
    assert questions[0].expected_documents == ["中国劳动法.pdf"]
    assert questions[0].expected_chunks == [2, 3]
    assert questions[0].expected_keywords == ["劳动合同", "试用期"]
    assert questions[0].knowledge_base_id == 4
    assert questions[0].top_k == 6
    assert questions[0].score_threshold == 0.5
