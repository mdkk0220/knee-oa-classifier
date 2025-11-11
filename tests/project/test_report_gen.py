from pathlib import Path
from src.reports.report_gen import write_txt_report

def test_write_txt_report(tmp_path):
    out_path = tmp_path / "report.txt"
    metrics = {"acc": 0.98, "f1": 0.95}

    # 인자 순서 수정: (metrics, out_path)
    write_txt_report(metrics, out_path)

    assert out_path.exists()
    text = out_path.read_text(encoding="utf-8")
    assert "acc" in text and "0.98" in text
