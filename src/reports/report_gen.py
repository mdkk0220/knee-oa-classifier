from pathlib import Path
from typing import Dict

def write_txt_report(results: Dict[str, float], out_path) -> str:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# Auto Evaluation Report', '']
    for k, v in results.items():
        lines.append(f'- {k}: {v}')
    p.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return str(p)
