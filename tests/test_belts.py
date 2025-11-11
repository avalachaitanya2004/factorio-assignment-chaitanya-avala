import json
import subprocess
import sys

sample_input = {
  "edges": [
    {"from": "s1", "to": "a", "lo": 0, "hi": 900},
    {"from": "a", "to": "b", "lo": 0, "hi": 900},
    {"from": "b", "to": "sink", "lo": 0, "hi": 900},
    {"from": "s2", "to": "a", "lo": 0, "hi": 600},
    {"from": "a", "to": "c", "lo": 0, "hi": 600},
    {"from": "c", "to": "sink", "lo": 0, "hi": 600}
  ],
  "sources": {"s1": 900, "s2": 600},
  "sink": "sink"
}


def test_sample_belts(tmp_path):
    p = subprocess.run([sys.executable, 'belts/main.py'], input=json.dumps(sample_input).encode(), capture_output=True)
    assert p.returncode == 0
    out = json.loads(p.stdout)
    assert out['status'] == 'ok'
    assert abs(out['max_flow_per_min'] - 1500) < 1e-6
