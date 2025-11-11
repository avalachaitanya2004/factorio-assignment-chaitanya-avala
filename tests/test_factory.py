import json
import subprocess
import sys

sample_input = {
  "machines": {
    "assembler_1": {"crafts_per_min": 30},
    "chemical": {"crafts_per_min": 60}
  },
  "recipes": {
    "iron_plate": {
      "machine": "chemical",
      "time_s": 3.2,
      "in": {"iron_ore": 1},
      "out": {"iron_plate": 1}
    },
    "copper_plate": {
      "machine": "chemical",
      "time_s": 3.2,
      "in": {"copper_ore": 1},
      "out": {"copper_plate": 1}
    },
    "green_circuit": {
      "machine": "assembler_1",
      "time_s": 0.5,
      "in": {"iron_plate": 1, "copper_plate": 3},
      "out": {"green_circuit": 1}
    }
  },
  "modules": {
    "assembler_1": {"prod": 0.1, "speed": 0.15},
    "chemical": {"prod": 0.2, "speed": 0.1}
  },
  "limits": {
    "raw_supply_per_min": {"iron_ore": 5000, "copper_ore": 5000},
    "max_machines": {"assembler_1": 300, "chemical": 300}
  },
  "target": {"item": "green_circuit", "rate_per_min": 1800}
}


def test_sample_factory(tmp_path):
    p = subprocess.run([sys.executable, 'factory/main.py'], input=json.dumps(sample_input).encode(), capture_output=True)
    assert p.returncode == 0
    out = json.loads(p.stdout)
    assert out['status'] == 'ok'
    assert abs(out['per_recipe_crafts_per_min']['green_circuit'] - 1800.0) < 1e-6
