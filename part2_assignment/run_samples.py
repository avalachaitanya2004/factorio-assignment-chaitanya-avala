"""Run the sample inputs through the two CLIs and print outputs to files.
Usage: python run_samples.py
"""
import json
import subprocess

factory_cmd = "python factory/main.py"
belts_cmd = "python belts/main.py"

factory_sample = {
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

belts_sample = {
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

with open('factory_sample_in.json', 'w') as f:
    json.dump(factory_sample, f, indent=2)
with open('belts_sample_in.json', 'w') as f:
    json.dump(belts_sample, f, indent=2)

print('Running factory sample...')
with open('factory_sample_in.json','r') as fin:
    out = subprocess.run(factory_cmd.split(), stdin=fin, capture_output=True)
    print(out.stdout.decode())

print('Running belts sample...')
with open('belts_sample_in.json','r') as fin:
    out = subprocess.run(belts_cmd.split(), stdin=fin, capture_output=True)
    print(out.stdout.decode())