import re
import glob
import pandas as pd

rows = []

for fname in glob.glob("logs/*_baseline_100_out.txt"):
    with open(fname) as f:
        text = f.read()

    strict = re.search(
        r"strict-match\s*\|\s*8\s*\|\s*exact_match\|↑\s*\|\s*([0-9.]+)",
        text
    )
    flexible = re.search(
        r"flexible-extract\s*\|\s*8\s*\|\s*exact_match\|↑\s*\|\s*([0-9.]+)",
        text
    )

    runtime = re.search(
        r"Total time:\s*([0-9]+)\s*seconds",
        text
    )

    memory = re.search(
        r"Peak GPU memory:\s*([0-9]+)",
        text
    )

    if (strict and flexible):
        rows.append({
            "file": fname,
            "strict_acc": float(strict.group(1)),
            "flexible_acc": float(flexible.group(1)),
            "runtime_sec": int(runtime.group(1)) if runtime else None,
            "peak_mem_mib": int(memory.group(1)) if memory else None,
        })


df = pd.DataFrame(rows)

print(df.describe())

df.to_csv("aggregate_results.csv", index=False)