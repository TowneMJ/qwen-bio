# save this as analyze_errors.py
import json

# Load the results
samples_file = "lm-evaluation-harness/results/qwen3-4b-baseline/Qwen__Qwen3-4B-Instruct-2507/samples_mmlu_pro_biology_2026-01-12T21-49-16.616598.jsonl"

correct = []
wrong = []

with open(samples_file, 'r') as f:
    for line in f:
        sample = json.loads(line)
        if sample['exact_match'] == 1.0:
            correct.append(sample)
        else:
            wrong.append(sample)

print(f"Total questions: {len(correct) + len(wrong)}")
print(f"Correct: {len(correct)} ({100*len(correct)/(len(correct)+len(wrong)):.1f}%)")
print(f"Wrong: {len(wrong)} ({100*len(wrong)/(len(correct)+len(wrong)):.1f}%)")

# Analyze sources of wrong answers
print("\n--- Wrong answers by source ---")
sources = {}
for sample in wrong:
    src = sample['doc'].get('src', 'unknown')
    sources[src] = sources.get(src, 0) + 1

for src, count in sorted(sources.items(), key=lambda x: -x[1]):
    print(f"  {src}: {count}")

# Show a few wrong answers
print("\n--- Sample wrong answers ---")
for sample in wrong[:5]:
    print(f"\nQ: {sample['doc']['question'][:200]}...")
    print(f"Options: {sample['doc']['options']}")
    print(f"Correct: {sample['doc']['answer']}, Model said: {sample['filtered_resps'][0]}")