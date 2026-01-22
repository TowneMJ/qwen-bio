import json

with open("genetics_training_data_v2/genetics_qa_v2.jsonl", "r") as f:
    questions = [json.loads(line) for line in f]

for i, q in enumerate(questions):
    print(f"\n{'='*60}")
    print(f"QUESTION {i+1} of {len(questions)}")
    print(f"Category: {q.get('category', 'N/A')} | Topic: {q.get('topic', 'N/A')}")
    print(f"{'='*60}")
    print(f"\n{q['question']}\n")
    
    for letter, option in q['options'].items():
        print(f"  {letter}. {option}")
    
    print(f"\nCORRECT ANSWER: {q['correct_answer']}")
    print(f"\nREASONING:\n{q['reasoning']}")
    
    response = input("\n[Enter] next | [f] flag as bad | [q] quit: ").strip().lower()
    
    if response == 'q':
        break
    elif response == 'f':
        print(f"  -> Flagged question {i+1}")
        with open("flagged_v2.txt", "a") as flag_file:
            flag_file.write(f"{i}\n")
