"""
Prepare training data from MedMCQA dataset.
Filters for molecular biology-relevant topics and formats for axolotl training.
"""

from datasets import load_dataset
import json
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("./genetics_training_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Topics relevant to molecular genetics
RELEVANT_TOPICS = [
    'Molecular Genetics',
    'Transcription',
    'Metabolism of nucleic acids',
    'Techniques in molecular biology',
]


def format_question(d):
    """Convert a MedMCQA question to chat format."""
    options = f"A. {d['opa']}\nB. {d['opb']}\nC. {d['opc']}\nD. {d['opd']}"
    answer_letter = ['A', 'B', 'C', 'D'][d['cop']]
    answer_text = [d['opa'], d['opb'], d['opc'], d['opd']][d['cop']]
    
    user_msg = f"{d['question']}\n\n{options}"
    
    # Use explanation if available, otherwise just give answer
    if d['exp'] and len(d['exp']) > 10:
        assistant_msg = f"{d['exp']}\n\nThe answer is {answer_letter}. {answer_text}"
    else:
        assistant_msg = f"The answer is {answer_letter}. {answer_text}"
    
    return {
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    }


def main():
    print("Loading MedMCQA dataset...")
    ds = load_dataset('openlifescienceai/medmcqa', split='train')
    
    # Filter for biochemistry subject and relevant topics
    print("Filtering for molecular biology topics...")
    biochem = [d for d in ds if d['subject_name'] == 'Biochemistry']
    filtered = [d for d in biochem if d['topic_name'] in RELEVANT_TOPICS]
    
    print(f"Found {len(filtered)} questions in relevant topics:")
    for topic in RELEVANT_TOPICS:
        count = len([d for d in filtered if d['topic_name'] == topic])
        print(f"  - {topic}: {count}")
    
    # Format for training
    print("\nFormatting for chat training...")
    formatted = [format_question(d) for d in filtered]
    
    # Save to JSONL
    output_path = OUTPUT_DIR / "medmcqa_molgen.jsonl"
    with open(output_path, 'w') as f:
        for item in formatted:
            f.write(json.dumps(item) + '\n')
    
    print(f"\nSaved {len(formatted)} examples to {output_path}")
    
    # Show a sample
    print("\n" + "="*50)
    print("Sample formatted example:")
    print("="*50)
    print(json.dumps(formatted[0], indent=2))


if __name__ == "__main__":
    main()