"""
Defense-based auto-review script for generated genetics questions.

Instead of looking for problems, asks Claude Opus to DEFEND each question.
If Opus can't confidently defend a question, it gets flagged for human review.

Outputs two files:
- v3_defended.jsonl: Questions Opus could confidently defend
- v3_cant_defend.jsonl: Questions Opus couldn't defend (flagged for review)
"""

import requests
import json
import time
import os
from pathlib import Path

# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "your-key-here")
REVIEWER_MODEL = "anthropic/claude-opus-4"
INPUT_FILE = Path("./genetics_training_data/v3_genetics_qa.jsonl")
OUTPUT_DIR = Path("./genetics_training_data")

DEFENSE_PROMPT = """You are a PhD molecular biologist. Your task is to DEFEND this multiple-choice question as suitable for an exam.

QUESTION:
{question}

OPTIONS:
{options}

STATED CORRECT ANSWER: {correct_answer}

---

Make the strongest case you can that:
1. The stated answer ({correct_answer}) is DEFINITIVELY correct
2. NO other option is defensible as correct
3. The question is clear and unambiguous

Really try to defend it. But be honest — if you cannot make a confident defense, say so.

Respond with JSON in this exact format:
{{
    "can_defend": true or false,
    "defense": "Your argument for why this question is solid" OR "Why you cannot defend it",
    "weak_points": ["Any reservations you have, even if you can still defend it overall"]
}}

Set "can_defend" to true ONLY if you can confidently argue that the stated answer is correct AND no other option is defensible.

Return ONLY the JSON, no other text."""


def load_questions(filepath: Path) -> list:
    """Load questions from JSONL file."""
    questions = []
    with open(filepath, "r") as f:
        for line in f:
            questions.append(json.loads(line))
    return questions


def format_options(options: dict) -> str:
    """Format options dict as string."""
    return "\n".join([f"{k}. {v}" for k, v in options.items()])


def defend_question(question: dict) -> dict | None:
    """Ask Opus to defend a question."""
    
    prompt = DEFENSE_PROMPT.format(
        question=question["question"],
        options=format_options(question["options"]),
        correct_answer=question["correct_answer"]
    )
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": REVIEWER_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 600,
                "temperature": 0.3,
            },
            timeout=90
        )
        
        if response.status_code != 200:
            print(f"API error: {response.status_code} - {response.text}")
            return None
            
        content = response.json()["choices"][0]["message"]["content"]
        
        # Parse JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        defense_data = json.loads(content.strip())
        return defense_data
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"Error defending question: {e}")
        return None


def main():
    print("Question Defense Review Script")
    print("="*50)
    print(f"Reviewer model: {REVIEWER_MODEL}")
    print(f"Input file: {INPUT_FILE}")
    
    # Load questions
    if not INPUT_FILE.exists():
        print(f"Error: Input file {INPUT_FILE} not found")
        return
    
    questions = load_questions(INPUT_FILE)
    print(f"Loaded {len(questions)} questions")
    
    defended = []
    cant_defend = []
    
    print("\n" + "="*50)
    print("Starting defense review...")
    print("="*50)
    
    for i, q in enumerate(questions):
        print(f"\nQuestion {i+1}/{len(questions)}...", end=" ")
        
        defense = defend_question(q)
        
        if defense is None:
            print("⚠ (defense failed, flagging for human review)")
            q["defense"] = {"can_defend": False, "defense": "Auto-defense failed"}
            cant_defend.append(q)
        elif defense.get("can_defend") == True:
            weak_points = defense.get("weak_points", [])
            if weak_points:
                print(f"✓ DEFENDED (with notes: {', '.join(weak_points[:2])})")
            else:
                print(f"✓ DEFENDED")
            q["defense"] = defense
            defended.append(q)
        else:
            reason = defense.get("defense", "unspecified")
            # Truncate long reasons for display
            if len(reason) > 80:
                reason = reason[:77] + "..."
            print(f"✗ CAN'T DEFEND - {reason}")
            q["defense"] = defense
            cant_defend.append(q)
        
        # Rate limiting
        time.sleep(1)
    
    # Save results
    defended_file = OUTPUT_DIR / "v3_defended.jsonl"
    cant_defend_file = OUTPUT_DIR / "v3_cant_defend.jsonl"
    
    with open(defended_file, "w") as f:
        for q in defended:
            f.write(json.dumps(q) + "\n")
    
    with open(cant_defend_file, "w") as f:
        for q in cant_defend:
            f.write(json.dumps(q) + "\n")
    
    print("\n" + "="*50)
    print("Defense review complete!")
    print("="*50)
    print(f"Defended: {len(defended)} questions → {defended_file}")
    print(f"Can't defend: {len(cant_defend)} questions → {cant_defend_file}")
    print(f"Defense rate: {100*len(defended)/len(questions):.0f}%")


if __name__ == "__main__":
    main()