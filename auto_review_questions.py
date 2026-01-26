"""
Auto-review script for generated genetics questions.

Uses Claude Opus 4.5 to review each question for:
- Multiple defensible answers
- Accuracy issues
- Reasoning that doesn't support the conclusion
- Ambiguous wording

Outputs two files:
- passed.jsonl: Questions that passed review
- needs_review.jsonl: Questions flagged for human review
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

REVIEW_PROMPT = """You are a PhD molecular biologist reviewing multiple-choice exam questions for accuracy and quality.

Review the following question and assess whether it has any issues:

QUESTION:
{question}

OPTIONS:
{options}

STATED CORRECT ANSWER: {correct_answer}

REASONING PROVIDED:
{reasoning}

---

Check for the following issues:

1. MULTIPLE DEFENSIBLE ANSWERS: Could a knowledgeable expert reasonably argue for a different answer than the stated correct one? Are any distractors actually correct or partially correct?

2. ACCURACY: Is the stated correct answer actually correct? Is the reasoning factually accurate? Are there any scientific errors?

3. REASONING SUPPORTS CONCLUSION: Does the provided reasoning actually lead to the stated answer, or does it contradict itself?

4. AMBIGUITY: Is the question wording clear? Could it be interpreted in multiple ways that would lead to different answers?

5. QUESTION QUALITY: Is this a good test of understanding, or is it flawed in some way?

Respond with JSON in this exact format:
{{
    "verdict": "PASS" or "FLAG",
    "confidence": "high" or "medium" or "low",
    "concerns": ["list", "of", "specific", "concerns"] or [],
    "notes": "Brief explanation of your assessment"
}}

If you have ANY uncertainty or concerns about accuracy or question quality, set verdict to "FLAG".
Only set verdict to "PASS" if you are confident the question is accurate and has exactly one defensible answer.

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


def review_question(question: dict) -> dict | None:
    """Send a question to Opus for review."""
    
    prompt = REVIEW_PROMPT.format(
        question=question["question"],
        options=format_options(question["options"]),
        correct_answer=question["correct_answer"],
        reasoning=question["reasoning"]
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
                "max_tokens": 500,
                "temperature": 0.3,  # Low temperature for consistent review
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
            
        review_data = json.loads(content.strip())
        return review_data
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        return None
    except Exception as e:
        print(f"Error reviewing question: {e}")
        return None


def main():
    print("Question Auto-Review Script")
    print("="*50)
    print(f"Reviewer model: {REVIEWER_MODEL}")
    print(f"Input file: {INPUT_FILE}")
    
    # Load questions
    if not INPUT_FILE.exists():
        print(f"Error: Input file {INPUT_FILE} not found")
        return
    
    questions = load_questions(INPUT_FILE)
    print(f"Loaded {len(questions)} questions")
    
    passed = []
    needs_review = []
    
    print("\n" + "="*50)
    print("Starting review...")
    print("="*50)
    
    for i, q in enumerate(questions):
        print(f"\nReviewing question {i+1}/{len(questions)}...", end=" ")
        
        review = review_question(q)
        
        if review is None:
            print("⚠ (review failed, flagging for human review)")
            q["review"] = {"verdict": "FLAG", "notes": "Auto-review failed"}
            needs_review.append(q)
        elif review.get("verdict") == "PASS":
            print(f"✓ PASS")
            q["review"] = review
            passed.append(q)
        else:
            concerns = review.get("concerns", [])
            print(f"⚑ FLAG - {', '.join(concerns) if concerns else review.get('notes', 'unspecified concern')}")
            q["review"] = review
            needs_review.append(q)
        
        # Rate limiting
        time.sleep(1)
    
    # Save results
    passed_file = OUTPUT_DIR / "v3_passed.jsonl"
    review_file = OUTPUT_DIR / "v3_needs_review.jsonl"
    
    with open(passed_file, "w") as f:
        for q in passed:
            f.write(json.dumps(q) + "\n")
    
    with open(review_file, "w") as f:
        for q in needs_review:
            f.write(json.dumps(q) + "\n")
    
    print("\n" + "="*50)
    print("Review complete!")
    print("="*50)
    print(f"Passed: {len(passed)} questions → {passed_file}")
    print(f"Needs review: {len(needs_review)} questions → {review_file}")
    print(f"Pass rate: {100*len(passed)/len(questions):.0f}%")


if __name__ == "__main__":
    main()