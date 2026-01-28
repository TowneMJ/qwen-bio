"""
Improved Synthetic Genetics Training Data Generator v3

Generates chain-of-thought Q&A pairs for fine-tuning LLMs on genetics topics.
Uses OpenRouter API with improved prompts to reduce error rate.

Changes from v2:
- Stronger reasoning requirement (not just recall)
- Deduplicated/rebalanced topic list
- Anti-duplication instruction
"""

import requests
import json
import time
import os
from pathlib import Path

# Track core concepts to avoid duplicates
generated_concepts = []

# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "your-key-here")
MODEL = "anthropic/claude-sonnet-4"
OUTPUT_DIR = Path("./genetics_training_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Rebalanced molecular genetics topics - consolidated to avoid overlap
TOPICS = {
    "molecular_genetics": [
        # DNA-focused
        "DNA replication fork dynamics and coordination of enzymes",
        "DNA damage recognition and repair pathway selection",
        "Telomere maintenance and consequences of telomerase dysfunction",
        
        # RNA-focused (consolidated into one broader topic)
        "Regulation of gene expression from transcription through translation",
        
        # Protein synthesis
        "Ribosome assembly and translation quality control mechanisms",
        "Post-translational modifications and protein targeting",
        
        # Gene regulation
        "Chromatin remodeling and epigenetic inheritance",
        "Transcription factor interactions and combinatorial gene regulation",
        
        # Comparative/applied
        "Prokaryotic vs eukaryotic gene expression control points",
        "Experimental techniques for studying gene expression (PCR, blotting, sequencing)",
    ],
}

GENERATION_PROMPT = """You are an expert biology professor creating exam questions for advanced undergraduates. Your questions will be reviewed by a PhD molecular biologist, so accuracy is critical.

Generate a multiple-choice question about: {topic}

CRITICAL REQUIREMENTS:

1. ACCURACY FIRST: Only write questions where you are highly confident in the correct answer. If a topic is ambiguous or has competing valid interpretations, choose a different angle.

2. REASONING REQUIRED: The question MUST require integrating multiple concepts or applying principles to a scenario. 
   
   DO NOT write simple recall questions like:
   - "What is the function of X?"
   - "Which enzyme does Y?"
   - "What is the name of the process that does Z?"
   
   DO write questions that require reasoning. The following are example question structures, though questions should be original and not simply replicate these structures:
   - "A researcher observes X. What is the most likely explanation?"
   - "If mutation Y occurred, what would be the expected effect on Z?"
   - "Which of the following scenarios would result in X?"
   - "A patient has condition X. Which molecular mechanism is most likely disrupted?"

3. SIMPLE AND DIRECT: While the question should require reasoning, the language should be clear. Don't make it artificially complex.

4. AVOID ARITHMETIC: Do not write questions requiring multi-step calculations.

5. ONE CLEAR ANSWER: There must be exactly one defensible correct answer. All distractors must be clearly wrong to an expert.

6. ANSWER OPTIONS: Provide exactly 10 options (A-J) to match MMLU-Pro format. Distractors should represent plausible misconceptions or partial understanding.

7. REASONING STRUCTURE - Solve the problem step by step as a student would:
   - State what the question is asking
   - Recall the relevant biological principle(s)
   - Work through the logic: what must be true? what can we rule out?
   - Evaluate the options based on this reasoning
   - Arrive at the answer
   
   Do NOT start with the answer and work backward.

8. SELF-CHECK: Before outputting, verify:
   - Does your reasoning genuinely support your chosen answer?
   - Is there any option that could arguably be more correct?
   - Would a biology PhD agree with your answer?
   - Is this question testing reasoning, not just recall?

9. CORE CONCEPT TAG: Provide a short (3-5 word) tag identifying the specific concept being tested. This is used to prevent duplicate questions, so be specific.

   Examples of GOOD tags (specific):
   - "Dom34 ribosome rescue function"
   - "mRNA-protein level discrepancy"
   - "telomerase reverse transcriptase mechanism"
   - "nonsense-mediated decay triggers"
   
   Examples of BAD tags (too vague):
   - "gene regulation"
   - "translation quality control"
   - "DNA repair mechanisms"
   - "protein synthesis"

ALREADY COVERED CONCEPTS (do not repeat these):
{covered_concepts}

Output JSON with this exact structure:
{{
    "question": "The question text",
    "options": {{
        "A": "First option",
        "B": "Second option",
        "C": "Third option",
        "D": "Fourth option",
        "E": "Fifth option",
        "F": "Sixth option",
        "G": "Seventh option",
        "H": "Eighth option",
        "I": "Ninth option",
        "J": "Tenth option"
    }},
    "concept_tested": "One sentence describing what reasoning this question requires",
    "reasoning": "Step-by-step reasoning as described above",
    "correct_answer": "The letter (A-J)",
    "confidence": "high/medium/low",
    "topic": "{category}",
    "subtopic": "{topic}"
}}

Only output questions where your confidence is HIGH.

Return ONLY the JSON, no other text."""


def generate_question(category: str, topic: str) -> dict | None:
    """Generate a single genetics question using OpenRouter."""
    
    # Format covered concepts for prompt
    if generated_concepts:
        covered = "\n".join(f"- {c}" for c in generated_concepts)
    else:
        covered = "- None yet"
    
    prompt = GENERATION_PROMPT.format(category=category, topic=topic, covered_concepts=covered)
    
    try:
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 2500,
                "temperature": 0.7,
                "core_concept": "3-5 word specific concept tag"
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
            
        question_data = json.loads(content.strip())
        
        # Validate required fields
        required_fields = ["question", "options", "reasoning", "correct_answer", "confidence"]
        if not all(field in question_data for field in required_fields):
            print(f"Missing fields in response")
            return None
        
        # Only accept high-confidence questions
        if question_data.get("confidence", "").lower() != "high":
            print(f"Skipping low/medium confidence question")
            return None
        
        # Validate we have 10 options
        if len(question_data.get("options", {})) != 10:
            print(f"Wrong number of options: {len(question_data.get('options', {}))}")
            return None
            
        # Add metadata
        question_data["category"] = category
        question_data["subtopic"] = topic
        
        return question_data
        
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Content was: {content[:500]}...")
        return None
    except Exception as e:
        print(f"Error generating question: {e}")
        return None


def generate_dataset(questions_per_topic: int = 2, categories: list = None):
    """Generate a dataset of genetics questions."""
    
    if categories is None:
        categories = list(TOPICS.keys())
    
    all_questions = []
    
    for category in categories:
        print(f"\n{'='*50}")
        print(f"Generating questions for: {category}")
        print(f"{'='*50}")
        
        topics = TOPICS[category]
        
        for topic in topics:
            print(f"\n  Topic: {topic}")
            
            for i in range(questions_per_topic):
                print(f"    Generating question {i+1}/{questions_per_topic}...", end=" ")
                
                question = generate_question(category, topic)
                
                if question:
                    all_questions.append(question)
                    # Track concept to avoid duplicates
                    if "core_concept" in question:
                        generated_concepts.append(question["core_concept"])
                    print("✓")
                else:
                    print("✗ (failed/low confidence)")

                # Rate limiting
                time.sleep(1)
    
    return all_questions


def save_dataset(questions: list, filename: str = "v3_genetics_qa.jsonl"):
    """Save questions in JSONL format."""
    
    output_path = OUTPUT_DIR / filename
    
    with open(output_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    
    print(f"\nSaved {len(questions)} questions to {output_path}")
    return output_path


def convert_to_chat_format(questions: list, filename: str = "v3_genetics_chat.jsonl"):
    """Convert questions to chat format for instruction tuning."""
    
    output_path = OUTPUT_DIR / filename
    
    chat_examples = []
    
    for q in questions:
        # Format options as a string
        options_str = "\n".join([f"{k}. {v}" for k, v in q["options"].items()])
        
        # Create the user message (question)
        user_content = f"""Answer the following genetics question. Think through it step by step before giving your final answer.

Question: {q["question"]}

Options:
{options_str}"""
        
        # Create the assistant response (reasoning + answer)
        assistant_content = f"""{q["reasoning"]}

The answer is ({q["correct_answer"]})."""
        
        chat_example = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": assistant_content}
            ],
            "category": q.get("category", "genetics"),
            "subtopic": q.get("subtopic", "")
        }
        
        chat_examples.append(chat_example)
    
    with open(output_path, "w") as f:
        for example in chat_examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"Saved {len(chat_examples)} chat examples to {output_path}")
    return output_path


def main():
    """Main function to generate the dataset."""
    
    print("Genetics Training Data Generator v3")
    print("="*50)
    print(f"Model: {MODEL}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Topics ({len(TOPICS['molecular_genetics'])}):")
    for t in TOPICS['molecular_genetics']:
        print(f"  - {t}")
    
    # Generate 2 questions per topic = 20 questions for review
    print("\n" + "="*50)
    print(f"Starting generation (2 questions per topic, {len(TOPICS['molecular_genetics'])} topics = {2*len(TOPICS['molecular_genetics'])} questions)...")
    print("="*50)
    
    questions = generate_dataset(questions_per_topic=60)
    
    if questions:
        # Save raw Q&A format
        save_dataset(questions, "v3_genetics_qa.jsonl")
        
        # Save chat format for training
        convert_to_chat_format(questions, "v3_genetics_chat.jsonl")
        
        print(f"\n{'='*50}")
        print(f"Generation complete!")
        print(f"Total questions generated: {len(questions)}")
        expected = 2 * len(TOPICS['molecular_genetics'])
        print(f"Success rate: {len(questions)}/{expected} ({100*len(questions)/expected:.0f}%)")
        print(f"{'='*50}")
        
        # Show a sample
        if questions:
            print("\nSample question:")
            sample = questions[0]
            print(f"Q: {sample['question'][:200]}...")
            print(f"Concept: {sample.get('concept_tested', 'N/A')}")
            print(f"Answer: {sample['correct_answer']}")
            print(f"Confidence: {sample.get('confidence', 'N/A')}")
    else:
        print("No questions were generated successfully.")


if __name__ == "__main__":
    main()