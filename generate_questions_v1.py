"""
Synthetic Genetics Training Data Generator

Generates chain-of-thought Q&A pairs for fine-tuning LLMs on genetics topics.
Uses OpenRouter API to generate high-quality training examples.
"""

import requests
import json
import random
import time
import os
from pathlib import Path

# Configuration
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "your-key-here")
MODEL = "anthropic/claude-sonnet-4"  # Good balance of quality and cost
OUTPUT_DIR = Path("./genetics_training_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# Genetics topics to generate questions about - focused on areas where model struggled
TOPICS = {
    "molecular_genetics": [
        "DNA replication mechanisms and enzymes",
        "Transcription and RNA processing (splicing, capping, polyadenylation)",
        "Translation and protein synthesis",
        "cDNA synthesis and reverse transcription",
        "mRNA processing and post-transcriptional modifications",
        "Differences between prokaryotic and eukaryotic gene expression",
        "Introns, exons, and splicing mechanisms",
        "DNA repair mechanisms",
        "Telomeres and telomerase",
        "Chromatin structure and gene regulation",
    ],
    "classical_genetics": [
        "Mendelian inheritance patterns",
        "Punnett squares and probability calculations",
        "Incomplete dominance and codominance",
        "Multiple alleles and blood types",
        "Sex-linked inheritance",
        "Epistasis and gene interactions",
        "Pedigree analysis",
        "Test crosses and phenotype ratios",
        "Linked genes and recombination frequency",
        "Genetic mapping and chromosome maps",
    ],
    "population_genetics": [
        "Hardy-Weinberg equilibrium calculations",
        "Allele frequency changes",
        "Genetic drift and founder effect",
        "Natural selection and fitness",
        "Gene flow and migration",
        "Heterozygote advantage",
        "Inbreeding and its effects",
        "Effective population size",
        "Selection coefficients",
        "Mutation-selection balance",
    ],
    "mutations_and_variation": [
        "Point mutations (missense, nonsense, silent)",
        "Frameshift mutations",
        "Chromosomal mutations (deletions, duplications, inversions, translocations)",
        "Aneuploidy and polyploidy",
        "Trinucleotide repeat disorders",
        "Transposons and mobile genetic elements",
        "Mutation rates and mutagens",
        "Somatic vs germline mutations",
        "Effects of mutations on protein function",
        "Genetic diseases and inheritance patterns",
    ],
}

GENERATION_PROMPT = """You are an expert genetics professor creating challenging multiple-choice questions for graduate-level students.

Generate a multiple-choice question about: {topic}

Requirements:
1. The question should require REASONING, not just fact recall
2. Include exactly 8 answer options (A through H)
3. Make the wrong answers plausible - they should represent common misconceptions
4. Provide detailed step-by-step reasoning that works through the problem
5. The reasoning should explicitly consider why wrong answers are wrong

Format your response as JSON with this exact structure:
{{
    "question": "The full question text",
    "options": {{
        "A": "First option",
        "B": "Second option", 
        "C": "Third option",
        "D": "Fourth option",
        "E": "Fifth option",
        "F": "Sixth option",
        "G": "Seventh option",
        "H": "Eighth option"
    }},
    "thinking": "Step-by-step reasoning that a student should use to solve this problem. Start with what we know, work through the logic, consider each option, and arrive at the answer.",
    "correct_answer": "The letter of the correct answer (A-H)",
    "topic": "{category}",
    "subtopic": "{topic}"
}}

Return ONLY the JSON, no other text."""


def generate_question(category: str, topic: str) -> dict | None:
    """Generate a single genetics question using OpenRouter."""
    
    prompt = GENERATION_PROMPT.format(category=category, topic=topic)
    
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
                "max_tokens": 2000,
                "temperature": 0.8,  # Some creativity for diverse questions
            },
            timeout=60
        )
        
        if response.status_code != 200:
            print(f"API error: {response.status_code} - {response.text}")
            return None
            
        content = response.json()["choices"][0]["message"]["content"]
        
        # Parse JSON from response
        # Handle potential markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            
        question_data = json.loads(content.strip())
        
        # Validate required fields
        required_fields = ["question", "options", "thinking", "correct_answer"]
        if not all(field in question_data for field in required_fields):
            print(f"Missing fields in response")
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


def generate_dataset(questions_per_topic: int = 5, categories: list = None):
    """Generate a full dataset of genetics questions."""
    
    if categories is None:
        categories = list(TOPICS.keys())
    
    all_questions = []
    total_cost = 0
    
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
                    print("✓")
                else:
                    print("✗ (failed)")
                
                # Rate limiting - be nice to the API
                time.sleep(0.5)
    
    return all_questions


def save_dataset(questions: list, filename: str = "genetics_qa.jsonl"):
    """Save questions in JSONL format for training."""
    
    output_path = OUTPUT_DIR / filename
    
    with open(output_path, "w") as f:
        for q in questions:
            f.write(json.dumps(q) + "\n")
    
    print(f"\nSaved {len(questions)} questions to {output_path}")
    return output_path


def convert_to_chat_format(questions: list, filename: str = "genetics_chat.jsonl"):
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
        
        # Create the assistant response (thinking + answer)
        assistant_content = f"""{q["thinking"]}

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
    
    print("Genetics Training Data Generator")
    print("="*50)
    print(f"Model: {MODEL}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Categories: {list(TOPICS.keys())}")
    print(f"Topics per category: {[len(t) for t in TOPICS.values()]}")
    
    # Start with a small test run
    print("\n" + "="*50)
    print("Starting generation (5 questions per topic)...")
    print("="*50)
    
    questions = generate_dataset(questions_per_topic=1)
    
    if questions:
        # Save raw Q&A format
        save_dataset(questions, "genetics_qa.jsonl")
        
        # Save chat format for training
        convert_to_chat_format(questions, "genetics_chat.jsonl")
        
        print(f"\n{'='*50}")
        print(f"Generation complete!")
        print(f"Total questions generated: {len(questions)}")
        print(f"{'='*50}")
        
        # Show a sample
        if questions:
            print("\nSample question:")
            sample = questions[0]
            print(f"Q: {sample['question'][:200]}...")
            print(f"Answer: {sample['correct_answer']}")
    else:
        print("No questions were generated successfully.")


if __name__ == "__main__":
    main()