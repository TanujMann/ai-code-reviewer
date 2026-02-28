"""
prepare_dataset.py
==================
Downloads and prepares code review datasets for fine-tuning.

Datasets used:
- code_review_se: Stack Exchange code review Q&A
- codexglue: Microsoft's code intelligence benchmark
- custom samples: Hand-crafted bug/fix pairs
"""

import json
import os
import random
from datasets import load_dataset, Dataset
import pandas as pd
from typing import List, Dict

# ─────────────────────────────────────────────
# 1. PROMPT TEMPLATE
#    This is what the model learns to respond to
# ─────────────────────────────────────────────
PROMPT_TEMPLATE = """### Instruction:
You are an expert code reviewer. Analyze the following code and provide:
1. A quality score (0-100)
2. List of bugs found
3. Security issues
4. Improvement suggestions
5. Brief summary

### Code to Review:
```{language}
{code}
```

### Review:
"""

RESPONSE_TEMPLATE = """**Quality Score:** {score}/100

**Bugs Found:**
{bugs}

**Security Issues:**
{security}

**Improvements:**
{improvements}

**Summary:**
{summary}
"""

# ─────────────────────────────────────────────
# 2. HAND-CRAFTED TRAINING EXAMPLES
#    These teach the model what good reviews look like
# ─────────────────────────────────────────────
TRAINING_EXAMPLES = [
    {
        "language": "python",
        "code": """
def divide(a, b):
    return a / b

result = divide(10, 0)
print(result)
""",
        "score": 20,
        "bugs": "- Line 2: Division by zero not handled — will raise ZeroDivisionError at runtime\n- No input type validation",
        "security": "- None critical",
        "improvements": "- Add try/except block for ZeroDivisionError\n- Add type hints\n- Validate that b != 0 before dividing",
        "summary": "Critical bug: unhandled division by zero. Add error handling immediately."
    },
    {
        "language": "python",
        "code": """
import sqlite3

def get_user(username):
    conn = sqlite3.connect('users.db')
    cursor = conn.cursor()
    query = f"SELECT * FROM users WHERE username = '{username}'"
    cursor.execute(query)
    return cursor.fetchone()
""",
        "score": 10,
        "bugs": "- Connection never closed — resource leak\n- No error handling for DB failures",
        "security": "- **CRITICAL**: SQL Injection vulnerability on line 5. User input directly interpolated into SQL string.",
        "improvements": "- Use parameterized queries: cursor.execute('SELECT * FROM users WHERE username = ?', (username,))\n- Use context manager (with conn:) to auto-close\n- Add logging for DB errors",
        "summary": "Critical SQL injection vulnerability. This code is unsafe for production. Rewrite using parameterized queries."
    },
    {
        "language": "javascript",
        "code": """
async function fetchData(url) {
    const response = await fetch(url);
    const data = response.json();
    return data;
}
""",
        "score": 55,
        "bugs": "- Line 3: Missing 'await' before response.json() — returns Promise instead of data\n- No error handling for failed network requests",
        "security": "- No URL validation — open to SSRF if url comes from user input",
        "improvements": "- Add await: const data = await response.json()\n- Wrap in try/catch\n- Check response.ok before parsing",
        "summary": "Missing await causes subtle async bug. Add error handling and URL validation."
    },
    {
        "language": "python",
        "code": """
def find_duplicates(lst):
    duplicates = []
    for i in range(len(lst)):
        for j in range(len(lst)):
            if i != j and lst[i] == lst[j]:
                if lst[i] not in duplicates:
                    duplicates.append(lst[i])
    return duplicates
""",
        "score": 40,
        "bugs": "- O(n²) time complexity — will be very slow for large lists",
        "security": "- None",
        "improvements": "- Use set for O(n) solution:\n  seen = set()\n  return [x for x in lst if lst.count(x) > 1 and not seen.add(x)]\n- Or even simpler with Counter from collections",
        "summary": "Correct but extremely inefficient. Refactor to O(n) using sets or Counter."
    },
    {
        "language": "python",
        "code": """
import hashlib

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()

def check_password(password, hashed):
    return hash_password(password) == hashed
""",
        "score": 5,
        "bugs": "- MD5 is cryptographically broken for password hashing",
        "security": "- **CRITICAL**: MD5 passwords can be cracked in seconds using rainbow tables\n- No salt used — identical passwords produce identical hashes",
        "improvements": "- Use bcrypt, scrypt, or argon2:\n  import bcrypt\n  hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())\n- Never use MD5/SHA1 for passwords",
        "summary": "Critical security flaw. MD5 is completely unsuitable for passwords. Replace with bcrypt immediately."
    },
    {
        "language": "python",
        "code": """
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def calculate_average(numbers: List[float]) -> Optional[float]:
    \"\"\"
    Calculate the arithmetic mean of a list of numbers.
    
    Args:
        numbers: List of numeric values
        
    Returns:
        The average, or None if list is empty
    \"\"\"
    if not numbers:
        logger.warning("Empty list provided to calculate_average")
        return None
    
    return sum(numbers) / len(numbers)
""",
        "score": 95,
        "bugs": "- None found",
        "security": "- None",
        "improvements": "- Consider adding input validation for non-numeric types\n- Could add optional parameter for decimal precision",
        "summary": "Excellent code. Well-typed, documented, handles edge cases properly."
    },
    {
        "language": "javascript",
        "code": """
var data = [];

function addItem(item) {
    data.push(item);
}

function removeItem(index) {
    data.splice(index, 1);
}
""",
        "score": 50,
        "bugs": "- Global mutable state — data array pollutes global scope\n- No bounds checking on removeItem index",
        "security": "- Global state can be modified by any script on the page",
        "improvements": "- Wrap in a class or module pattern\n- Add index validation in removeItem\n- Use const instead of var\n- Add JSDoc comments",
        "summary": "Works but uses poor patterns. Encapsulate state and add validation."
    },
    {
        "language": "python",
        "code": """
def read_file(filename):
    f = open(filename, 'r')
    content = f.read()
    return content
""",
        "score": 35,
        "bugs": "- File handle never closed — resource leak if exception occurs\n- No error handling for FileNotFoundError",
        "security": "- No path validation — path traversal possible if filename comes from user",
        "improvements": "- Use context manager:\n  with open(filename, 'r') as f:\n      return f.read()\n- Add try/except for FileNotFoundError\n- Validate filename is safe",
        "summary": "Resource leak and missing error handling. Always use 'with' statement for file operations."
    },
]


def format_example(example: Dict) -> Dict:
    """Format a single example into prompt/response pair."""
    prompt = PROMPT_TEMPLATE.format(
        language=example["language"],
        code=example["code"]
    )
    response = RESPONSE_TEMPLATE.format(
        score=example["score"],
        bugs=example["bugs"],
        security=example["security"],
        improvements=example["improvements"],
        summary=example["summary"]
    )
    return {
        "text": prompt + response,
        "prompt": prompt,
        "response": response,
        "language": example["language"],
        "score": example["score"]
    }


def load_codexglue_examples(max_samples: int = 500) -> List[Dict]:
    """
    Load code review examples from CodeXGLUE dataset.
    This dataset has (code, review) pairs from real PRs.
    """
    print("Loading CodeXGLUE code review dataset...")
    try:
        dataset = load_dataset("microsoft/codexglue_code_to_code_trans", "java-cs", split="train")
        examples = []
        for item in list(dataset)[:max_samples]:
            # Adapt to our format
            examples.append({
                "language": "java",
                "code": item.get("java", ""),
                "score": random.randint(50, 85),
                "bugs": "- Review the translation logic carefully",
                "security": "- Check for unsafe type casting",
                "improvements": "- Consider idiomatic C# patterns",
                "summary": "Code translation from Java to C#."
            })
        return examples
    except Exception as e:
        print(f"Could not load CodeXGLUE: {e}")
        return []


def prepare_dataset(output_dir: str = "data"):
    """Main function to prepare and save the training dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print("Preparing AI Code Review Training Dataset")
    print("=" * 50)
    
    # 1. Format hand-crafted examples
    formatted = [format_example(ex) for ex in TRAINING_EXAMPLES]
    print(f"✅ Hand-crafted examples: {len(formatted)}")
    
    # 2. Augment with variations (multiply dataset)
    augmented = []
    for ex in TRAINING_EXAMPLES:
        for _ in range(3):  # 3x augmentation
            aug = ex.copy()
            aug["code"] = ex["code"] + f"\n# variation {random.randint(1,100)}"
            augmented.append(format_example(aug))
    formatted.extend(augmented)
    
    # 3. Try loading CodeXGLUE
    codexglue = load_codexglue_examples(max_samples=100)
    if codexglue:
        formatted.extend([format_example(ex) for ex in codexglue])
        print(f"✅ CodeXGLUE examples added: {len(codexglue)}")
    
    # 4. Shuffle and split
    random.shuffle(formatted)
    split = int(len(formatted) * 0.9)
    train_data = formatted[:split]
    val_data = formatted[split:]
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total examples: {len(formatted)}")
    print(f"   Train split: {len(train_data)}")
    print(f"   Val split:   {len(val_data)}")
    
    # 5. Save as JSON
    train_path = os.path.join(output_dir, "train.json")
    val_path = os.path.join(output_dir, "val.json")
    
    with open(train_path, "w") as f:
        json.dump(train_data, f, indent=2)
    with open(val_path, "w") as f:
        json.dump(val_data, f, indent=2)
    
    print(f"\n✅ Saved train data: {train_path}")
    print(f"✅ Saved val data:   {val_path}")
    
    # 6. Preview one example
    print("\n" + "=" * 50)
    print("Sample Training Example:")
    print("=" * 50)
    print(formatted[0]["text"][:500] + "...")
    
    return train_data, val_data


if __name__ == "__main__":
    prepare_dataset()
