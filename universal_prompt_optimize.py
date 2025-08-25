# universal_prompt_optimize.py

import re
import matplotlib.pyplot as plt
import spacy
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Load spaCy and T5
# -----------------------------
nlp = spacy.load("en_core_web_sm")
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# -----------------------------
# NLP-based filler removal
# -----------------------------
def dynamic_filler_removal(prompt):
    """
    Keeps content words (nouns, proper nouns, verbs, adjectives, numbers)
    and essential connecting words for readability.
    """
    doc = nlp(prompt)
    keep_pos = ["NOUN", "PROPN", "VERB", "ADJ", "NUM"]
    essential_words = {"of", "in", "on", "for", "and", "compared", "to", "with", "from"}
    
    optimized_tokens = [
        token.text for token in doc
        if token.pos_ in keep_pos or token.text.lower() in essential_words
    ]
    
    return " ".join(optimized_tokens)

# -----------------------------
# Prompt optimizer
# -----------------------------
def optimize_prompt(prompt):
    orig_tokens = len(tokenizer(prompt)["input_ids"])
    
    if orig_tokens <= 25:
        # Short prompt: NLP-based filler removal
        optimized = dynamic_filler_removal(prompt)
    else:
        # Long prompt: NLP simplification + mild T5 summarization
        simplified = dynamic_filler_removal(prompt)
        target_tokens = max(10, int(orig_tokens * 0.6))  # reduce ~40%
        inputs = tokenizer("summarize: " + simplified, return_tensors="pt", truncation=True)
        outputs = model.generate(
            **inputs,
            max_length=target_tokens,
            min_length=max(8, target_tokens // 2),
            length_penalty=1.5,
            num_beams=5,
            early_stopping=True
        )
        optimized = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Fallback: simplified if T5 fails to reduce
        if len(tokenizer(optimized)["input_ids"]) >= len(tokenizer(simplified)["input_ids"]):
            optimized = simplified
    
    return optimized

# -----------------------------
# Sample prompts
# -----------------------------
prompts = [
    "Can you please, in a detailed manner, explain the benefits of renewable energy sources for reducing carbon footprint?",
    "I would like you to provide a comprehensive overview of blockchain technology and how it can be applied beyond cryptocurrencies.",
    "Could you kindly summarize the main points of the latest AI research paper published in Nature?",
    "Please describe in detail the potential advantages and disadvantages of edge computing compared to cloud computing.",
    "I want you to elaborate on the various types of neural networks used in natural language processing and their applications."
]

# -----------------------------
# Run optimization
# -----------------------------
optimized_prompts = []
token_before = []
token_after = []

for p in prompts:
    optimized = optimize_prompt(p)
    optimized_prompts.append(optimized)
    token_before.append(len(tokenizer(p)["input_ids"]))
    token_after.append(len(tokenizer(optimized)["input_ids"]))

# -----------------------------
# Display comparison
# -----------------------------
print("\nPrompt Optimization Results:\n")
for i, p in enumerate(prompts):
    reduction = token_before[i] - token_after[i]
    percent = 100 * reduction / token_before[i]
    print(f"Original ({token_before[i]} tokens): {p}")
    print(f"Optimized ({token_after[i]} tokens): {optimized_prompts[i]}")
    print(f"Token Reduction: {reduction} tokens ({percent:.1f}%)\n")

# -----------------------------
# Plot token reduction
# -----------------------------
x = range(len(prompts))
plt.bar(x, token_before, width=0.3, label='Original Tokens', alpha=0.7)
plt.bar([i + 0.3 for i in x], token_after, width=0.3, label='Optimized Tokens', alpha=0.7)
plt.xticks([i + 0.15 for i in x], [f"Prompt {i+1}" for i in x], rotation=45)
plt.ylabel("Number of Tokens")
plt.title("Universal Prompt Optimizer - Token Reduction Demo")
plt.legend()
plt.tight_layout()
plt.show()
