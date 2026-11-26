import torch
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "openai-community/gpt2"

print("Loading GPT-2 model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

model.eval()

print("Model loaded successfully!")

prompts = [
    "The cat is",
    "Artificial intelligence is",
    "India is a",
    "Machine learning helps"
]

TOP_K = 5
results = []

for prompt in prompts:

    print("\n" + "=" * 60)
    print("PROMPT:", prompt)
    print("=" * 60)

    inputs = tokenizer(prompt, return_tensors="pt")

    print("Input Token IDs:", inputs["input_ids"][0].tolist())

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0, -1, :]

    probabilities = torch.softmax(logits, dim=-1)

    top_probs, top_ids = torch.topk(probabilities, TOP_K)

    print("\nTop", TOP_K, "Next-Token Candidates:")
    print("-" * 50)

    for rank, (token_id, probability) in enumerate(
        zip(top_ids, top_probs), start=1
    ):

        token = tokenizer.decode([token_id.item()])
        prob = probability.item()

        print(
            f"Rank: {rank} | "
            f"Token: {repr(token)} | "
            f"Probability: {prob:.6f}"
        )

        results.append([
            prompt,
            rank,
            token,
            token_id.item(),
            round(prob, 6)
        ])

with open(
    "week7_probability_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(file)

    writer.writerow([
        "Prompt",
        "Rank",
        "Candidate Token",
        "Token ID",
        "Probability"
    ])

    writer.writerows(results)

print("\nResults saved to week7_probability_results.csv")