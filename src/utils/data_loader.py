from transformers import AutoTokenizer

def load_and_tokenize(dataset, tokenizer_name="nomic-ai/gpt4all-j", max_length=128):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize(batch):
        return tokenizer(
            batch["input"],
            truncation=True,
            padding="max_length",
            max_length=max_length
        )

    return dataset.map(tokenize, batched=True), tokenizer
