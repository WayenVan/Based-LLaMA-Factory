from datasets import load_dataset


dataset = load_dataset("sartifyllc/east_africa_language")


data = dataset["train"][0]

print(data)
