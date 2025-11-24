from datasets import load_dataset


dataset = load_dataset("sartifyllc/east_africa_language")
dataset = dataset["test"]
print(len(dataset))


# data = dataset["train"][0]
#
# print(data)
unique_col1 = set(dataset.unique("source_language"))
unique_col2 = set(dataset.unique("translated_language"))

all_unique_values = unique_col1 | unique_col2
normalized_values = {value.strip().lower() for value in all_unique_values}
# 转换为字典，key 从 0 开始
value_dict = {i: v for i, v in enumerate(sorted(normalized_values))}

print(value_dict)
