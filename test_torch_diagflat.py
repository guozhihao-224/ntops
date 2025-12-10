import torch

# size = 3, offset = 10
input_tensor = torch.tensor([1.0, 2.0, 3.0])
offset = 10

result = torch.diagflat(input_tensor, offset=offset)

print(f"Input: {input_tensor}")
print(f"Input size: {input_tensor.numel()}")
print(f"Offset: {offset}")
print(f"Output shape: {result.shape}")
print(f"Output:\n{result}")

# 也测试一下 offset = -10
result_neg = torch.diagflat(input_tensor, offset=-10)
print(f"\nOffset: -10")
print(f"Output shape: {result_neg.shape}")
print(f"Output:\n{result_neg}")
