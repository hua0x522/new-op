import torch 
import nop
import nop.backend

tensors = torch.load('tensors.pt')

input = tensors['input']
weight = tensors['weight']
out_in_map = tensors['out_in_map']

print(input.shape)
print(weight.shape)
out = nop.backend.conv_forward_implicit_gemm_cuda(input, weight, out_in_map, input.shape[0], weight.shape[2], False, True)
print(out[0])