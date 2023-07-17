import torch


def forward_hook(module, input_, output_):
    print(module)  # 해당 모듈 출력
    print("Input:")
    for tensor in input_:
        if type(tensor) is torch.Tensor:
            tensor = tensor.to(float)
            print(f"{tensor.shape}, mean: {tensor.mean()}, std: {tensor.std()}")
            if torch.isnan(tensor).any():
                breakpoint()
                print("   nan or inf")
            break
    print("Output:")
    for tensor in output_:
        if type(tensor) is torch.Tensor:
            tensor = tensor.to(float)
            print(f"{tensor.shape}, mean: {tensor.mean()}, std: {tensor.std()}")
            if torch.isnan(tensor).any():
                breakpoint()
                print("   nan or inf")
            break


# 역전파 과정에서 그래디언트 확인을 위한 hook 함수
def gradient_hook(module, grad_input, grad_output):
    print(module)  # 해당 모듈 출력
    print("Gradient Input:")
    for grad in grad_input:
        if grad is not None:
            print(f"{grad.shape}, mean: {grad.mean()}, std: {grad.std()}")
            if torch.isnan(grad).any():
                breakpoint()
                print("   nan or inf")
    print("Gradient Output:")
    for grad in grad_output:
        if grad is not None:
            print(f"{grad.shape}, mean: {grad.mean()}, std: {grad.std()}")
            if torch.isnan(grad).any():
                breakpoint()
                print("   nan or inf")
    print()


def gradient_hook_for_tensor(grad):
    if grad is not None:
        print(f"x hook: {grad.shape}, mean: {grad.mean()}, std: {grad.std()}")
        if torch.isnan(grad).any():
            breakpoint()
            print("   nan or inf")
