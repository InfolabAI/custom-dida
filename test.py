import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


def get_gpu_memory_usage(device_id):
    import subprocess

    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,nounits,noheader"],
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip()
    memory_usage = [int(x) for x in output.split("\n")]
    return memory_usage[int(device_id)]


writer = SummaryWriter("./logs/tmp/test/test/")

x1 = torch.Tensor(1000, 1000).to("cuda")
aa = nn.Linear(1000, 100).to("cuda")
bb = nn.Linear(100, 10).to("cuda")
optima = torch.optim.Adam(aa.parameters())
optimb = torch.optim.Adam(bb.parameters())

step = 0
for j in range(100):
    aout = aa(x1)
    for i in range(10):
        out = bb(nn.functional.relu(aout))
        loss = nn.functional.mse_loss(out.mean(), torch.tensor(5.0).to("cuda"))
        loss.backward(retain_graph=True)
        print(
            f"epoch {j} iter {i} |loss {float(loss.cpu().detach()):.2f}|gpu: {get_gpu_memory_usage(0)} MiB|aa grad {float((aa.weight.grad.mean() + aa.bias.grad.mean()).cpu().detach()):.2f}|bb grad {float((bb.weight.grad.mean() + bb.bias.grad.mean()).cpu().detach()):.2f}",
        )
        writer.add_scalar("loss", loss, step)
        writer.add_scalar(
            "aa_grad",
            (aa.weight.grad.mean().cpu().detach() + aa.bias.grad.mean().cpu().detach()),
            step,
        )
        writer.add_scalar(
            "bb_grad",
            (bb.weight.grad.mean().cpu().detach() + bb.bias.grad.mean().cpu().detach()),
            step,
        )
        writer.add_scalar("gpu", get_gpu_memory_usage(0), step)
        optimb.step()
        optimb.zero_grad()
        step += 1

    optima.step()
    optima.zero_grad()
