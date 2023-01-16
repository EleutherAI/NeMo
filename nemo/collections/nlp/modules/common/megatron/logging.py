import sys
import torch

def human_readable_flops(num):
    for unit in [
        "",
        "KFLOPS",
        "MFLOPS",
        "GFLOPS",
        "TFLOPS",
        "PFLOPS",
        "EFLOPS",
        "ZFLOPS",
    ]:
        if abs(num) < 1000.0:
            return "%3.1f%s" % (num, unit)
        num /= 1000.0
    return "%.1f%s" % (num, "Yi")


def get_flops(seq_length, hidden_size, num_layers, total_params, train_batch_size, iter_time_s):
    world_size = torch.distributed.get_world_size()
    ff = total_params * 6
    attn = seq_length * hidden_size * num_layers * 60
    flops = (
        train_batch_size
        * seq_length
        * (ff + attn)
        / (iter_time_s * world_size)
    )
    #tflops = flops / 1.0e12
    return flops
