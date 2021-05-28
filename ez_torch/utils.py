import torch


def get_uv_grid(W, H):
    # W, H = 128, 128
    x = torch.arange(-1, 1, 1 / W * 2)
    y = torch.arange(-1, 1, 1 / H * 2)
    xx, yy = torch.meshgrid(x, y)

    uv_grid = torch.stack([yy, xx], dim=0)
    return uv_grid


def pad_in_dim(tensor, pad_size, dim, val=0):
    shape = list(tensor.shape)
    shape[dim] = pad_size - shape[dim]
    padding = torch.full(shape, val).to(tensor.device)
    out = torch.cat([tensor, padding], dim=dim)

    return out


# Trim and/or slice tensor
def fix_dim_size(tensor, size, dim, pad_value=0):
    # Slice only dim
    indices = {dim: slice(0, size)}
    idx = [indices.get(dim, slice(None)) for dim in range(tensor.ndim)]

    tensor = tensor[idx]
    tensor = pad_in_dim(tensor, size, dim, pad_value)

    return tensor


def soft_addressing(keys, address_space, bank):
    """Lets you address tensors from the bank addressed by address_space, using
    the keys.
    Types:
                 keys: Tensor[*A, address_size]
        address_space: Tensor[num_addresses, address_size]
                 bank: Tensor[num_addresses, *B]
               return: Tensor[*A, *B]
    """
    A = keys.size()[:-1]
    address_size = keys.size(-1)

    flat_keys = keys.reshape(-1, address_size)
    selectors = torch.matmul(flat_keys, address_space.T)
    selectors = torch.softmax(selectors, dim=-1)
    selectors = selectors.view(*A, -1)

    return generic_matmul(selectors, bank)


def generic_matmul(first, second):
    """Matmul in the last dim.
    Types:
         first: Tensor[*A,  N]
        second: Tensor[ N, *B]
        return: Tensor[*A, *B]
    """
    A = first.size()[:-1]
    B = second.size()[1:]
    N = first.size(-1)

    flat_first = first.reshape(-1, N)
    flat_second = second.reshape(N, -1)

    return torch.matmul(flat_first, flat_second).view(*A, *B)


def unsqueeze_expand(tensor, dim, times):
    if times == 0:
        return tensor

    tensor = tensor.unsqueeze(dim)
    new_shape = list(tensor.shape)
    new_shape[dim] = times
    return tensor.expand(new_shape)


def reshape_in_time(tensor):
    return tensor.reshape(-1, *tensor.shape[2:])


def count_parameters(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def batch_conv(x, w, b, p=0, s=1):
    # SRC - https://discuss.pytorch.org/t/apply-different-convolutions-to-a-batch-of-tensors/56901/2

    batch_size = x.size(0)
    output_size = w.size(1)

    o = F.conv2d(
        x.reshape(1, batch_size * x.size(1), x.size(2), x.size(3)),
        w.reshape(batch_size * w.size(1), w.size(2), w.size(3), w.size(4)),
        b.reshape(batch_size * b.size(1)),
        groups=batch_size,
        padding=p,
        stride=s,
        dilation=1,
    )
    o = o.reshape(batch_size, output_size, o.size(2), o.size(3))

    return o


def compute_output_shape(net, frame_shape):
    with torch.no_grad():
        t = torch.rand(1, *frame_shape)
        out = net(t)

    return out.shape


def sample_dim(tensor, n, dim, dim_size=None):
    if dim_size is None:
        dim_size = tensor.size(dim)

    index = torch.randperm(dim_size).to(tensor.device)
    index = index[:n]
    return tensor.index_select(dim=dim, index=index)


def sample_padded_sequences(sequences, lens, sample_size):
    seq_dim = 1

    new_shape = list(sequences.shape)
    new_shape[seq_dim] = sample_size

    data = torch.zeros(new_shape).to(sequences.device)
    new_lens = torch.zeros_like(lens).to(lens.device)

    # TODO: Make this vector operation if possible!
    for i, (length, seq) in enumerate(zip(lens, sequences)):
        new_lens[i] = min(length, sample_size)
        sample = sample_dim(
            seq,
            n=new_lens[i],
            dim=0,
            dim_size=length,
        )
        data[i, : sample.size(0)] = sample

    return data, new_lens


def mask_seq_from_lens(tensor, lens):
    # SRC - <https://stackoverflow.com/a/53403392>
    seq_dim = 1
    mask = torch.arange(tensor.size(seq_dim))[None, :] < lens[:, None]
    mask = mask.reshape(*mask.shape, *([1] * (len(tensor.shape) - 2)))
    mask = mask.expand(*tensor.shape)
    mask = mask.to(tensor.device)

    return tensor * mask


@torch.jit.script
def mask_sequence(tensor, mask):
    initial_shape = tensor.shape
    bs, seq = mask.shape
    masked = torch.where(
        mask.reshape(bs * seq, -1),
        tensor.reshape(bs * seq, -1),
        torch.tensor(0, dtype=torch.float32).to(tensor.device),
    )

    return masked.reshape(initial_shape)


def prepare_rnn_state(state, num_rnn_layers):
    """
    RNN cells expect the initial state
    in the shape -> [rnn_num_layers, bs, rnn_state_size]
    In this case rnn_state_size = state // rnn_num_layers.
    The state is distributed among the layers
    state          -> [bs, state_size]
    rnn_num_layers -> int
    """
    return torch.stack(
        state.chunk(torch.tensor(num_rnn_layers), dim=1),
        dim=0,
    )


def time_distribute(module, input=None):
    """
    Distribute execution of module over batched sequential input tensor.
    This is done in the batch dimension to facilitate parallel execution.
    input  -> [bs, seq, *x*]
    module -> something that takes *x*
    return -> [bs, seq, module(x)]
    """
    shape = input[0].size() if type(input) is list else input.size()
    bs = shape[0]
    seq_len = shape[1]

    if type(input) is list:
        input = [i.reshape(-1, *i.shape[2:]) for i in input]
    else:
        input = input.reshape(-1, *shape[2:])

    out = module(input)  # should return iterable or tensor

    if hasattr(out, "view"):
        # assume it is tensor:
        return out.view(bs, seq_len, *out.shape[1:])

    return map(lambda o: o.view(bs, seq_len, *o.shape[1:]), out)
