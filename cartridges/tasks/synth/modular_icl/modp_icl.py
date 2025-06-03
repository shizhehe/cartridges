import torch


def mod_p_data(p, task="multiplication", add_fill_tokens=True):
    """x◦y = x/y (mod p) for 0 ≤ x < p, 0 < y < p"""

    # tokens for <op> and <=>. It's not clear why <=> is needed at all since it
    # has no effect on the output, but we'll leave it in to best follow the
    # paper.
    eq_token = p
    op_token = p + 1

    x = torch.arange(p)
    y = torch.arange(p)
    x, y = torch.cartesian_prod(x, y).T

    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token

    if task == "multiplication":
        result = (x * y) % p
    elif task == "addition":
        result = (x + y) % p
    elif task == "subtraction":
        result = (x - y) % p
    elif task == "division":
        x = torch.arange(p)
        y = torch.arange(1, p)
        x, y = torch.cartesian_prod(x, y).T
        eq = torch.ones_like(x) * eq_token
        op = torch.ones_like(x) * op_token
        result = x
        x = y * result % p
    elif task == "parity_division":  # TODO JL fix
        x = torch.arange(p)
        y_odd = torch.arange(1, p, step=2)
        y_even = torch.arange(0, p, step=2)
        x_1, y_odd = torch.cartesian_prod(x, y_odd).T
        x_2, y_even = torch.cartesian_prod(x, y_even).T
        result_1 = x_1
        x_1 = y_odd * result_1 % p
        result_2 = (x_2 - y_even) % p
        x = torch.cat((x_1, x_2))
        y = torch.cat((y_odd, y_even))
        result = torch.cat((result_1, result_2))
    elif task == "sum_of_squares":
        result = (x**2 + y**2) % p
    elif task == "quad1":
        result = (x**2 + x * y + y**2) % p
    elif task == "quad2":
        result = (x**2 + x * y + y**2 + x) % p
    elif task == "cubic1":
        result = (x**3 + x * y) % p
    elif task == "cubic2":
        result = (x**3 + x * (y**2) + y) % p
    elif task == "s5":
        operands = list(range(5))
        elems = map(Permutation, itertools.permutations(operands))
        tuples = itertools.product(elems, repeat=2)
        for a, b in tuples:
            c = a * b
        if add_fill_tokens:
            return torch.stack([a, op, b, eq, c]).transpose(0, 1)
        else:
            return torch.stack([a, b, c]).transpose(0, 1)
    elif task == "s5conj":
        operands = list(range(5))
        elems = map(Permutation, itertools.permutations(operands))
        tuples = itertools.product(elems, repeat=2)
        for a, b in tuples:
            c = a * b * (a.__invert__())
        if add_fill_tokens:
            return torch.stack([a, op, b, eq, c]).transpose(0, 1)
        else:
            return torch.stack([a, b, c]).transpose(0, 1)
    elif task == "s5aba":
        operands = list(range(5))
        elems = map(Permutation, itertools.permutations(operands))
        tuples = itertools.product(elems, repeat=2)
        for a, b in tuples:
            c = a * b * a
        if add_fill_tokens:
            return torch.stack([a, op, b, eq, c]).transpose(0, 1)
        else:
            return torch.stack([a, b, c]).transpose(0, 1)

    # "All of our experiments used a small transformer trained on datasets of
    # equations of the form a◦b = c, where each of “a”, “◦”, “b”, “=”, and “c”
    # is a seperate token"
    if add_fill_tokens:
        return torch.stack([x, op, y, eq, result]).transpose(0, 1)  # (N, L)
    else:
        return torch.stack([x, y, result]).transpose(0, 1)  # (N, L)


class ModPContextConfig(BaseContextConfig):
    p: int = 5
    task: str = "multiplication"
    add_fill_tokens: bool = True

    def instantiate(self) -> Context:
        data = mod_p_data(self.p, self.task, self.add_fill_tokens)
        return Context(
            
        )