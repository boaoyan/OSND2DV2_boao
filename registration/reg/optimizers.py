import pandas as pd
import torch
from tqdm import tqdm

from registration.registration import Registration
from projector.post_processing import apply_circular_mask, normalize_to_255

ITERS = 1000

def optimize(
    reg: Registration,
    ground_truth,
    criterion,
    lr_rotations=5e-2,
    lr_translations=1e2,
    momentum=0,
    dampening=0,
    n_itrs=ITERS,
    optimizer="sgd",  # 'sgd' or `adam`
    is_maximize=True
):
    # Initialize an optimizer with different learning rates
    # for rotations and translations since they have different scales
    if optimizer == "sgd":
        optim = torch.optim.SGD(
            [
                {"params": [reg._rotation], "lr": lr_rotations},
                {"params": [reg._translation], "lr": lr_translations},
            ],
            momentum=momentum,
            dampening=dampening,
            maximize=is_maximize,
        )
        optimizer = optimizer.upper()
    elif optimizer == "adam":
        optim = torch.optim.Adam(
            [
                {"params": [reg._rotation], "lr": lr_rotations},
                {"params": [reg._translation], "lr": lr_translations},
            ],
            maximize=is_maximize,
        )
        optimizer = optimizer.title()
    else:
        raise ValueError(f"Unrecognized optimizer {optimizer}")

    params = []
    losses = [criterion(ground_truth, reg()).item()]
    is_converged = False  # 新增标志位
    for itr in (pbar := tqdm(range(n_itrs), ncols=100)):
        # Save the current set of parameters
        alpha, beta, gamma = reg.rotation.squeeze().tolist()
        bx, by, bz = reg.translation.squeeze().tolist()
        params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

        # Run the optimization loop
        optim.zero_grad()
        estimate = reg()
        estimate = torch.max(estimate) - estimate
        estimate = apply_circular_mask(estimate)
        loss = criterion(ground_truth, estimate)
        loss.backward()
        optim.step()
        losses.append(loss.item())
        pbar.set_description(f"NCC = {loss.item():06f}")

        # Stop the optimization if the estimated and ground truth images are 99.9% correlated
        if is_maximize:
            if loss > 0.9999:
                if momentum != 0:
                    optimizer += " + momentum"
                if dampening != 0:
                    optimizer += " + dampening"
                tqdm.write(f"{optimizer} converged in {itr + 1} iterations")
                is_converged = True
                break
        else:
            if loss < 0.0001:
                if momentum != 0:
                    optimizer += " + momentum"
                if dampening != 0:
                    optimizer += " + dampening"
                tqdm.write(f"{optimizer} converged in {itr + 1} iterations")
                is_converged = True
                break
    # Save the final estimated pose
    alpha, beta, gamma = reg.rotation.squeeze().tolist()
    bx, by, bz = reg.translation.squeeze().tolist()
    params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

    df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    df["loss"] = losses
    df["converged"] = False
    df.loc[df.index[-1], "converged"] = is_converged  # 最后一行标记是否收敛
    return df, is_converged

#| code-fold: true
def optimize_lbfgs(
    reg: Registration,
    ground_truth,
    criterion,
    lr,
    line_search_fn=None,
    n_itrs=ITERS,
):
    # Initialize the optimizer and define the closure function
    optim = torch.optim.LBFGS(reg.parameters(), lr, line_search_fn=line_search_fn)

    def closure():
        if torch.is_grad_enabled():
            optim.zero_grad()
        estimate = reg()
        loss = -criterion(ground_truth, estimate)
        if loss.requires_grad:
            loss.backward()
        return loss

    params = []
    losses = [closure().abs().item()]
    for itr in (pbar := tqdm(range(n_itrs), ncols=100)):
        # Save the current set of parameters
        alpha, beta, gamma = reg.rotation.squeeze().tolist()
        bx, by, bz = reg.translation.squeeze().tolist()
        params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

        # Run the optimization loop
        optim.step(closure)
        with torch.no_grad():
            loss = closure().abs().item()
            losses.append(loss)
            pbar.set_description(f"NCC = {loss:06f}")

        # Stop the optimization if the estimated and ground truth images are 99.9% correlated
        if loss > 0.999:
            if line_search_fn is not None:
                method = f"L-BFGS + strong Wolfe conditions"
            else:
                method = "L-BFGS"
            tqdm.write(f"{method} converged in {itr + 1} iterations")
            break

    # Save the final estimated pose
    alpha, beta, gamma = reg.rotation.squeeze().tolist()
    bx, by, bz = reg.translation.squeeze().tolist()
    params.append([i for i in [alpha, beta, gamma, bx, by, bz]])

    df = pd.DataFrame(params, columns=["alpha", "beta", "gamma", "bx", "by", "bz"])
    df["loss"] = losses
    return df