import torch
import torch.nn as nn


class RPTHNet(nn.Module):
    def __init__(self, control):
        super().__init__()
        self.control = control

    def forward(self, Q, r):
        x = RPTHLayer.apply(Q, r, self.control)
        # --- normalization:
        normalize = self.control.get('normalize', True)
        if normalize:
            x_sum = torch.abs(x).sum(axis=1, keepdim=True)
            x = x / (x_sum + 1e-8)
        return x


class RPTHLayer(torch.autograd.Function):
    """
    Autograd function for forward solving and backward differentiating constraint QP
    """

    @staticmethod
    def forward(ctx, Q, r, control):
        """
        Newton algorithm for forward solving RP
        """

        # --- forward solve
        x = rpth_solve(Q=Q, r=r, control=control)

        # --- save for backwards:
        ctx.save_for_backward(Q, r, x)

        return x

    @staticmethod
    def backward(ctx, dl_dx):
        """
        KKT backward differentiation
        """
        Q, r, x = ctx.saved_tensors
        grads = rpth_grad(dl_dx=dl_dx, Q=Q, r=r, x=x)
        return grads


class RPTH:
    def __init__(self, Q, r, control):
        # --- input space:
        self.Q = Q
        self.r = r
        self.control = control

        # --- solution storage:
        self.sol = {}

    def solve(self):
        # --- solve QP:
        x = rpth_solve(Q=self.Q, r=self.r, control=self.control)
        normalize = self.control.get('normalize', True)
        if normalize:
            x_sum = torch.abs(x).sum(axis=1, keepdim=True)
            x = x / (x_sum + 1e-8)
        return x

    def update(self, Q=None, r=None, control=None):
        if Q is not None:
            self.Q = Q
        if r is not None:
            self.r = r
        if control is not None:
            self.control = control
        return None


def rpth_solve(Q, r, control):
    #######################################################################
    # Solve a batch convex risk parity programs:
    #   x_star =   argmin_x 0.5*x^TQx -sum(r*ln(-Gx))
    #                        subject to Gx <= 0
    # Q:  A (n_batch,n_x,n_x) SPD tensor
    # r:  A (n_batch, n_x) matrix
    # Returns: x_star:  A (n_batch,n_x,1) tensor
    #######################################################################

    # --- unpacking control:
    max_iters = control.get('max_iters', 10)
    eps = control.get('eps', 1e-3)
    orthant = control.get('orthant')
    verbose = control.get('verbose', False)

    # --- prep:
    n_batch = Q.shape[0]
    n_x = Q.shape[1]
    dtype = Q.dtype
    # --- initialize risk and orthant:
    r = torch.abs(r)
    if orthant is None:
        orthant = 1.0

    # --- starting point at inverse vol portfolio
    Q_diag = torch.diagonal(Q, dim1=1, dim2=2)
    x = r / torch.sqrt(Q_diag).unsqueeze(2)
    x = orthant * x

    # --- main loop:
    fl = torch.ones(size=(n_batch, 1, 1), dtype=dtype) * 0.95 * ((3 - 5 ** 0.5) / 2)
    lk = fl + 1
    # --- main loop:
    for i in range(max_iters):
        # --- gradient and hessian
        r_x = r / x
        u = Q.matmul(x) - r_x
        #h = Q + torch.diag_embed((r_x / x).squeeze(2), dim1=1, dim2=2)
        h = Q * 1.0
        h[:, range(n_x), range(n_x)] = Q_diag + (r_x / x).squeeze(2)

        # --- newton step:
        dk = torch.linalg.solve(h, u)
        gk = torch.linalg.norm(dk / x, ord=torch.inf, dim=1, keepdim=True)
        gk[lk <= fl] = 0.0

        # --- update:
        lk = (dk*u).sum(axis=1, keepdim=True)
        lk = torch.sqrt(lk)
        x = x - (1 / (1 + gk)) * dk

        # --- verbose
        error = lk.max().item()
        if verbose:
            print('iteration = {:d}'.format(i))
            print('||error||_2 = {:f}'.format(error))

        # ---- break
        if error < eps:
            break

    # --- output:
    return x


# --- if scale is true then we can't re-use ATA, LU and P
def rpth_grad(dl_dx, Q, r, x):
    # --- prep:
    n_x = Q.shape[1]
    # --- initialize risk and orthant:
    if len(r.shape) < 2:
        r = r.unsqueeze(2)
    r = torch.abs(r)
    r_x = r / x
    h = Q * 1.0
    Q_diag = torch.diagonal(Q, dim1=1, dim2=2)
    h[:, range(n_x), range(n_x)] = Q_diag + (r_x / x).squeeze(2)

    # --- newton step:
    dx = torch.linalg.solve(h, -dl_dx)

    # --- gradients:
    xt = torch.transpose(x, 1, 2)
    dl_dQ1 = torch.matmul(0.50 * dx, xt)
    dl_dQ = dl_dQ1 + torch.transpose(dl_dQ1, 1, 2)
    dl_dr = -(1 / x) * dx

    # --- return grads:
    grads = (dl_dQ, dl_dr, None)

    return grads
