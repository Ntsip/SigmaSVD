import torch
import torch.nn.functional as F
import numpy as np
import torchvision
import torch.optim as optim
import time
from torch.func import jacrev
from functorch import make_functional_with_buffers
import torchvision.transforms as transforms


torch.manual_seed(seed);

class _SolversJacrev(object):
    def __init__(self, model, loss, method, batch_size_train, batch_size_test, lr, b1, b2, log_interval,
                 n_epochs, coarse_dim, kappa, eig_threshold, train_loader, test_loader, chunk_size):
        self.model = model # AutoEncoder model of the paper
        self.loss = loss #mse, nll, cross_entropy, see self.get_loss function below
        self.method = method # pick between "mult_newton" and "adam"
        self.batch_size_train = batch_size_train
        self.batch_size_test = batch_size_test
        self.lr = lr # learning rate
        self.b1 = b1 # mult_newton and adam's first momentum
        self.b2 = b2 #adam's second momentum
        self.log_interval = log_interval # set default 1
        self.n_epochs = n_epochs # 20 epochs for paper's experiments
        self.coarse_dim = coarse_dim #select the dimensions for the coarse model
        self.kappa = kappa # desired rank for the (T)-SVD, None for full SVD
        self.eig_threshold = eig_threshold # nu in the paper, default 1e-8
        self.train_loader = train_loader #take from torchvision.datasets
        self.test_loader = test_loader # from torchvision.datasets
        self.chunk_size = chunk_size # set smaller if memorys issues

    def flatten_weights(self, params, mask):
        shapes = []
        res = []
        for p in params:
            shapes.append(p.shape)
            res.append(p.reshape(-1))
        flat = torch.cat(res)
        sub = flat[mask]
        return sub, flat, shapes

    def unflatten_weights(self, diff_params, params, mask, shapes):
        if diff_params is not None:
            params[mask] = diff_params
        i = 0
        res = []
        for shape in shapes:
            p = i + np.prod(shape)
            res.append(params[i:p].reshape(shape))
            i = p
        return res

    def compute_loss_stateless_model(self, diff_flat_params, flat_params, mask, shapes, buffers, inputs, target,
                                     reduction):
        params = self.unflatten_weights(diff_flat_params, flat_params, mask, shapes)
        batch = inputs
        targets = target

        predictions = self.fmodel(params, buffers, batch)
        loss = self.loss_function(predictions, inputs)
        return loss

    def compute_direction(self, idx, H, g):
        g = g.double()
        H = H.double()
        grad = g[idx]
        if self.kappa == None:
          Sigma_kappa, U_kappa = torch.linalg.eigh(H)
          Sigma_kappa = torch.abs(Sigma_kappa)
          Sigma_kappa = torch.clamp(Sigma_kappa, min=self.eig_threshold)
          dH = - U_kappa @ ((1.0 / Sigma_kappa) * (U_kappa.T @ grad))
          d = torch.zeros(self.n_weights, device=self.device).double()
          d[idx] = dH
          return d.to(torch.float32)
        elif self.kappa < H.size(0):
          U_kappa, Sigma_kappa, v = torch.svd_lowrank(H, q=self.kappa)
          if Sigma_kappa.min() < self.eig_threshold:
            Sigma_kappa[Sigma_kappa < self.eig_threshold] = self.eig_threshold
          U_kappa_minus1 = U_kappa[:, :self.kappa - 1]
          Sigma_minus1 = Sigma_kappa[:self.kappa - 1]
          U_grad = U_kappa_minus1.T @ grad
          parenthesis = torch.diag(Sigma_minus1**(-1) - Sigma_kappa[-1]**(-1))
          LU_grad = parenthesis @ U_grad
          ULU_grad = U_kappa_minus1 @ LU_grad
          dH = - grad / Sigma_kappa[-1] - ULU_grad
          d = torch.zeros(self.n_weights, device=self.device).double()
          d[idx] = dH
          return d.to(torch.float32)

    def train_losses_fun(self):
        losses_sum = 0
        correct = 0
        losses = []
        model = self.model
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(self.train_loader):
                data = data.view(-1, 784).to(device)
                data = data.to(device)
                output = model(data)
                loss = self.loss_function(output, data).item()
                losses_sum = losses_sum + loss
            loss_av = losses_sum / (batch_idx + 1)
            losses.append(loss_av)
            print('\nTrain set: Avg. loss: {:.4f}\n'.format(loss_av))
        return losses

    def test(self, params):
        losses_sum = 0
        correct = 0
        losses, accuracy = [], []
        model = self.model
        model.eval()
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(self.test_loader):
                data = data.view(-1, 784).to(device)
                output = model(data)
                output = self.fmodel(params, self.buffers, data)
                loss = self.loss_function(output, data).item()
                losses_sum = losses_sum + loss
            loss_av = losses_sum / (batch_idx + 1)
            losses.append(loss_av)
            print('\nTest set: Avg. loss: {:.7f}\n'.format( loss_av))
        return losses

    def get_loss(self):
        if self.loss == 'nll':
            return F.nll_loss
        elif self.loss == 'cross_entropy':
            return F.cross_entropy
        elif self.loss == 'mse':
            return torch.nn.MSELoss()

    def solve(self):
        model = self.model
        self.device = device
        self.loss_function = self.get_loss()
        fmodel, params, buffers = make_functional_with_buffers(model)
        self.fmodel, self.buffers = fmodel, buffers
        train_losses = self.train_losses_fun()
        test_losses = self.test(params)
        self.n_weights = sum(p.numel() for p in model.parameters())
        coarse_dim = round(self.coarse_dim * self.n_weights)
        ft_compute_hess = jacrev(jacrev(self.compute_loss_stateless_model), chunk_size=self.chunk_size)  # HESSIAN
        ft_compute_grad = jacrev(self.compute_loss_stateless_model)  # JACOBIAN
        reduced_weights, all_weights, shapes = self.flatten_weights(params, torch.arange(self.n_weights))
        m1, u1, iter = 0, 0, 1
        times = [0]
        self.iter_hist = torch.zeros(3)
        self.iter_hist = torch.tensor((train_losses[0], test_losses[0], times[0]))
        t1_start = time.perf_counter()
        for epoch in range(1, self.n_epochs + 1):
            train_losses_sum = 0
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(self.train_loader):
                    data = data.view(-1, 784).to(device)
                    data = data.to(device)
                    reduced_weights, all_weights, shapes = self.flatten_weights(params, torch.arange(self.n_weights))
                    loss = self.compute_loss_stateless_model(None, all_weights, None, shapes, buffers, data, data,
                                                             'mean')
                    train_losses_sum = train_losses_sum + loss
                    grads = ft_compute_grad(reduced_weights, all_weights, torch.arange(self.n_weights), shapes,
                                            buffers, data, data, 'mean')
                    if self.method == 'mult_newton':
                        mask = torch.randperm(self.n_weights)[:coarse_dim]
                        reduced_weights, all_weights, shapes = self.flatten_weights(params, mask)
                        H = ft_compute_hess(reduced_weights, all_weights, mask, shapes, buffers, data, data, 'mean')
                        m2 = self.b1 * m1 + (1 - self.b1) * grads
                        m2_hat = m2 / (1 - self.b1 ** iter)
                        d = self.compute_direction(mask, H, m2_hat)
                        all_weights = all_weights + self.lr * d
                        m1 = m2
                        iter += 1
                    elif self.method == 'adam':
                        m2 = self.b1 * m1 + (1 - self.b1) * grads
                        m2_hat = m2 / (1 - self.b1 ** iter)
                        u2 = self.b2 * u1 + (1 - self.b2) * (grads ** 2)
                        u2_hat = u2 / (1 - self.b2 ** iter)
                        d = - m2_hat / (torch.sqrt(u2_hat) + self.eig_threshold)
                        all_weights = all_weights + self.lr * d
                        m1, u1 = m2, u2
                        iter += 1

                    params = self.unflatten_weights(None, all_weights, None, shapes)

                    if batch_idx % self.log_interval == 0:
                        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(self.train_loader.dataset),
                                  100. * batch_idx / len(self.train_loader), loss.item()))

                train_losses_av = train_losses_sum / (batch_idx + 1)
                train_losses.append(train_losses_av.item())
                print('\nTrain set: Avg. loss: {:.7f}\n'.format(train_losses_av))

                l= self.test(params)
                test_losses.append(l[0])
                t1_stop = time.perf_counter()
                curr_time = t1_stop - t1_start
                print(curr_time)
                times.append(curr_time)
                self.iter_hist = torch.vstack((self.iter_hist, torch.tensor((train_losses[epoch], test_losses[epoch],
                                                                            times[epoch]))))
        self.train_losses = train_losses
        self.test_losses = test_losses
        self.times = times
        return self
