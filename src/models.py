import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BayesianCNN(nn.Module):
    """
    Bayesian CNN with MC Dropout for uncertainty estimation. Same as paper.
    """

    def __init__(self, dropout_p1: float = 0.25, dropout_p2: float = 0.5):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=1, padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=4, stride=1, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(p=0.25)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 11 * 11, 128)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x, apply_dropout: bool = True):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        if apply_dropout:
            x = self.dropout1(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        if apply_dropout:
            x = self.dropout2(x)

        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def predict_proba(self, x, n_samples: int = 100, return_samples: bool = True):
        """
        Get predictive distribution using MC Dropout. Can return sample probs for variance estimation.
        """
        self.train()  # Enables dropout
        batch_size = x.size(0)
        device = x.device

        with torch.inference_mode():
            if return_samples:
                mc_batch_size = n_samples

                # Expand in order to allow for parallel MC sampling
                # [B, 1, 28, 28] -> [T * B, 1, 28, 28]
                x_rep = x.unsqueeze(0).expand(mc_batch_size, -1, -1, -1, -1)
                x_rep = x_rep.reshape(mc_batch_size * batch_size, *x.shape[1:])

                log_probs = self.forward(x_rep, apply_dropout=True)
                probs = torch.exp(log_probs)  # [T * B, 10]
                samples = probs.view(mc_batch_size, batch_size, -1)  # [T, B, 10]
                mean_probs = samples.mean(dim=0)  # [B, 10]
                return mean_probs, samples
            else:
                prob_sum = torch.zeros(batch_size, 10, device=device)
                for _ in range(n_samples):
                    log_probs = self.forward(x, apply_dropout=True)
                    prob_sum += torch.exp(log_probs)
                mean_probs = prob_sum / n_samples
                return mean_probs, None

    def extract_features(self, x, apply_dropout: bool = False):
        """
        Extract features from the penultimate layer.
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        if apply_dropout:
            x = self.dropout1(x)

        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))

        if apply_dropout:
            x = self.dropout2(x)

        return x


class IndependentBLR:
    """
    Analytic Bayesian Linear Regression (Minimal Extension),
    assumes independent output dimensions.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 10,
        prior_var: float = 1.0,
        noise_var: float = 1.0,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_var = prior_var
        self.noise_var = noise_var

        # Parameters (Computed during fit)
        self.mu = None
        self.S = None
        self.fitted = False

    def fit(self, Phi: torch.Tensor, Y: torch.Tensor):
        device = Phi.device
        N, D = Phi.shape

        # Adds bias term to linear regression
        bias = torch.ones(N, 1, device=device, dtype=Phi.dtype)
        Phi_aug = torch.cat([Phi, bias], dim=1)
        D_aug = D + 1

        # Compute Precision: A = (1/sigma^2)Phi^T Phi + (1/s^2)I
        inv_noise = 1.0 / self.noise_var
        inv_prior = 1.0 / self.prior_var

        PhiT_Phi = Phi_aug.T @ Phi_aug
        Precision = inv_noise * PhiT_Phi

        diag_idx = torch.arange(D_aug, device=device)
        Precision[diag_idx, diag_idx] += inv_prior

        # Weak regularization for bias, no reason for bias to be small
        Precision[-1, -1] -= inv_prior
        Precision[-1, -1] += 1e-6

        # Had to move to cpu for numerically stable inversions
        Precision_cpu = Precision.cpu()

        try:
            L_cpu = torch.linalg.cholesky(Precision_cpu)
            S_cpu = torch.cholesky_inverse(L_cpu)
        except RuntimeError:
            # Non-positive definite matrix
            S_cpu = torch.inverse(Precision_cpu)

        self.S = S_cpu.to(device)  # Posterior Covariance

        # Compute Posterior Mean
        PhiT_Y = Phi_aug.T @ Y
        self.mu = inv_noise * (self.S @ PhiT_Y)

        self.fitted = True

    def predict(self, Phi: torch.Tensor, return_std: bool = True):
        """
        Computes the Predictive Mean and Standard Deviation.
        """
        device = Phi.device
        N = Phi.shape[0]

        # Add bias
        bias = torch.ones(N, 1, device=device, dtype=Phi.dtype)
        Phi_aug = torch.cat([Phi, bias], dim=1)

        # Predictive Mean
        pred_mean = Phi_aug @ self.mu

        if return_std:
            # Predictive Variance
            # Aleatoric (sigma^2) + Epistemic (phi^T S phi)

            # Efficiently computes diagonal of (Phi S Phi^T)
            Phi_S = Phi_aug @ self.S
            epistemic_var = (Phi_S * Phi_aug).sum(dim=1, keepdim=True)  # [N, 1]

            # Total variance
            total_var = self.noise_var + epistemic_var

            # Expand to [N, K] for consistency with other methods
            pred_std = torch.sqrt(total_var).repeat(1, self.output_dim)

            return pred_mean, pred_std

        return pred_mean

    def get_acquisition_score(self, Phi: torch.Tensor, method="determinant"):
        """
        Returns scalar uncertainty score for Active Learning.
        Since outputs are independent, the predictive covariance is diagonal:
        Sigma_pred = (sigma^2 + phi^T S phi) * I
        """
        device = Phi.device
        N = Phi.shape[0]

        bias = torch.ones(N, 1, device=device, dtype=Phi.dtype)
        Phi_aug = torch.cat([Phi, bias], dim=1)

        Phi_S = Phi_aug @ self.S
        epistemic_term = (Phi_S * Phi_aug).sum(dim=1)  # [N]

        # Total variance per dimension
        total_var_scalar = self.noise_var + epistemic_term

        return total_var_scalar


class MFVIRegression(nn.Module):
    """
    Mean-Field Variational Inference (MFVI) with Analytic ELBO.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 10,
        prior_var: float = 1.0,
        noise_var: float = 1.0,
    ):
        super().__init__()
        # Add 1 to include bias
        self.input_dim = input_dim + 1
        self.output_dim = output_dim
        self.prior_var = prior_var
        self.noise_var = noise_var

        # We initialize weights to 0
        self.mu = nn.Parameter(torch.zeros(self.input_dim, output_dim))

        # Log Variance
        # Initialize to low variance
        self.log_var = nn.Parameter(torch.full((self.input_dim, output_dim), -5.0))

    def _add_bias(self, Phi: torch.Tensor) -> torch.Tensor:
        # Avoid adding bias twice if user manually did it (simple check)
        if Phi.shape[1] == self.input_dim:
            return Phi

        device = Phi.device
        N = Phi.shape[0]
        bias_col = torch.ones(N, 1, device=device, dtype=Phi.dtype)
        return torch.cat([Phi, bias_col], dim=1)

    @property
    def var(self) -> torch.Tensor:
        return torch.exp(self.log_var)

    def kl_divergence(self) -> torch.Tensor:
        """
        Analytic KL(q(W) || p(W)).
        """
        var = self.var
        kl = 0.5 * (
            (var + self.mu**2) / self.prior_var
            - 1.0
            - self.log_var
            + math.log(self.prior_var)
        )
        return kl.sum()

    def expected_log_likelihood(
        self, Phi: torch.Tensor, Y: torch.Tensor
    ) -> torch.Tensor:
        Phi_aug = self._add_bias(Phi)
        N = Y.shape[0]

        pred_mean = Phi_aug @ self.mu

        mse_term = ((Y - pred_mean) ** 2).sum()

        # Variance Trace Term
        # (Phi_aug^2) @ Var
        phi_sq = Phi_aug**2
        var_term_matrix = phi_sq @ self.var
        var_term = var_term_matrix.sum()

        expected_sse = mse_term + var_term

        # Constant term (needed because contains variance)
        const_term = -0.5 * N * self.output_dim * math.log(2 * math.pi * self.noise_var)

        return const_term - (0.5 / self.noise_var) * expected_sse

    def elbo(self, Phi: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        return self.expected_log_likelihood(Phi, Y) - self.kl_divergence()

    def fit(
        self,
        Phi: torch.Tensor,
        Y: torch.Tensor,
        n_iterations: int = 2000,
        lr: float = 0.01,
        verbose: bool = False,
    ) -> float:
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_iterations
        )

        final_loss = 0.0
        for i in range(n_iterations):
            optimizer.zero_grad()

            # Loss is Negative ELBO
            loss = -self.elbo(Phi, Y)

            loss.backward()
            optimizer.step()
            scheduler.step()

            final_loss = loss.item()

            if verbose and i % 500 == 0:
                print(f"Iter {i}: ELBO = {-final_loss:.2f}")

        return -final_loss

    @torch.inference_mode()
    def predict_analytic(self, Phi: torch.Tensor, return_std: bool = True):
        Phi_aug = self._add_bias(Phi)

        pred_mean = Phi_aug @ self.mu

        if return_std:
            epistemic_var = (Phi_aug**2) @ self.var

            total_var = epistemic_var + self.noise_var
            pred_std = torch.sqrt(total_var.clamp(min=1e-10))
            return pred_mean, pred_std

        return pred_mean

    @torch.inference_mode()
    def epistemic_variance(self, Phi: torch.Tensor) -> torch.Tensor:
        Phi_aug = self._add_bias(Phi)
        epistemic_matrix = (Phi_aug**2) @ self.var

        # Sum across classes for acquisition score
        return epistemic_matrix.sum(dim=1)


class MatrixNormalRegression:
    """
    Analytic Regression with Matrix Normal Posterior
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        prior_var: float = 1.0,
        noise_cov: torch.Tensor = None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.prior_var = prior_var

        if noise_cov is None:
            self.noise_cov = torch.eye(output_dim)
        else:
            self.noise_cov = noise_cov

        self.mu = None
        self.U = None  # This is A^-1
        self.fitted = False

    def fit(self, Phi: torch.Tensor, Y: torch.Tensor):
        device = Phi.device
        N, D = Phi.shape

        # Add bias
        bias = torch.ones(N, 1, device=device)
        Phi_aug = torch.cat([Phi, bias], dim=1)
        D_aug = D + 1

        # Move to Double for stability
        Phi_cpu = Phi_aug.cpu().double()
        Y_cpu = Y.cpu().double()

        # Conjugate Prior Setup
        # We assume scalar prior precision lambda = 1/s^2 for inputs.
        lambda_reg = 1.0 / self.prior_var

        # I found numerical instability with the standard inverse analytic solution
        # So I had to reformulate ridge as a least squares problem
        sqrt_lam = math.sqrt(lambda_reg)
        reg_matrix = sqrt_lam * torch.eye(D_aug, dtype=torch.float64)
        reg_matrix[-1, -1] = 0.0  # Don't regularize bias

        A = torch.cat([Phi_cpu, reg_matrix], dim=0)
        B_zeros = torch.zeros(D_aug, self.output_dim, dtype=torch.float64)
        B = torch.cat([Y_cpu, B_zeros], dim=0)

        # Posterior Mean M
        solution = torch.linalg.lstsq(A, B).solution
        self.mu = solution.float().to(device)

        # Compute Input Covariance U = (Phi^T Phi + Lambda I)^-1
        PhiT_Phi = Phi_cpu.T @ Phi_cpu
        Reg = lambda_reg * torch.eye(D_aug, dtype=torch.float64)
        Reg[-1, -1] = 1e-6  # Ensures positive definite precision

        Precision = PhiT_Phi + Reg
        # Inverting Precision to get U
        L = torch.linalg.cholesky(Precision)
        self.U = torch.cholesky_inverse(L).float().to(device)

        self.fitted = True

    def predict(self, Phi: torch.Tensor):
        device = Phi.device
        N = Phi.shape[0]
        bias = torch.ones(N, 1, device=device)
        Phi_aug = torch.cat([Phi, bias], dim=1)
        return Phi_aug @ self.mu

    def get_acquisition_score(self, Phi: torch.Tensor, method="determinant"):
        """
        Computes acquisition scores based on Predictive Variance.
        For Matrix Normal: Cov[y*] = (1 + phi^T U phi) * Sigma_noise
        We need to scalarize this matrix using either determinant or trace.
        """
        device = Phi.device
        N = Phi.shape[0]
        bias = torch.ones(N, 1, device=device)
        Phi_aug = torch.cat([Phi, bias], dim=1)

        # Compute scalar epistemic factor: c = 1 + phi^T U phi
        # We only need the diagonal term
        c = 1.0 + (Phi_aug @ self.U * Phi_aug).sum(dim=1)

        if method == "determinant":
            # det(c * Sigma) = c^K * det(Sigma)
            log_det_sigma = torch.logdet(self.noise_cov.to(device))
            scores = self.output_dim * torch.log(c) + log_det_sigma
            return scores

        elif method == "sum_marginal":
            # Tr(c * Sigma) = c * Tr(Sigma)
            tr_sigma = torch.trace(self.noise_cov.to(device))
            scores = c * tr_sigma
            return scores
