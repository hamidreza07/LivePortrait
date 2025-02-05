import torch


class KalmanFilterTorch:
    def __init__(self, F, H, Q, R, initial_state, P, device):
        """
        Args:
            F (torch.Tensor): State transition matrix.
            H (torch.Tensor): Observation matrix.
            Q (torch.Tensor): Process noise covariance.
            R (torch.Tensor): Observation noise covariance.
            initial_state (torch.Tensor): Initial state estimate.
            P (torch.Tensor): Initial covariance estimate.
            device (torch.device): Device for computations (CPU/GPU).
        """
        self.device = device
        self.F = F.to(device)
        self.H = H.to(device)
        self.Q = Q.to(device)
        self.R = R.to(device)
        self.state_estimate = initial_state.to(device)
        self.estimate_covariance = P.to(device)

        self.eps = 1e-1000 # Regularization for stability
        self._validate_parameters()

    def _validate_parameters(self):
        assert self.F.dim() == 2, "State transition matrix F must be 2D."
        assert self.F.shape[0] == self.F.shape[1], "State transition matrix F must be square."
        assert self.H.dim() == 2, "Observation matrix H must be 2D."
        assert self.H.shape[1] == self.F.shape[0], "Observation matrix H must match state dimensions."
        assert self.Q.shape == self.F.shape, "Process noise covariance Q must match F dimensions."
        assert self.R.shape[0] == self.H.shape[0] and self.R.shape[1] == self.H.shape[0], \
            "Observation noise covariance R must match observation dimensions and be square."


    def predict(self, control_input=None, B=None, F=None):
        """
        Predict the next state.
        Args:
            control_input (torch.Tensor, optional): Control input vector.
            B (torch.Tensor, optional): Control matrix.
            F (torch.Tensor, optional): Dynamic state transition matrix.
        """
        if F is not None:
            self.F = F.to(self.device)

        # Predict state estimate
        self.state_estimate = self.F @ self.state_estimate
        if control_input is not None and B is not None:
            self.state_estimate += B.to(self.device) @ control_input.to(self.device)

        # Predict covariance
        self.estimate_covariance = self.F @ self.estimate_covariance @ self.F.T + self.Q

        # Numerical stability
        self.estimate_covariance += torch.eye(self.estimate_covariance.size(0), device=self.device) * self.eps

    def update(self, observation):
        """
        Update the state estimate with a new observation.
        Args:
            observation (torch.Tensor): Observation vector.
        Returns:
            torch.Tensor: Updated state estimate.
        """
        # Compute Kalman Gain
        S = self.H @ self.estimate_covariance @ self.H.T + self.R
        S += torch.eye(S.size(0), device=self.device) * self.eps  # Regularization
        K = self.estimate_covariance @ self.H.T @ torch.linalg.inv(S)

        # Update state estimate
        residual = observation.to(self.device) - self.H @ self.state_estimate
        self.state_estimate += K @ residual

        # Update covariance estimate
        I = torch.eye(self.estimate_covariance.size(0), device=self.device)
        self.estimate_covariance = (I - K @ self.H) @ self.estimate_covariance

        return self.state_estimate, residual

    def smooth(self, observations, forward_only=True):
        """
        Apply Kalman filter to a sequence of observations.
        Args:
            observations (torch.Tensor): Sequence of observations.
            forward_only (bool): If False, applies forward-backward smoothing.
        Returns:
            list[torch.Tensor]: Smoothed state estimates.
        """
        smoothed_states = []
        residuals = []

        # Forward pass
        for obs in observations:
            self.predict()
            state, residual = self.update(obs)
            smoothed_states.append(state.clone())
            residuals.append(residual.clone())

        if forward_only:
            return smoothed_states, residuals

        # Backward pass
        for i in range(len(smoothed_states) - 2, -1, -1):
            P_pred = self.F @ self.estimate_covariance @ self.F.T + self.Q
            J = self.estimate_covariance @ self.F.T @ torch.linalg.inv(P_pred)
            smoothed_states[i] += J @ (smoothed_states[i + 1] - self.F @ smoothed_states[i])

        return smoothed_states, residuals

    def adapt_noise_covariance(self, residuals, adapt_rate=0.01, decay_factor=0.9):
        residual_variances = torch.var(torch.stack(residuals), dim=0)
        self.Q = decay_factor * self.Q + (1 - decay_factor) * torch.diag(residual_variances)
        self.R = decay_factor * self.R + (1 - decay_factor) * torch.diag(residual_variances)



# Usage example remains the same
def smooth(x_d_lst, shape, device, observation_variance=3e-7, process_variance=500e35, adapt=True):
    x_d_lst_reshape = [x.reshape(-1) for x in x_d_lst]
    x_d_stacked = torch.stack([torch.tensor(x, device=device) for x in x_d_lst_reshape])

    # Define Kalman filter parameters
    dim = x_d_stacked.size(1)
    F = torch.eye(dim, device=device)
    H = torch.eye(dim, device=device)
    Q = torch.eye(dim, device=device) * process_variance
    R = torch.eye(dim, device=device) * observation_variance
    initial_state = x_d_stacked[0]
    P = torch.eye(dim, device=device) * process_variance

    # Initialize Kalman filter
    kf_torch = KalmanFilterTorch(F, H, Q, R, initial_state, P, device)

    smoothed_states = []
    residuals = []

    # Smooth the data
    for obs in x_d_stacked:
        kf_torch.predict()
        smoothed_state, residual = kf_torch.update(obs)
        smoothed_states.append(smoothed_state.clone())
        residuals.append(residual.clone())

    # Adapt noise covariances dynamically
    if adapt:
        kf_torch.adapt_noise_covariance(residuals)

    x_d_lst_smooth = [state.reshape(shape[-2:]) for state in smoothed_states]
    return x_d_lst_smooth
