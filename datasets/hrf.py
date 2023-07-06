from benchopt import BaseDataset
from benchopt import safe_import_context


with safe_import_context() as import_ctx:
    import numpy as np
    from nilearn.glm.first_level import compute_regressor
    from scipy.sparse.linalg import LinearOperator


class Dataset(BaseDataset):
    """Dataset modeling a block designed BOLD fMRI time serie."""

    name = "HRF"

    parameters = {
        "sim_tr": [1.0],  # time resolution in ms
        "block_on": [10],  # time on in seconds
        "block_off": [10],  # time off in seconds
        "n_blocks": [5],  # number of blocks
        "use_hrf": [True, False],  # If the data is convolve with an HRF.
        "noise_level": [0.1, 0.5, 1, 2, 5, 10],  # noise level
        "random_state": [42],
    }
    requirements = ["pip:nilearn"]

    def __init__(
        self,
        sim_tr=1,
        n_blocks=2,
        block_on=10,
        block_off=20,
        use_hrf=True,
        noise_level=0.1,
        random_state=42,
    ):
        self.sim_tr = sim_tr
        self.n_blocks = n_blocks
        self.block_on = block_on
        self.block_off = block_off
        self.use_hrf = use_hrf
        self.noise_level = noise_level
        self.random_state = random_state

    def get_data(self):
        rng = np.random.RandomState(self.random_state)
        block_size = self.block_on + self.block_off
        n_samples = self.n_blocks * block_size
        duration = self.sim_tr * n_samples

        # create a repeating block design
        event = []
        t = 0
        while t < duration:
            event.append((t, self.block_on, 1))
            t += block_size
        events = np.array(event)
        if self.use_hrf:
            regressor, _ = compute_regressor(
                events.T,
                "glover",
                np.arange(0, n_samples) * self.sim_tr,
                oversampling=50,
                min_onset=-24,
            )
            regressor = np.squeeze(regressor)
        else:
            regressor = np.zeros(n_samples)
            while t < duration:
                regressor[t: t + self.block_on] = 1
                t += block_size

        # Add noise
        noise = rng.randn(n_samples)
        noise *= np.linalg.norm(regressor) * self.noise_level
        regressor_noise = regressor + noise

        # A = Identity
        # y = A x + noise

        return {
            "A": LinearOperator(
                dtype=np.float64,
                matvec=lambda x: x,
                matmat=lambda X: X,
                rmatvec=lambda x: x,
                rmatmat=lambda X: X,
                shape=(n_samples, n_samples),
            ),
            "y": regressor_noise,
            "x": regressor,
        }
