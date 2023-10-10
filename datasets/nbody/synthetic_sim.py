import numpy as np
import matplotlib.pyplot as plt
import time

"""
See https://github.com/RobDHess/Steerable-E3-GNN for the original code.
"""   

class SpringSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=.5, vel_norm=.5,
                 interaction_strength=.1, noise_var=0., k_plane=1.):
        """
        Simulates a system of particles connected by springs.
        Args:
            n_balls: number of particles
            box_size: size of the box where particles are simulated
            loc_std: standard deviation of initial location
            vel_norm: norm of initial velocity
            interaction_strength: spring constant
            noise_var: variance of noise added to observations
            k_plane: stiffness of the spring attached from each particle to the XY plane
        """
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var
        self.k_plane = k_plane

        self._spring_types = np.array([0., 0.5, 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T
        self.dim = 3
        
    def _energy(self, loc, vel, edges):
        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] * (dist ** 2) / 2
            return U + K

    def _clamp(self, loc, vel):
        """
        Input:
            loc: 2xN location at one time stamp
            vel: 2xN velocity at one time stamp
        Output:
            location and velocity after hiting walls and returning after
            elastically colliding with walls
        """
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def sample_trajectory(self, T=10000, sample_freq=10,
                          spring_prob=[1. / 2, 0, 1. / 2]):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # sample the length between a particle and XY plane (shape = (self.n_balls, 1))
        lengths = np.random.rand(self.n_balls)
        # Sample edges
        edges = np.random.choice(self._spring_types,
                                 size=(self.n_balls, self.n_balls),
                                 p=spring_prob)
        edges = np.tril(edges) + np.tril(edges, -1).T
        np.fill_diagonal(edges, 0)
        # Initialize location and velocity
        loc = np.zeros((T_save, self.dim, n))
        vel = np.zeros((T_save, self.dim, n))
        loc_next = np.random.randn(self.dim, n) * self.loc_std
        vel_next = np.random.randn(self.dim, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            forces_size = - self.interaction_strength * edges
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            # Hook's law between patricles + Hook's law between particles and XY plane
            # Force acting on particle i from the XY plane is - k * (displacement z - length of the spring)
            F_plane = np.zeros((n,3))
            F_plane[:, 2] = -self.k_plane * (loc_next[2, :] - lengths)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n),
                     np.subtract.outer(loc_next[2, :],
                                       loc_next[2, :]).reshape(1, n, n)
                 ))).sum(
                axis=-1)
            F += F_plane.T

            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                #loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                forces_size = - self.interaction_strength * edges
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
                # Hook's law between patricles + Hook's law between particles and XY plane
                F_plane = np.zeros((n,3))
                F_plane[:, 2] = -self.k_plane * (loc_next[2, :]-lengths)
                F = (forces_size.reshape(1, n, n) *
                    np.concatenate((
                        np.subtract.outer(loc_next[0, :],
                                        loc_next[0, :]).reshape(1, n, n),
                        np.subtract.outer(loc_next[1, :],
                                        loc_next[1, :]).reshape(1, n, n),
                        np.subtract.outer(loc_next[2, :],
                                        loc_next[2, :]).reshape(1, n, n)
                    ))).sum(
                    axis=-1)
                F += F_plane.T
                
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            return loc, vel, edges, lengths.reshape(n,1)


class ChargedParticlesSim(object):
    def __init__(self, n_balls=5, box_size=5., loc_std=1., vel_norm=0.5,
                 interaction_strength=[1.,1.,1.], noise_var=0., e_z=1.):
        """
        Simulates a system of charged particles.
        Args:
            n_balls: number of particles
            box_size: size of the box where particles are simulated
            loc_std: standard deviation of initial location
            vel_norm: norm of initial velocity
            interaction_strength: spring constant
            noise_var: variance of noise added to observations
            e_z: strength of the electric field in the z direction
        """
        self.n_balls = n_balls
        self.box_size = box_size
        self.loc_std = loc_std
        self.loc_std = loc_std * (float(n_balls)/5.) ** (1/3)
        self.vel_norm = vel_norm
        self.interaction_strength = interaction_strength
        self.noise_var = noise_var
        self.e_z = e_z

        self._charge_types = np.array([-1., 0., 1.])
        self._delta_T = 0.001
        self._max_F = 0.1 / self._delta_T
        self.dim = 3

    def _l2(self, A, B):
        """
        Input: A is a Nxd matrix
               B is a Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm
            between A[i,:] and B[j,:]
        i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
        """
        A_norm = (A ** 2).sum(axis=1).reshape(A.shape[0], 1)
        B_norm = (B ** 2).sum(axis=1).reshape(1, B.shape[0])
        dist = A_norm + B_norm - 2 * A.dot(B.transpose())
        return dist

    def _energy(self, loc, vel, edges):

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):

            K = 0.5 * (vel ** 2).sum()
            U = 0
            for i in range(loc.shape[1]):
                for j in range(loc.shape[1]):
                    if i != j:
                        r = loc[:, i] - loc[:, j]
                        dist = np.sqrt((r ** 2).sum())
                        U += 0.5 * self.interaction_strength * edges[
                            i, j] / dist
            return U + K

    def _clamp(self, loc, vel):
        """
        Input:
            loc: 2xN location at one time stamp
            vel: 2xN velocity at one time stamp
        Output:
            location and velocity after hiting walls and returning after
            elastically colliding with walls
        """
        assert (np.all(loc < self.box_size * 3))
        assert (np.all(loc > -self.box_size * 3))

        over = loc > self.box_size
        loc[over] = 2 * self.box_size - loc[over]
        assert (np.all(loc <= self.box_size))

        # assert(np.all(vel[over]>0))
        vel[over] = -np.abs(vel[over])

        under = loc < -self.box_size
        loc[under] = -2 * self.box_size - loc[under]
        # assert (np.all(vel[under] < 0))
        assert (np.all(loc >= -self.box_size))
        vel[under] = np.abs(vel[under])

        return loc, vel

    def sample_trajectory(self, T=10000, sample_freq=10,
                          charge_prob=[1. / 2, 0, 1. / 2]):
        n = self.n_balls
        assert (T % sample_freq == 0)
        T_save = int(T / sample_freq - 1)
        diag_mask = np.ones((n, n), dtype=bool)
        np.fill_diagonal(diag_mask, 0)
        counter = 0
        # Sample edges
        charges = np.random.choice(self._charge_types, size=(self.n_balls, 1),
                                   p=charge_prob)
        edges = charges.dot(charges.transpose())
        # Initialize location and velocity
        loc = np.zeros((T_save, self.dim, n))
        vel = np.zeros((T_save, self.dim, n))
        loc_next = np.random.randn(self.dim, n) * self.loc_std
        vel_next = np.random.randn(self.dim, n)
        v_norm = np.sqrt((vel_next ** 2).sum(axis=0)).reshape(1, -1)
        vel_next = vel_next * self.vel_norm / v_norm
        loc[0, :, :], vel[0, :, :] = self._clamp(loc_next, vel_next)

        # disables division by zero warning, since I fix it with fill_diagonal
        with np.errstate(divide='ignore'):
            # half step leapfrog
            l2_dist_power3 = np.power(
                self._l2(loc_next.transpose(), loc_next.transpose()), 3. / 2.)

            # size of forces up to a 1/|r| factor
            # since I later multiply by an unnormalized r vector
            forces_size = edges / l2_dist_power3
            np.fill_diagonal(forces_size,
                             0)  # self forces are zero (fixes division by zero)
            assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
            # Lorentz force F = q*E of electric field E = (0, 0, self.E) applied to each particle
            F_lorentz = np.zeros((n,3))
            F_lorentz[:,2] = self.e_z * charges.reshape(-1)
            F = (forces_size.reshape(1, n, n) *
                 np.concatenate((
                     np.subtract.outer(loc_next[0, :],
                                       loc_next[0, :]).reshape(1, n, n)*self.interaction_strength[0],
                     np.subtract.outer(loc_next[1, :],
                                       loc_next[1, :]).reshape(1, n, n)*self.interaction_strength[1],
                     np.subtract.outer(loc_next[2, :],
                                       loc_next[2, :]).reshape(1, n, n)*self.interaction_strength[2]
                                ))).sum(axis=-1) + F_lorentz.T
            F[F > self._max_F] = self._max_F
            F[F < -self._max_F] = -self._max_F

            vel_next += self._delta_T * F
            # run leapfrog
            for i in range(1, T):
                loc_next += self._delta_T * vel_next
                #loc_next, vel_next = self._clamp(loc_next, vel_next)

                if i % sample_freq == 0:
                    loc[counter, :, :], vel[counter, :, :] = loc_next, vel_next
                    counter += 1

                l2_dist_power3 = np.power(
                    self._l2(loc_next.transpose(), loc_next.transpose()),
                    3. / 2.)
                forces_size = edges / l2_dist_power3
                np.fill_diagonal(forces_size, 0)
                # assert (np.abs(forces_size[diag_mask]).min() > 1e-10)
                # Lorentz force F = q*E of electric field E = (0, 0, self.E) applied to each particle
                F_lorentz = np.zeros((n,3))
                F_lorentz[:,2] = self.e_z * charges.reshape(-1)
                F = (forces_size.reshape(1, n, n) *
                     np.concatenate((
                         np.subtract.outer(loc_next[0, :],
                                           loc_next[0, :]).reshape(1, n, n)*self.interaction_strength[0],
                         np.subtract.outer(loc_next[1, :],
                                           loc_next[1, :]).reshape(1, n, n)*self.interaction_strength[1],
                         np.subtract.outer(loc_next[2, :],
                                           loc_next[2, :]).reshape(1, n, n)*self.interaction_strength[2]
                     ))).sum(axis=-1) + F_lorentz.T
                F[F > self._max_F] = self._max_F
                F[F < -self._max_F] = -self._max_F
                vel_next += self._delta_T * F
            # Add noise to observations
            loc += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            vel += np.random.randn(T_save, self.dim, self.n_balls) * self.noise_var
            return loc, vel, edges, charges
