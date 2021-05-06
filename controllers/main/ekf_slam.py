import numpy as np
import math

class EKF_SLAM():
    def __init__(self, init_mu, init_P, dt, W, V, n):
        """Initialize EKF SLAM

        Create and initialize an EKF SLAM to estimate the robot's pose and
        the location of map features

        Args:
            init_mu: A numpy array of size (3+2*n, ). Initial guess of the mean 
            of state. 
            init_P: A numpy array of size (3+2*n, 3+2*n). Initial guess of 
            the covariance of state.
            dt: A double. The time step.
            W: A numpy array of size (3+2*n, 3+2*n). Process noise
            V: A numpy array of size (2*n, 2*n). Observation noise
            n: A int. Number of map features
            

        Returns:
            An EKF SLAM object.
        """
        self.mu = init_mu  # initial guess of state mean
        self.P = init_P  # initial guess of state covariance
        self.dt = dt  # time step
        self.W = W  # process noise 
        self.V = V  # observation noise
        self.n = n  # number of map features


    def _f(self, x, u):
        """Non-linear dynamic function.

        Compute the state at next time step according to the nonlinear dynamics f.

        Args:
            x: A numpy array of size (3+2*n, ). State at current time step.
            u: A numpy array of size (3, ). The control input [\dot{x}, \dot{y}, \dot{\psi}]

        Returns:
            x_next: A numpy array of size (3+2*n, ). The state at next time step
        """
        x_next = np.copy(x)

        X_t, Y_t, Phi_t = x[0:3]
        Phi_t = self._wrap_to_pi(Phi_t)
        A_t = np.array([[np.cos(Phi_t)*self.dt, -np.sin(Phi_t)*self.dt, 0.0],
                        [np.sin(Phi_t)*self.dt, np.cos(Phi_t)*self.dt, 0.0],
                        [0.0, 0.0, self.dt]])
        x_next[0:3] = x[0:3] + A_t @ u
        x_next[2] = self._wrap_to_pi(x_next[2])
        return x_next


    def _h(self, x):
        """Non-linear measurement function.

        Compute the sensor measurement according to the nonlinear function h.

        Args:
            x: A numpy array of size (3+2*n, ). State at current time step.

        Returns:
            y: A numpy array of size (2*n, ). The sensor measurement.
        """
        # extract info from current state
        Pt = x[0:2] # current position
        theta = x[2]

        x_idx = range(3, len(x), 2)
        y_idx = range(4, len(x), 2)
        Mx = x[x_idx]
        My = x[y_idx]
        assert len(Mx)==len(My), "x y length should be equal!"
        assert len(Mx)==(len(x)-3)/2, "not fully extracted feature!"
        M = np.vstack((Mx, My))

        # initialize measurement array
        y = np.zeros(self.n*2) # measurement

        # fill in positional and angular measurements
        for k in range(self.n):
            diff = M[:, k]-Pt
            y[k] = np.linalg.norm(diff)
            y[k+self.n] = math.atan2(diff[1], diff[0])-theta
        # ?? wrap angle?
        return y


    def _compute_F(self, u):
        """Compute Jacobian of f
        
        You will use self.mu in this function.

        Args:
            u: A numpy array of size (3, ). The control input [\dot{x}, \dot{y}, \dot{\psi}]

        Returns:
            F: A numpy array of size (3+2*n, 3+2*n). The jacobian of f evaluated at x_k.
        """
        xdot, ydot = u[0:2]
        theta = self.mu[2]
        F = np.eye(self.n*2+3)
        F[0][2] = -self.dt * (xdot*np.sin(theta) + ydot*np.cos(theta))
        F[1][2] = self.dt * (xdot*np.cos(theta) - ydot*np.sin(theta))

        return F


    def _compute_H(self):
        """Compute Jacobian of h
        
        You will use self.mu in this function.

        Args:

        Returns:
            H: A numpy array of size (2*n, 3+2*n). The jacobian of h evaluated at x_k.
        """
        Xr, Yr = self.mu[0:2]
        x_idx = range(3, len(self.mu), 2)
        y_idx = range(4, len(self.mu), 2)
        Mx = self.mu[x_idx]
        My = self.mu[y_idx]

        H = np.zeros((2*self.n, 3+2*self.n))

        d_vec = np.zeros(self.n)
        # 0. compute ds
        for k in range(self.n):
            d_vec[k] = np.sqrt((Xr-Mx[k])**2 + (Yr-My[k])**2)

        # 1. first 3 cols
        for k in range(self.n):
            H[k][0] = (Xr-Mx[k])/d_vec[k]
            H[k][1] = (Yr-My[k])/d_vec[k]
            H[k+self.n][0] = (My[k]-Yr)/(d_vec[k]**2)
            H[k+self.n][1] = (Xr-Mx[k])/(d_vec[k]**2)
            H[k+self.n][2] = -1.0

        # 2. remaining 2n cols
        for k in range(self.n):
            H[k][3+k*2] = (Mx[k]-Xr)/d_vec[k]
            H[k][3+k*2+1] = (My[k]-Yr)/d_vec[k]
            H[k+self.n][3+k*2] = (Yr-My[k])/(d_vec[k]**2)
            H[k+self.n][3+k*2+1] = (Mx[k]-Xr)/(d_vec[k]**2)

        return H


    def predict_and_correct(self, y, u):
        """Predice and correct step of EKF
        
        You will use self.mu in this function. You must update self.mu in this function.

        Args:
            y: A numpy array of size (2*n, ). The measurements according to the project description.
            u: A numpy array of size (3, ). The control input [\dot{x}, \dot{y}, \dot{\psi}]

        Returns:
            self.mu: A numpy array of size (3+2*n, ). The corrected state estimation
            self.P: A numpy array of size (3+2*n, 3+2*n). The corrected state covariance
        """

        # compute F and H matrix
        Ck = self._compute_H()
        Ak = self._compute_F(u)

        last_mu = self.mu
        #***************** Predict step *****************#
        # predict the state
        x_bar = self._f(last_mu, u)

        # predict the error covariance
        P_bar = Ak @ self.P @ Ak.T + self.W

        #***************** Correct step *****************#
        # compute the Kalman gain
        Lk = P_bar @ Ck.T @ np.linalg.inv(Ck @ P_bar @ Ck.T + self.V)

        # update estimation with new measurement
        self.mu = x_bar + Lk @ (y - self._h(x_bar))
        self.mu[2] = self._wrap_to_pi(self.mu[2])

        # update the error covariance
        self.P = (np.eye(len(P_bar)) - Lk @ Ck) @ P_bar

        return self.mu, self.P

    def wrap_angle(self, theta):
        return (theta + 2 * math.pi) % (2 * math.pi)

    def _wrap_to_pi(self, angle):
        angle_old = angle
        angle = angle - 2*np.pi*np.floor((angle+np.pi )/(2*np.pi))
        if angle_old!=angle:
            print("changed from ", angle_old, " to ", angle)
        return angle


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    m = np.array([[0.,  0.],
                  [0.,  20.],
                  [20., 0.],
                  [20., 20.],
                  [0,  -20],
                  [-20, 0],
                  [-20, -20],
                  [-50, -50]]).reshape(-1)

    dt = 0.01
    T = np.arange(0, 20, dt)
    n = int(len(m)/2)
    W = np.zeros((3+2*n, 3+2*n))
    W[0:3, 0:3] = dt**2 * 1 * np.eye(3)
    V = 0.1*np.eye(2*n)
    V[n:,n:] = 0.01*np.eye(n)

    # EKF estimation
    mu_ekf = np.zeros((3+2*n, len(T)))
    mu_ekf[0:3,0] = np.array([2.2, 1.8, 0.])
    # mu_ekf[3:,0] = m + 0.1
    mu_ekf[3:,0] = m + np.random.multivariate_normal(np.zeros(2*n), 0.5*np.eye(2*n))
    init_P = 1*np.eye(3+2*n)

    # initialize EKF SLAM
    slam = EKF_SLAM(mu_ekf[:,0], init_P, dt, W, V, n)
    
    # real state
    mu = np.zeros((3+2*n, len(T)))
    mu[0:3,0] = np.array([2, 2, 0.])
    mu[3:,0] = m

    y_hist = np.zeros((2*n, len(T)))
    for i, t in enumerate(T):
        if i > 0:
            # real dynamics
            u = [-5, 2*np.sin(t*0.5), 1*np.sin(t*3)]
            # u = [0.5, 0.5*np.sin(t*0.5), 0]
            # u = [0.5, 0.5, 0]
            mu[:,i] = slam._f(mu[:,i-1], u) + \
                np.random.multivariate_normal(np.zeros(3+2*n), W)

            # measurements
            y = slam._h(mu[:,i]) + np.random.multivariate_normal(np.zeros(2*n), V)
            y_hist[:,i] = (y-slam._h(slam.mu))
            # apply EKF SLAM
            mu_est, _ = slam.predict_and_correct(y, u)
            mu_ekf[:,i] = mu_est

            # debugger
            # print(t, ": ", mu_est[2])


    plt.figure(1, figsize=(10,6))
    ax1 = plt.subplot(121, aspect='equal')
    ax1.plot(mu[0,:], mu[1,:], 'b')
    ax1.plot(mu_ekf[0,:], mu_ekf[1,:], 'r--')
    mf = m.reshape((-1,2))
    ax1.scatter(mf[:,0], mf[:,1])
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')

    ax2 = plt.subplot(322)
    ax2.plot(T, mu[0,:], 'b')
    ax2.plot(T, mu_ekf[0,:], 'r--')
    ax2.set_xlabel('t')
    ax2.set_ylabel('X')

    ax3 = plt.subplot(324)
    ax3.plot(T, mu[1,:], 'b')
    ax3.plot(T, mu_ekf[1,:], 'r--')
    ax3.set_xlabel('t')
    ax3.set_ylabel('Y')

    ax4 = plt.subplot(326)
    ax4.plot(T, mu[2,:], 'b')
    ax4.plot(T, mu_ekf[2,:], 'r--')
    ax4.set_xlabel('t')
    ax4.set_ylabel('psi')

    plt.figure(2)
    ax1 = plt.subplot(211)
    ax1.plot(T, y_hist[0:n, :].T)
    ax2 = plt.subplot(212)
    ax2.plot(T, y_hist[n:, :].T)

    plt.show()
