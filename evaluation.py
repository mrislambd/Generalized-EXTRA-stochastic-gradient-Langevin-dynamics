import numpy as np
from scipy.linalg import sqrtm
import math


class ClassificationAccuracy:
    """Calculate the classification accuracy in Bayesian Logistic Regression

    Args:
        x_all (list): the input data
        y_all (list): the output data
        history_all (list): contains the approximation from all the nodes
        T (int): the number of iterations
    """

    def __init__(self, x_all, y_all, history_all, T):
        self.x_all = x_all
        self.y_all = y_all
        self.history_all = history_all
        self.T = T

    def compute_accuracy(self):
        """Function that computes the classification accuracy

        Args:
            params: the same parameters from the class

        Returns:
        result_acc (float): the classification accuracy
        result_std (float): the standard deviation of the classification
        accuracy
        """
        mis_class = np.empty((self.T + 1, len(self.history_all[0, 0, 0])))

        for t in range(self.T + 1):
            for n in range(len(self.history_all[t, 0, 0])):
                temp0 = 0
                for i in range(len(self.x_all)):
                    z = 1 / (
                        1
                        + np.exp(
                            -np.dot(
                                np.transpose(self.history_all[t, 1])[n],
                                self.x_all[i],
                            )
                        )
                    )
                    if z >= 0.5:
                        z = 1
                    else:
                        z = 0
                    if self.y_all[i] != z:
                        temp0 += 1
                mis_class[t, n] = 1 - temp0 / len(self.x_all)

        result_acc = np.mean(mis_class, axis=1)
        result_std = np.std(mis_class, axis=1)
        return result_acc, result_std


class Wasserstein2Distance:
    """Class for: Wasserstein 2 distance in Bayesian Linear Regression

    Args:
        size_w (int): the size of the network
        T (int): the number of iterations
        avg_post (list): the mean of the posterior distribution
        cov_post (list): the covariance of the posterior distribution
        history_all (list): contains the approximation from all the nodes
        beta_mean_all (list): contains the mean of the approximation
        from all the nodes
    """

    def __init__(
        self, size_w, T, avg_post, cov_post, history_all, beta_mean_all
    ):
        self.size_w = size_w
        self.T = T
        self.avg_post = avg_post
        self.cov_post = cov_post
        self.history_all = history_all
        self.beta_mean_all = beta_mean_all

    def W2_dist(self):
        """Class for: Wasserstein 2 distance in Bayesian Linear Regression

        Args:
            size_w (int): the size of the network
            T (int): the number of iterations
            avg_post (list): the mean of the posterior distribution
            cov_post (list): the covariance of the posterior distribution
            history_all (list): contains the approximation from all the nodes
            beta_mean_all (list): contains the mean of the approximation
            from all the nodes
        Returns:
        w2dis (list): contains the W2 distance of each agent and
        the mean of the approximation from all the agents
        """
        w2dis = []
        for i in range(self.size_w):
            temp = []
            w2dis.append(temp)
        temp = []
        w2dis.append(temp)
        """
        W2 distance of each agent
        """
        for i in range(self.size_w):
            for t in range(self.T + 1):
                d = 0
                avg_temp = []
                avg_temp.append(np.mean(self.history_all[t][i][0]))
                avg_temp.append(np.mean(self.history_all[t][i][1]))
                avg_temp = np.array(avg_temp)
                cov_temp = np.cov(self.history_all[t][i])
                d = np.linalg.norm(self.avg_post - avg_temp) * np.linalg.norm(
                    self.avg_post - avg_temp
                )
                d = d + np.trace(
                    self.cov_post
                    + cov_temp
                    - 2
                    * sqrtm(
                        np.dot(
                            np.dot(sqrtm(cov_temp), self.cov_post),
                            sqrtm(cov_temp),
                        )
                    )
                )
                w2dis[i].append(np.array(math.sqrt(abs(d))))
        """
        W2 distance of the mean of agents
        """
        for t in range(self.T + 1):
            d = 0
            avg_temp = []
            avg_temp.append(np.mean(self.beta_mean_all[t][0]))
            avg_temp.append(np.mean(self.beta_mean_all[t][1]))
            avg_temp = np.array(avg_temp)
            cov_temp = np.cov(self.beta_mean_all[t])
            d = np.linalg.norm(self.avg_post - avg_temp) * np.linalg.norm(
                self.avg_post - avg_temp
            )
            d = d + np.trace(
                self.cov_post
                + cov_temp
                - 2
                * sqrtm(
                    np.dot(
                        np.dot(sqrtm(cov_temp), self.cov_post), sqrtm(cov_temp)
                    )
                )
            )
            w2dis[self.size_w].append(np.array(math.sqrt(abs(d))))

        for i in range(len(w2dis)):
            w2dis[i] = np.array(w2dis[i])

        return w2dis
