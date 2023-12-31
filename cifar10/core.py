import torch
from scipy.stats import norm, binom_test
import numpy as np
from math import ceil
from statsmodels.stats.proportion import proportion_confint


# Adapted from https://github.com/locuslab/smoothing/blob/master/code/core.py
class Smooth(object):
    """A smoothed classifier g """

    # to abstain, Smooth returns this int
    ABSTAIN = -1

    def __init__(self, base_classifier: torch.nn.Module, num_classes: int, sigma: float, t: int):
        """
        :param base_classifier: maps from [batch x channel x height x width] to [batch x num_classes]
        :param num_classes:
        :param sigma: the noise level hyperparameter
        """
        self.base_classifier = base_classifier
        self.num_classes = num_classes
        self.sigma = sigma
        self.t = t 

    def certify(self, x: torch.tensor, n0: int, n: int, alpha: float, batch_size: int) -> (int, float):
        """ Monte Carlo algorithm for certifying that g's prediction around x is constant within some L2 radius.
        With probability at least 1 - alpha, the class returned by this method will equal g(x), and g's prediction will
        robust within a L2 ball of radius R around x.
        :param x: the input [channel x height x width]
        :param n0: the number of Monte Carlo samples to use for selection
        :param n: the number of Monte Carlo samples to use for estimation
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: (predicted class, certified radius)
                 in the case of abstention, the class will be ABSTAIN and the radius 0.
        """
        self.base_classifier.eval()
        # draw samples of f(x+ epsilon)
        counts_selection = self._sample_noise(x, n0, batch_size)
        # use these samples to take a guess at the top class
        cAHat = counts_selection.argmax().item()
        # draw more samples of f(x + epsilon)
        counts_estimation = self._sample_noise(x, n, batch_size)
        # use these samples to estimate a lower bound on pA
        nA = counts_estimation[cAHat].item()
        pABar = self._lower_confidence_bound(nA, n, alpha)
        if pABar < 0.5:
            return Smooth.ABSTAIN, 0.0
        else:
            radius = self.sigma * norm.ppf(pABar)
            return cAHat, radius

    def predict(self, x: torch.tensor, n: int, alpha: float, batch_size: int) -> int:
        """ Monte Carlo algorithm for evaluating the prediction of g at x.  With probability at least 1 - alpha, the
        class returned by this method will equal g(x).
        This function uses the hypothesis test described in https://arxiv.org/abs/1610.03944
        for identifying the top category of a multinomial distribution.
        :param x: the input [channel x height x width]
        :param n: the number of Monte Carlo samples to use
        :param alpha: the failure probability
        :param batch_size: batch size to use when evaluating the base classifier
        :return: the predicted class, or ABSTAIN
        """
        self.base_classifier.eval()
        counts = self._sample_noise(x, n, batch_size)
        top2 = counts.argsort()[::-1][:2]
        count1 = counts[top2[0]]
        count2 = counts[top2[1]]

        if binom_test(count1, count1 + count2, p=0.5) > alpha:
            return Smooth.ABSTAIN
        else:
            return top2[0]

    
    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))

                predictions = self.base_classifier(batch, self.t).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts
    '''


    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under Laplace noise corruptions of the input x.
        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :param laplace_scale: scale parameter for Laplace distribution 0.5
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))

                predictions = self.base_classifier(batch, self.t)
                predictions_cpu = predictions.cpu()  # Move predictions to CPU
                noisy_predictions = predictions_cpu + np.random.laplace(scale=0.5, size=predictions_cpu.shape)

                noisy_argmax = np.argmax(noisy_predictions, axis=1)
                counts += self._count_arr(noisy_argmax, self.num_classes)
            return counts
    '''
    '''
        
    ## Laplace noise

    ## In this implementation, we create a new batch tensor by iterating over the input x and adding noise to each image individually. 
    ## Specifically, for each image in the batch, we generate a random noise tensor from a Laplace distribution with mean 0 
    ## and a standard deviation proportional to the L2 norm of the image. We then normalize this noise tensor and add it to the image 
    ## to create a new, noisy image. Finally, we append the noisy image to the batch tensor.

    ## We use the L2 norm of the image to scale the noise because the amount of noise added should depend on the content of the image, 
    ## and the L2 norm is a reasonable measure of image content. 
    ## We also use the L2 norm to scale the noise added in the original implementation.

    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                # create a new batch tensor by adding noise to each image individually
                batch = []
                for i in range(this_batch_size):
                    image = x[i:i+1]
                    noise = torch.distributions.laplace.Laplace(0, self.sigma * torch.norm(image).item()).sample(image.size()).to(device='cuda')
                    normalized_noise = noise / (torch.norm(noise.view(-1)) + 1e-8) * self.sigma * torch.norm(image).item()
                    noisy_image = image + normalized_noise
                    batch.append(noisy_image)
                batch = torch.cat(batch, dim=0)

                predictions = self.base_classifier(batch).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)

            return counts
    
    '''
    
    '''
    
    ## add L2 norm Laplace noise to each input image
    
    ##Generate random noise with the same shape as the input tensor x using torch.randn_like().

    ##Calculate the L2 norm of the noise tensor along the channel dimension using torch.norm().

    ##Generate a random scale factor for each image in the batch using torch.rand() and 
    ##adjust the scale factor based on the L2 norm calculated in step 2.

    ##Multiply the noise tensor by the scale factor.

    ##Add the scaled noise tensor to the input tensor x.
    
    
    def _sample_noise(self, x: torch.tensor, num: int, batch_size) -> np.ndarray:
        """ Sample the base classifier's prediction under noisy corruptions of the input x.

        :param x: the input [channel x width x height]
        :param num: number of samples to collect
        :param batch_size:
        :return: an ndarray[int] of length num_classes containing the per-class counts
        """
        with torch.no_grad():
            counts = np.zeros(self.num_classes, dtype=int)
            for _ in range(ceil(num / batch_size)):
                this_batch_size = min(batch_size, num)
                num -= this_batch_size

                batch = x.repeat((this_batch_size, 1, 1, 1))
                noise = torch.randn_like(batch, device='cuda')
                noise_norm = torch.norm(noise, p=2, dim=1, keepdim=True)
                scale_factors = torch.rand((this_batch_size, 1, 1, 1), device='cuda')
                scale_factors *= self.sigma / noise_norm
                scaled_noise = noise * scale_factors
                predictions = self.base_classifier(batch + scaled_noise).argmax(1)
                counts += self._count_arr(predictions.cpu().numpy(), self.num_classes)
            return counts
    '''        
            
    def _count_arr(self, arr: np.ndarray, length: int) -> np.ndarray:
        counts = np.zeros(length, dtype=int)
        for idx in arr:
            counts[idx] += 1
        return counts

    def _lower_confidence_bound(self, NA: int, N: int, alpha: float) -> float:
        """ Returns a (1 - alpha) lower confidence bound on a bernoulli proportion.
        This function uses the Clopper-Pearson method.
        :param NA: the number of "successes"
        :param N: the number of total draws
        :param alpha: the confidence level
        :return: a lower bound on the binomial proportion which holds true w.p at least (1 - alpha) over the samples
        """
        return proportion_confint(NA, N, alpha=2 * alpha, method="beta")[0]