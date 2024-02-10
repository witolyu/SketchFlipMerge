# This is a sample Python script.
import math
import numpy as np


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

    B = 8192*2
    P = 24
    n = 1000000
    n_U = 2000000/6*5
    rho = [2**(-min(j,P-1))/B for j in range(P)]
    gamma= [1-rho[j] for j in range(P)]

    eps = 2
    p = math.e**eps/(math.e**eps + 1)
    q = 1 - p

    # given the orginal eps, derive the equivalence eps_star after the (union) merge, according to Theorem 4.8

    eps_star = - math.log(2*math.e**(-eps)-math.e**(-2 * eps))

    # Compute the corresponding p and q, according to Definition 4.6
    p = math.e ** eps_star / (math.e ** eps_star + 1)
    q = 1 - p

    # derive the SE for cardinality estimation through log likelihood,
    # according to Sec 5.1, equation 2. (squareroot of inverse of the hessian)

    estimated_SE = (B*(p-q)*sum([(math.log(gamma[j]))**2* gamma[j]**n_U *
                                 (p/(p-(p-q)*gamma[j]**n_U)- (1-p)/(1-p+(p-q)*gamma[j]**n_U)) for j in range(P)]))**(-1/2)

    print("B:{}, P:{}, Communication_cost(bits):{}".format(B, P, B*P))

    print("estimated_SE:{}, estimated_relative_error:{}".format(estimated_SE, estimated_SE/n_U))

    # For comparison, we assume the same communication, assuming hash outputs are 64 bits.
    k = B*P /64
    JI = (2*n- n_U)/n_U

    print("JI:",JI)

    JI_SD = (k*JI*(1-JI))**(1/2)/k

    print("# of iterations k:{}, JI:{}, estimated_JI_SD:{}, relative JI error:{}".format(k,JI, JI_SD,JI_SD/JI))

    # n_U = 2n/(JI+1)

    num_samples = 1000

    # Sampling from the binomial distribution
    samples_matching = np.random.binomial(k, JI, num_samples)

    samples_JI = [val / k  for val in samples_matching]

    RRMSE_JI = (np.mean([(val-JI)**2 for val in samples_JI]))**(1/2)/JI

    samples_Union = [2*n/(val/k+1) for val in samples_matching]

    union_se = (np.mean([(val-n_U)**2 for val in samples_Union]))**(1/2)

    RRMSE_U = union_se/n_U

    # print(samples_matching)
    # print(samples_Union)
    print("Experimental relative error (JI):{}".format(RRMSE_JI))
    print("Experimental relative error (Union):{}".format(RRMSE_U))

    # Estimated relative standard error.

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
