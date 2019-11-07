# BMR (Bayes Minimum Risk) implementation
# Pozzolo et al., 2015, Calibrating Probability with Undersampling

class BMR:
    def beta(binary_target):
        return binary_target.sum()/len(binary_target)

    def tau(binary_target, beta):
        return binary_target.sum()/(binary_target.sum()+(beta*np.where(binary_target==0,1,0).sum()))

    def calibration(proba, beta):
        return proba/(proba+(1-proba)/beta)