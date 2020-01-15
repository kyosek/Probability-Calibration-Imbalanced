# BMR (Bayes Minimum Risk) calibration implementation
# Pozzolo et al., 2015, Calibrating Probability with Undersampling

class BMR:
    def beta(binary_target):
        return binary_target.sum()/np.where(binary_target==0,1,0).sum()

    def tau(binary_target):
        return binary_target.sum()/len(binary_target)
        
    def calibration(prob, beta):
        return prob/(prob+(1-prob)/beta)