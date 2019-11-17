# Logistic regression
logit = LogisticRegression(random_state=42, solver='lbfgs',max_iter=1000)

# Rundom Under Sampling (RUS)
tr_x_rus, tr_y_rus = rus.fit_resample(tr.drop(['Class'],1),tr.Class)
logit_rus = logit.fit(tr_x_rus,tr_y_rus)

# RUSBoost
bc = BalancedBaggingClassifier(base_estimator=logit,random_state=42)
logit_bc = bc.fit(tr.drop(['Class'],1),tr.Class)

# Calibration
beta = BMR.beta(tr.Class)
tau = BMR.tau(tr.Class)
# with rundom undersampling
y_pred_calib_rus = BMR.calibration(prob=logit_rus.predict_proba(te.drop(['Class'],1))[:,1],beta=beta)
y_predicted_calib_rus = np.where(y_pred_calib_rus>=.99,1,0)
# with RUSBoost
y_pred_calib_bc = BMR.calibration(prob=logit_bc.predict_proba(te.drop(['Class'],1))[:,1],beta=beta)
y_predicted_calib_bc = np.where(y_pred_calib_bc>=tau,1,0)