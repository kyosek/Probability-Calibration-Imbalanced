# Evaluation
## Random Under Sampling (RUS)
y_pred_rus, y_predicted_rus = make_prediction(model=logit_rus, X=te.drop(['Class'],1), threshold=.5)
evaluation(te.Class, y_predicted_rus)

## Undersampling + Bagging
y_pred_bc, y_predicted_bc = make_prediction(model=logit_bc, X=te.drop(['Class'],1), threshold=.5)
evaluation(te.Class, y_predicted_bc)

## Calibration with Rundom undersampling
evaluation(te.Class, y_predicted_calib_rus)

## Calibration with Rundom undersampling
evaluation(te.Class, y_predicted_calib_bc)

