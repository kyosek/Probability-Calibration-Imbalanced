# Comparison of prediction means and true mean in the test dataset
y_pred_calib_rus.mean()
y_pred_calib_bc.mean()
y_pred_rus.mean()
y_pred_bc.mean()
te.Class.mean()

sns.distplot(y_pred_calib_bc,bins=50,kde=False)
plt.show()