# See the class distribution with features
feats_distribution = pd.DataFrame(pca.fit_transform(tr.drop(['Class'],1)),columns=['before']).reset_index(drop=True)
feats_distribution['Class'] = tr['Class'].reset_index(drop=True)
feats_distribution_rus = pd.DataFrame(pca.fit_transform(tr_x_rus),columns=['after']).reset_index(drop=True)
feats_distribution_rus['Class'] = tr_y_rus
feats_distribution_test = pd.DataFrame(pca.fit_transform(te.drop(['Class'],1)),columns=['test']).reset_index(drop=True)
feats_distribution_test['Class'] = te['Class'].reset_index(drop=True)

sns.regplot(x='before',y='Class',data=feats_distribution,logistic=True, n_boot=500, y_jitter=.03)
plt.title('Class distribution of training set')
sns.regplot(x='after',y='Class',data=feats_distribution_rus,logistic=True, n_boot=500, y_jitter=.03)
plt.title('Class distribution of training set after undersampling')
sns.regplot(x='test',y='Class',data=feats_distribution_test,logistic=True, n_boot=500, y_jitter=.03)
plt.title('Class distribution of test set')
plt.show()

feats_distribution_rus['feats'] = pd.DataFrame(pca.fit_transform(tr_x_rus))
sns.scatterplot(x=feats_distribution['feats'],y=tr.Class)
plt.show()


# Comparison of prediction means and true mean in the test dataset
y_pred_calib_rus.mean()
y_pred_calib_bc.mean()
y_pred_rus.mean()
y_pred_bc.mean()
te.Class.mean()

sns.distplot(y_pred_calib_bc,bins=50,kde=False)
sns.distplot(y_pred_bc,bins=500,kde=False,color='red')
plt.axvline(y_pred_calib_bc.mean(),color='blue')
plt.axvline(te.Class.mean(),color='yellow')
plt.axvline(y_pred_bc.mean(),color='red')
plt.title('Prediction distribution and means')
plt.show()


