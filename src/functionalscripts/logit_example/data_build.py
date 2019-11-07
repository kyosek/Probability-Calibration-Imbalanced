# input data
df = pd.read_csv('src/resources/data/creditcard.csv')

# The percentage of positive class in this dataset
len(df[df['Class']==1])/len(df) # 0.0017

# Normalise the Amount feature
df['amt'] = preprocessing.normalize(np.array(df['Amount']).reshape(-1,1),norm='l2')

# Split the dataset into train and test, and drop unnecessary features
tr, te = train_test_split(df.drop(['Amount','Time'],1), test_size=.2,random_state=123)

# Check the distributions of positive class in both train and test dataset
len(tr[tr['Class']==1])/len(tr) # 0.0016546
len(te[te['Class']==1])/len(te) # 0.002018
