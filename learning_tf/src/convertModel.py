from sklearn import externals, svm, datasets
from sklearn.base import clone

# import some data to play with
iris = datasets.load_iris()
# Take the first two features. We could avoid this by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target

# we create an instance of SVM and fit out data. We do not scale our
# data since we want to plot the support vectors
C = 1.0  # SVM regularization parameter
newSVCM = svm.SVC(kernel='linear', C=C)
newSVCM.fit(X, y)


clf_svc = externals.joblib.load('/home/rubik/catkin_ws/src/isr_activity_recognition/learning_tf/src/SVC_torso_camera/svm_clf.pkl')
clf_svc.gamma = 'auto'
clf_svc._dual_coef_ = clf_svc.dual_coef_

print 'stop'

