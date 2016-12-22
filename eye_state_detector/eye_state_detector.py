import cv2
import numpy as np
from data_manager import utils
import os
from sklearn.svm import SVC
from sklearn.datasets.base import Bunch
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score, KFold
from scipy.stats import sem
from sklearn import metrics

class EyeStateDetector:

    img_def_size = 24

    def load_train_data(self, src_dir):

        data_size = self.img_def_size * self.img_def_size
        eyes = Bunch(images=np.ndarray(shape=(0, self.img_def_size, self.img_def_size) , dtype=float),
                                       data=np.ndarray(shape=(0,data_size), dtype=float), target=np.ndarray(shape=(0), dtype=int))
        open_eyes_dir = src_dir + '/openEyesTraining'
        close_eyes_dir = src_dir + '/closedEyesTraining'

        img_number = 300
        self.load_eyes_from_dir(open_eyes_dir, number=img_number, open=True, eyes=eyes)
        self.load_eyes_from_dir(close_eyes_dir, number=img_number, open=False, eyes=eyes)

        # print(np.max(eyes.data))
        # print(np.min(eyes.data))
        # print(np.mean(eyes.data))

        # print(eyes)

        svc_1 = SVC(kernel='linear')
        print(svc_1)

        X_train, X_test, y_train, y_test = train_test_split(eyes.data, eyes.target, test_size=0.25, random_state=0)

        self.evaluate_cross_validation(svc_1, X_train, y_train, 5)

        self.train_and_evaluate(svc_1, X_train, X_test, y_train, y_test)


    def load_eyes_from_dir(self, src_dir, open, number, eyes):

        count = 0
        for filename in os.listdir(src_dir):
            # предполагается, что картинки уже в серых тонах
            img = cv2.imread(src_dir + '/' + filename)
            if img is None:
                print('error: no image')
                continue

            h, w = utils.get_img_size(img)
            if h != w:
                print('error: img not a square')
                continue

            if h != self.img_def_size:
                img = utils.resize_img_to(img, self.img_def_size)

            img_data = img[:, :, 0]
            img_data = np.array(img_data, dtype=float)
            utils.normalize_2d_data(img_data, 1.0)
            data = img_data.flatten()

            eyes.data = np.append(eyes.data, [data], axis=0)
            eyes.target = np.append(eyes.target, [1 if open else 0], axis=0)
            eyes.images = np.append(eyes.images, [img_data], axis=0)

            count += 1
            print('load image: ', count)
            if count > number:
                break

    def evaluate_cross_validation(self, clf, X, y, K):
        # create a k-fold croos validation iterator
        cv = KFold(len(y), K, shuffle=True, random_state=0)
        # by default the score used is the one returned by score method of the estimator (accuracy)
        scores = cross_val_score(clf, X, y, cv=cv)
        print(scores)
        print("Mean score: {0:.3f} (+/-{1:.3f})".format(np.mean(scores), sem(scores)))

    def train_and_evaluate(self, clf, X_train, X_test, y_train, y_test):
        clf.fit(X_train, y_train)

        print("Accuracy on training set:")
        print(clf.score(X_train, y_train))
        print("Accuracy on testing set:")
        print(clf.score(X_test, y_test))

        y_pred = clf.predict(X_test)

        print("Classification Report:")
        print(metrics.classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(metrics.confusion_matrix(y_test, y_pred))

    # eye_img - 24x24 картинка глаза в серых тонах
    # возвращает вероятность того, что глаз закрыт от 0 до 100
    # def get_eye_state(self, src_img, eye_img, eye_rect):

