#!/usr/bin/env python  

import rospy
from std_msgs.msg import String
import tf
import numpy as np
import hcluster as mat
import os.path
import time
import scipy.io as sio
from scipy import linalg as lina
from sklearn import externals

conta_class = np.zeros(8)
activities = ['Walking', 'Standing still', 'Working on computer', 'Talking on the phone', 'Running', 'Jumping',
               'Falling', 'Sitting down']  # Activities


# Callback for reaction of the recognized activity "Falling"
def callback(data):
    rospy.loginfo(rospy.get_caller_id() + " I heard %s", data.data)

    if data.data == "yes" or data.data == "yes please" or data.data == "help" or data.data == "help me" or data.data == "please":
        os.system("rosrun sound_play say.py 'calling a doctor'")

    # Function that listens the answer of the person to a question


def resposta():
    # rospy.init_node('classifica', anonymous=True)

    rospy.Subscriber('recognizer/output', String, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()


# compute the weights trough entorpy - uncertainty measure to find
# the cnfidence level of a classifer
# input:  classifiers n x m matrix: n = probabilities of the m_th classifier type
# output: weights 1xm
def conf_class(classifiers):
    offset = 0.9
    w = np.array([])
    for j in range(0, classifiers.shape[1]):
        # avoiding extremes
        classifiers[classifiers <= 0.0001] = 0.001
        classifiers[classifiers > 0.989] = 0.99
        logar = np.log10(classifiers[:, j])
        prob = classifiers[:, j]
        h = prob * logar
        w = np.hstack((w, -np.sum(h))) if w.size else -np.sum(h)

    summ = sum(w)

    for j in range(0, w.size):
        w[j] = (w[j] / summ)

    # avoiding extremes
    w[w > 0.998] = offset
    w[w <= 0.0001] = 1 - offset

    summ = sum(w)
    for j in range(0, w.size):
        w[j] = w[j] / summ

    return w


def weight_update(prior_weights, post3rd):
    # computing the weights distribution inversely proportional to the entropy given the 3rd order Markov property
    #       i.e., given the posteriors: P(C_x|C_t-i), i=1,2,3, x=t, t-1, t-2
    w_t = conf_class(post3rd)

    # convert the current vector of weights into a diagonal matrix for multiplication with the prior
    D_wt = np.diag(w_t)
    weights = np.diag(prior_weights) * D_wt
    summ = np.sum(np.diag(weights))
    weights = np.diag(weights) / summ  # normalization / distribution

    return weights


# adjust the convergence, avoid the class reaching 1 or 0 and keeping it, allowing state transitions in framexframe
# classification
def adjust_prob_conv(dbmmcl, num_cl):
    yn = 0
    j = 0
    val = 0.65
    for c in range(0, num_cl):
        if dbmmcl[j, c] > val:
            yn = 1
            # resta = 1 - (dbmmcl(j, c)+(0.3/num_cl));
            resta = 1 - val
            r = resta / (num_cl - 1)
            dbmmcl[j, :] = r
            dbmmcl[j, c] = val
        elif dbmmcl[j, c] < 0.01:
            yn = 1
            dbmmcl[j, c] = 0.01

        if yn:
            sumc = np.sum(dbmmcl[j, :])
            dbmmcl[j, :] = dbmmcl[j, :] / sumc

    return dbmmcl


# DBMM

def dbmm(class_nb, class_svm, weights, labels, w_update, frame_update):
    # number of frames
    frames = class_nb.shape[0]
    # number of classifiers
    num_cl = 2
    size_class = class_nb.shape[1]

    # for weights update to control the short-time memory (5rd order Markov property)
    short_time_wup = 0

    # initially, uniform prior
    dbmmcl = np.zeros((frames, size_class))

    prior = 1.0 / size_class

    first_result_nb = weights[0] * class_nb[0, :]
    first_result_svm = weights[1] * class_svm[0, :]

    dbmmcl[0, :] = first_result_nb + first_result_svm
    dbmm_nop = dbmmcl
    dbmmcl[0, :] = dbmmcl[0, :] * prior
    dbmmcl[0, :] = dbmmcl[0, :] / np.sum(dbmmcl[0, :])

    # DBMM - fusion
    for j in range(1, frames):

        # prepare/get data (memory of te system) for weights
        # update computation
        if short_time_wup == frame_update:
            post3rd = dict(cl=[np.array([]), np.array([]), np.array([])])
            # get the previous posteriors from each classifier
            for k in range(0, short_time_wup):
                post3rd['cl'][0] = np.vstack((post3rd['cl'][0], class_nb[j - k, :])) if post3rd['cl'][
                    0].size else class_nb[j - k, :]
                post3rd['cl'][1] = np.vstack((post3rd['cl'][1], class_svm[j - k, :])) if post3rd['cl'][
                    1].size else class_svm[j - k, :]

            # verify the column that allocates the maximum a posteriori from each base classifier
            col = np.zeros((1, 8))
            col = col.astype(int)
            for c in range(0, num_cl):
                s = sum(post3rd['cl'][c])
                m = max(s)
                indC = [i for i, v in enumerate(s) if v == m]
                col[0, c] = indC[0]

            p3rd = np.array([])
            for c in range(0, num_cl):
                p3rd = np.c_[p3rd, post3rd['cl'][c][:, col[0, c]]] if p3rd.size else post3rd['cl'][c][:, col[0, c]]

            if w_update == 1:
                weights = weight_update(weights, p3rd)
            short_time_wup = 0

            sumw = np.sum(weights)
            weights = weights / sumw
            weights[weights > 0.9] = 0.9
            weights[weights < 0.1] = 0.1

        # DBMM weighted sum
        w_classif_nb = weights[0] * class_nb[j, :]
        w_classif_svm = weights[1] * class_svm[j, :]

        pond = w_classif_nb + w_classif_svm
        dbmm_nop = pond

        # transition probability using prior P(C_t | C_t-1) - dynamic probabilistic loop - 1st order Markov property
        dbmmcl[j, :] = dbmmcl[j - 1, :] * pond
        # normalization
        sc = np.sum(dbmmcl[j, :])
        dbmmcl[j, :] = dbmmcl[j, :] / sc
        dbmmcl[j, :] = dbmmcl[j, :]

        short_time_wup = short_time_wup + 1

        dbmm_cl = adjust_prob_conv(np.array([dbmmcl[j, :]]), dbmmcl.shape[1])
        dbmmcl[j, :] = dbmm_cl

    return [dbmmcl, dbmm_nop]


# Moving Average Filter
def moving_average(a, n):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


# Log-Cov Function
def apply_log_vect(M):
    d = M.shape[0]
    n = d * (d + 1) / 2
    V = np.zeros((n, 1))
    offset = 0.001 * np.eye(d, d)
    true_mat = np.ones((d, d))
    true_mat = true_mat.astype(np.int64)
    in_triu = np.triu(true_mat)
    logM = np.real(lina.logm(M + offset))
    V = logM[in_triu == 1]
    V = np.array([V])
    return V


# Features Extraction Function
def feature_extraction_torso_camera(input_torso, input_camera):
    numero_juntas = 15  # number of joints
    frame_rate = 1 / 30.0  # frame rate
    window = 10  # temporal window

    x = input_torso[:, 0::6]
    y = input_torso[:, 1::6]
    z = input_torso[:, 2::6]

    # Guarantees that the number of frames is the same for torso and camera features
    if input_torso.size < input_camera.size:
        [m, n] = input_torso.shape
    else:
        [m, n] = input_camera.shape

    # Log-Cov of distances between every joints relative to the torso

    distancias = np.zeros((numero_juntas, numero_juntas))
    distancias_total = np.array([[]])

    for frame in range(0, m):
        for i in range(0, 15):
            for j in range(0, 15):
                distancias[i, j] = mat.pdist(
                    [[x[frame, i], y[frame, i], z[frame, i]], [x[frame, j], y[frame, j], z[frame, j]]])

        distlower = np.tril(distancias)
        distupper = np.triu(distancias)
        distancias_final = distlower[1:, :] + distupper[0:-1, :]  # elimination of null diagonal

        cov_distancias = np.cov(distancias_final.T)

        aux = apply_log_vect(cov_distancias)

        distancias_total = np.concatenate([distancias_total, aux]) if distancias_total.size else aux

    # Distances between every joints and torso
    distancias = np.zeros((m, numero_juntas))

    for frame in range(0, m):
        for i in range(0, 15):
            distancias[frame, i] = mat.pdist(
                [[x[frame, i], y[frame, i], z[frame, i]], [x[frame, 3], y[frame, 3], z[frame, 3]]])

    distancias_ao_torso = distancias

    # Absolute velocities

    velocidades = np.zeros((m, numero_juntas))

    for frame in range(0, m):

        if frame == 0:
            anterior = frame
        else:
            anterior = frame - 1

        actual = frame

        for i in range(0, 15):
            velocidades[frame, i] = (mat.pdist(
                [[x[actual, i], y[actual, i], z[actual, i]], [x[anterior, i], y[anterior, i], z[anterior, i]]])) / (
                                        frame_rate)

    velocidades_total = velocidades

    # Velocities and directions for each dimension {x,y,z}

    vx = np.zeros((m, numero_juntas))
    vy = np.zeros((m, numero_juntas))
    vz = np.zeros((m, numero_juntas))
    dx = np.zeros((m, numero_juntas))
    dy = np.zeros((m, numero_juntas))
    dz = np.zeros((m, numero_juntas))

    for frame in range(0, m):

        if frame == 0:
            anterior = frame
        else:
            anterior = frame - 1

        actual = frame

        for i in range(0, 15):
            dx[frame, i] = x[actual, i] - x[anterior, i]
            dy[frame, i] = y[actual, i] - y[anterior, i]
            dz[frame, i] = z[actual, i] - z[anterior, i]
            vx[frame, i] = dx[frame, i] / frame_rate
            vy[frame, i] = dy[frame, i] / frame_rate
            vz[frame, i] = dz[frame, i] / frame_rate

    velocidade_xyz = np.c_[vx, vy, vz]
    direcao_xyz = np.c_[dx, dy, dz]

    # Angles of the triangles formed by {shoulders, elbows, hands}, {shoulders, hips, knees} and {hips, knees, feet}
    angulos = np.array([])

    for frame in range(0, m):
        shoulder_left_elbow_left = mat.pdist([[x[frame, 4 - 1], y[frame, 4 - 1], z[frame, 4 - 1]],
                                              [x[frame, 5 - 1], y[frame, 5 - 1],
                                               z[frame, 5 - 1]]])  # distance between left shoulder and left elbow
        shoulder_left_hand_left = mat.pdist([[x[frame, 4 - 1], y[frame, 4 - 1], z[frame, 4 - 1]],
                                             [x[frame, 12 - 1], y[frame, 12 - 1],
                                              z[frame, 12 - 1]]])  # distance between left shoulder and left hand
        hand_left_elbow_left = mat.pdist([[x[frame, 12 - 1], y[frame, 12 - 1], z[frame, 12 - 1]],
                                          [x[frame, 5 - 1], y[frame, 5 - 1],
                                           z[frame, 5 - 1]]])  # distance between left hand and left elbow
        cosine_left1 = (shoulder_left_elbow_left ** 2 + hand_left_elbow_left ** 2 - shoulder_left_hand_left ** 2)/(2 * shoulder_left_elbow_left * hand_left_elbow_left)
        angulo_left1 = np.arccos(np.clip(cosine_left1, -1, 1))

        shoulder_right_elbow_right = mat.pdist([[x[frame, 6 - 1], y[frame, 6 - 1], z[frame, 6 - 1]],
                                                [x[frame, 7 - 1], y[frame, 7 - 1],
                                                 z[frame, 7 - 1]]])  # distance between right shoulder and right elbow
        shoulder_right_hand_right = mat.pdist([[x[frame, 6 - 1], y[frame, 6 - 1], z[frame, 6 - 1]],
                                               [x[frame, 13 - 1], y[frame, 13 - 1],
                                                z[frame, 13 - 1]]])  # distance between right shoulder and right hand
        hand_right_elbow_right = mat.pdist([[x[frame, 13 - 1], y[frame, 13 - 1], z[frame, 13 - 1]],
                                            [x[frame, 7 - 1], y[frame, 7 - 1],
                                             z[frame, 7 - 1]]])  # distance between right hand and right elbow
        cosine_right1 = (shoulder_right_elbow_right ** 2 + hand_right_elbow_right ** 2 - shoulder_right_hand_right ** 2) / (2 * shoulder_right_elbow_right * hand_right_elbow_right)
        angulo_right1 = np.arccos(np.clip(cosine_right1, -1, 1))

        shoulder_left_hip_left = mat.pdist([[x[frame, 4 - 1], y[frame, 4 - 1], z[frame, 4 - 1]],
                                            [x[frame, 8 - 1], y[frame, 8 - 1],
                                             z[frame, 8 - 1]]])  # distance between left shoulder and left hip
        shoulder_left_knee_left = mat.pdist([[x[frame, 4 - 1], y[frame, 4 - 1], z[frame, 4 - 1]],
                                             [x[frame, 9 - 1], y[frame, 9 - 1],
                                              z[frame, 9 - 1]]])  # distance between left shoulder and left knee
        hip_left_knee_left = mat.pdist([[x[frame, 8 - 1], y[frame, 8 - 1], z[frame, 8 - 1]],
                                        [x[frame, 9 - 1], y[frame, 9 - 1],
                                         z[frame, 9 - 1]]])  # distance between left hip and left knee
        cosine_left2 = (shoulder_left_hip_left ** 2 + hip_left_knee_left ** 2 - shoulder_left_knee_left ** 2) / (2 * shoulder_left_hip_left * hip_left_knee_left)
        angulo_left2 = np.arccos(np.clip(cosine_left2, -1, 1))

        shoulder_right_hip_right = mat.pdist([[x[frame, 6 - 1], y[frame, 6 - 1], z[frame, 6 - 1]],
                                              [x[frame, 10 - 1], y[frame, 10 - 1],
                                               z[frame, 10 - 1]]])  # distance between right shoulder and right hip
        shoulder_right_knee_right = mat.pdist([[x[frame, 6 - 1], y[frame, 6 - 1], z[frame, 6 - 1]],
                                               [x[frame, 11 - 1], y[frame, 11 - 1],
                                                z[frame, 11 - 1]]])  # distance between right shoulder and right knee
        hip_right_knee_right = mat.pdist([[x[frame, 10 - 1], y[frame, 10 - 1], z[frame, 10 - 1]],
                                          [x[frame, 11 - 1], y[frame, 11 - 1],
                                           z[frame, 11 - 1]]])  # distance between right hip and right knee
        cosine_right2 = (shoulder_right_hip_right ** 2 + hip_right_knee_right ** 2 - shoulder_right_knee_right ** 2) / (2 * shoulder_right_hip_right * hip_right_knee_right)  # angle
        angulo_right2 = np.arccos(np.clip(cosine_right2, -1, 1))

        foot_left_hip_left = mat.pdist([[x[frame, 14 - 1], y[frame, 14 - 1], z[frame, 14 - 1]],
                                        [x[frame, 8 - 1], y[frame, 8 - 1],
                                         z[frame, 8 - 1]]])  # distance between left foot and left hip
        foot_left_knee_left = mat.pdist([[x[frame, 14 - 1], y[frame, 14 - 1], z[frame, 14 - 1]],
                                         [x[frame, 9 - 1], y[frame, 9 - 1],
                                          z[frame, 9 - 1]]])  # distance between left foot and left knee
        hip_left_knee_left = mat.pdist([[x[frame, 8 - 1], y[frame, 8 - 1], z[frame, 8 - 1]],
                                        [x[frame, 9 - 1], y[frame, 9 - 1],
                                         z[frame, 9 - 1]]])  # distance between left hip and left knee
        cosine_left3 = (foot_left_knee_left ** 2 + hip_left_knee_left ** 2 - foot_left_hip_left ** 2) / (2 * foot_left_knee_left * hip_left_knee_left)
        angulo_left3 = np.arccos(np.clip(cosine_left3, -1, 1))

        foot_right_hip_right = mat.pdist([[x[frame, 15 - 1], y[frame, 15 - 1], z[frame, 15 - 1]],
                                          [x[frame, 10 - 1], y[frame, 10 - 1],
                                           z[frame, 10 - 1]]])  # distance between right foot and right hip
        foot_right_knee_right = mat.pdist([[x[frame, 15 - 1], y[frame, 15 - 1], z[frame, 15 - 1]],
                                           [x[frame, 11 - 1], y[frame, 11 - 1],
                                            z[frame, 11 - 1]]])  # distance between right foot and right knee
        hip_right_knee_right = mat.pdist([[x[frame, 10 - 1], y[frame, 10 - 1], z[frame, 10 - 1]],
                                          [x[frame, 11 - 1], y[frame, 11 - 1],
                                           z[frame, 11 - 1]]])  # distance between right hip and right knee
        cosine_right3 = (foot_right_knee_right ** 2 + hip_right_knee_right ** 2 - foot_right_hip_right ** 2) / (2 * foot_right_knee_right * hip_right_knee_right)
        angulo_right3 = np.arccos(np.clip(cosine_right3, -1, 1))

        an = np.c_[angulo_left1, angulo_right1, angulo_left2, angulo_right2, angulo_left3, angulo_right3]
        angulos = np.r_[angulos, an] if angulos.size else an

    # Angular Difference

    variacao_angulos = np.array([[]])
    for frame in range(0, m):
        if frame == 0:
            anterior = frame
        else:
            anterior = frame - 1

        actual = frame

        dif = np.array([angulos[actual, :] - angulos[anterior, :]])
        variacao_angulos = np.r_[variacao_angulos, dif] if variacao_angulos.size else dif

        # Variation of all joints relative to the camera in {x,y,z}
    x_camera = input_camera[:, 0::6]
    y_camera = input_camera[:, 1::6]
    z_camera = input_camera[:, 2::6]
    dx_camera = np.zeros((m, numero_juntas))
    dy_camera = np.zeros((m, numero_juntas))
    dz_camera = np.zeros((m, numero_juntas))
    vx_camera = np.zeros((m, numero_juntas))
    vy_camera = np.zeros((m, numero_juntas))
    vz_camera = np.zeros((m, numero_juntas))

    for frame in range(0, m):
        if frame == 0:
            anterior = frame
        else:
            anterior = frame - 1

        actual = frame

        for i in range(0, 15):
            dx_camera[frame, i] = x_camera[actual, i] - x_camera[anterior, i]
            dy_camera[frame, i] = y_camera[actual, i] - y_camera[anterior, i]
            dz_camera[frame, i] = z_camera[actual, i] - z_camera[anterior, i]
            vx_camera[frame, i] = dx_camera[frame, i] / frame_rate
            vy_camera[frame, i] = dy_camera[frame, i] / frame_rate
            vz_camera[frame, i] = dz_camera[frame, i] / frame_rate

    variacao_xyz_camera = np.c_[dx_camera, dy_camera, dz_camera]
    velocidade_xyz_camera = np.c_[vx_camera, vy_camera, vz_camera]

    # Absolute velocities relative to the camera

    velocidades = np.zeros((m, numero_juntas))

    for frame in range(0, m):

        if frame == 0:
            anterior = frame
        else:
            anterior = frame - 1

        actual = frame

        for i in range(0, 15):
            velocidades[frame, i] = (mat.pdist([[x_camera[actual, i], y_camera[actual, i], z_camera[actual, i]],
                                                [x_camera[anterior, i], y_camera[anterior, i],
                                                 z_camera[anterior, i]]])) / (frame_rate)

    velocidades_total_camera = velocidades

    return [distancias_total, distancias_ao_torso, velocidades_total, velocidade_xyz, direcao_xyz, angulos,
            variacao_angulos, variacao_xyz_camera, velocidade_xyz_camera, velocidades_total_camera]


def fusao_DBMM(matriz_proba_NB, matriz_proba_SVC):
    [m, n] = np.shape(matriz_proba_NB)
    resultado = np.zeros((m, n))

    for i in range(0, len(matriz_proba_NB)):
        if i == 0:
            resultado[i, :] = matriz_proba_NB[i, :] * 0.6 + matriz_proba_SVC[i, :] * 0.4
        else:
            resultado[i, :] = (resultado[i - 1, :] * (matriz_proba_NB[i, :] * 0.6 + matriz_proba_SVC[i, :] * 0.4))
            resultado[i, :] = resultado[i, :] / np.sum(resultado[i, :])

    return resultado


# END OF FUNCTIONS

rospy.init_node('classifica')

# Load file with the training data
treino = sio.loadmat('treino_torso_camera.mat')
treino = treino['treino']
maxval = treino.max(0)
minval = treino.min(0)

MAIN_FRAME = 'torso_'

listener = tf.TransformListener()

rate = rospy.Rate(30.0)
dados_torso = np.array([])
dados_camera = np.array([])
M = np.array([])
segundos = 0

while not rospy.is_shutdown():
    counter = 0
    frame_list = listener.getFrameStrings()  # tf frame list

    for frame in frame_list:  # Count the number of users detected by the openni_tracker
        if MAIN_FRAME in frame:
            counter += 1

    while not os.path.exists('test_torso.txt'):
        print "Waiting..."
        time.sleep(3)

    if os.path.isfile('test_torso.txt'):
        # read file
        # time.sleep(2)
        segundos += 3
        with open("test_torso.txt") as f:
            num_lines = sum(1 for line in open("test_torso.txt"))
            print num_lines
            while not num_lines >= 150:
                time.sleep(0.001)
                num_lines = sum(1 for line in open("test_torso.txt"))
            dados_torso = np.loadtxt(f)
        with open("test_camera.txt") as f_c:
            num_lines = sum(1 for line in open("test_camera.txt"))
            dados_camera = np.loadtxt(f_c)
        print np.shape(dados_torso)
    else:
        raise ValueError("not a file!")

        # Features Extraction
    start = time.time()
    [distancias_total, distancias_ao_torso, velocidades_total, velocidade_xyz, direcao_xyz, angulos, variacao_angulos,
     variacao_xyz_camera, velocidade_xyz_camera, velocidades_total_camera] = feature_extraction_torso_camera(
        dados_torso[0::10, :], dados_camera[0::10, :])

    test = np.c_[
        distancias_total, velocidades_total, velocidade_xyz_camera, velocidades_total_camera, angulos, variacao_angulos]

    midstep1 = test != 0
    midstep2 = midstep1.sum(axis=0)
    midstep3 = midstep2 > 0

    test = test[:, (test != 0).sum(axis=0) > 0]

    [a, b] = test.shape

    # Normalization

    test = (test - minval) / (maxval - minval)

    print np.shape(test)

    # NAIVE BAYES
    clfNB = externals.joblib.load(
        '/home/rubik/catkin_ws/src/isr_activity_recognition/learning_tf/src/NaiveBayes_torso_camera/NB_clf.pkl')
    predicao_NB = clfNB.predict(test)
    predicao_NB = np.ndarray.tolist(predicao_NB)
    proba_predict_NB = clfNB.predict_proba(test)
    print "Naive Bayes:"
    print proba_predict_NB.mean(0)

    # SVC
    clf_svc = externals.joblib.load(
        '/home/rubik/catkin_ws/src/isr_activity_recognition/learning_tf/src/SVC_torso_camera/svm_clf.pkl')
    clf_svc.gamma = 'auto'
    clf_svc._dual_coef_ = clf_svc.dual_coef_
    predicao_svc = clf_svc.predict(test)
    predicao_svc = np.ndarray.tolist(predicao_svc)
    proba_predict_svc = clf_svc.predict_proba(test)
    print "SVM:"
    print proba_predict_svc.mean(0)

    weights = np.array([0.7, 0.3])
    proba_predict = 0
    for i in range(1, 9):
        c = predicao_NB.count(i)
        conta_class[i - 1] = c

    [proba_final, dbmm_nop] = dbmm(proba_predict_NB, proba_predict_svc, weights, 0, 1, 15)  # DBMM
    indexes = np.argmax(proba_final, 1)

    counts = np.bincount(indexes)
    most_freq = np.argmax(counts)
    prob_correcto = float(np.count_nonzero(indexes == most_freq)) / indexes.size
    proba_final = proba_final.mean(0)
    proba_final = np.array([proba_final])

    print "Fusion:"
    print proba_final
    print indexes
    print most_freq
    print prob_correcto
    end = time.time()
    print end - start  # Computation time

    if segundos > 0:

        proba_final = np.ndarray.tolist(proba_final.mean(axis=0))
        print "Activity: %s" % activities[proba_final.index(max(proba_final))]
        conta_class = np.zeros(8)
        if proba_final.index(max(proba_final)) == 6:  # If falling
            os.system("rosrun sound_play say.py 'do you need help?'")
            # os.system("roslaunch pocketsphinx robocup.launch")
            resposta()
        if proba_final.index(max(proba_final)) == 4:  # Se running
            os.system("rosrun sound_play say.py 'it is not allowed to run in this room'")
        if proba_final.index(max(proba_final)) == 5:  # Se saltar
            os.system("rosrun sound_play say.py 'it is not allowed to jump in this room'")

        os.remove("test_torso.txt")
        os.remove("test_camera.txt")
        segundos = 0
        time.sleep(1)

    rate.sleep()
