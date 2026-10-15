import numpy as np

from opytimark.markers import multi_objective


def test_ZDT1():
    f = multi_objective.ZDT1()

    x = np.array([0.37, 0.95, 0.73, 0.59, 0.15, 0.15, 0.05, 0.86, 0.60, 0.70, 0.02, 0.96, 0.83, 0.21, 0.18, 0.18, 0.30, 0.52, 0.43, 0.29, 0.61, 0.13, 0.29, 0.36, 0.45, 0.78, 0.19, 0.51, 0.59, 0.04])

    y = f(x)

    np.testing.assert_allclose(y, np.array([0.37, 3.57583653]))


def test_ZDT2():
    f = multi_objective.ZDT2()

    x = np.array([0.70, 0.29, 0.23, 0.55, 0.72, 0.42, 0.98, 0.68, 0.48, 0.39, 0.34, 0.73, 0.44, 0.06, 0.40, 0.74, 0.18, 0.18, 0.53, 0.53, 0.63, 0.85, 0.72, 0.61, 0.72, 0.32, 0.36, 0.23, 0.29, 0.63])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([0.7, 5.32573768]))


def test_ZDT3():
    f = multi_objective.ZDT3()

    x = np.array([0.78, 0.24, 0.82, 0.97, 0.97, 0.45, 0.61, 0.78, 0.64, 0.72, 0.04, 0.3, 0.06, 0.4, 0.3, 0.51, 0.17, 0.07, 0.03, 0.19, 0.95, 0.68, 0.86, 0.0, 0.3, 0.53, 0.67, 0.59, 0.62, 0.67])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([0.78, 3.79666173]))

def test_ZDT4():
    f = multi_objective.ZDT4()

    x = np.array([0.97, 0.47, 4.73, 2.15, 1.98, -2.84, 4.76, -4.94, -2.47, -0.65])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([0.97, 165.11131992]))


def test_ZDT6():
    f = multi_objective.ZDT6()

    x = np.array([0.97, 0.55, 0.97, 0.71, 0.70, 0.22, 0.98, 0.01, 0.25, 0.43])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([0.99951125, 8.58432404]))


def test_DTLZ1():
    f = multi_objective.DTLZ1()

    x = np.array([0.09, 0.97, 0.48, 0.24, 0.53, 0.29, 0.86])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([26.45895818, 0.81831829, 275.80357324]))


def test_DTLZ2():
    f = multi_objective.DTLZ2()

    x = np.array([0.63, 0.65, 0.62, 0.92, 0.07, 0.44, 0.83, 0.27, 0.17, 0.83, 0.53, 0.17])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([0.53606207, 0.8747738, 1.56187322]))


def test_DTLZ3():
    f = multi_objective.DTLZ3()

    x = np.array([0.15, 0.74, 0.26, 0.53, 0.01, 0.92, 0.9, 0.03, 0.96, 0.14, 0.28, 0.61])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([427.75088328, 988.47396909, 258.57848845]))


def test_DTLZ4():
    f = multi_objective.DTLZ4()

    x = np.array([0.44, 0.03, 0.55, 0.44, 0.42, 0.33, 0.2, 0.62, 0.3, 0.27, 0.62, 0.53])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([1.25400000e+000, 1.01517961e-152, 4.36199398e-036]))


def test_DTLZ5():
    f = multi_objective.DTLZ5()

    x = np.array([0.01, 0.50, 0.50, 0.13, 0.14, 0.22, 0.42, 0.25, 0.08, 0.35, 0.17, 0.88])

    y = f(x)
    
    np.testing.assert_allclose(np.round(y, 8), np.array([1.31929848, 1.31929848, 0.02930985]))


def test_DTLZ6():
    f = multi_objective.DTLZ6()

    x = np.array([0.38, 0.07, 0.28, 0.59, 0.55, 0.34, 0.34, 0.64, 0.83, 0.79, 0.77, 0.36])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([8.43596584, 1.49298056, 5.82216771]))


def test_DTLZ7():
    f = multi_objective.DTLZ7()

    x = np.array([0.44, 0.03, 0.55, 0.44, 0.42, 0.33, 0.2, 0.62, 0.3, 0.27, 0.62, 0.53, 0.13, 0.51, 0.18, 0.79, 0.85, 0.49, 0.85, 0.08, 0.51, 0.07])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([0.44, 0.03, 17.69213455]))


def test_DTLZ8():
    f = multi_objective.DTLZ8()

    x = np.array([0.99, 0.55, 0.28, 0.08, 0.44, 0.47, 0.05, 0.16, 0.98, 0.25, 0.39, 0.41, 0.9, 0.85, 0.42, 0.9, 0.38, 0.15, 0.77, 0.6, 0.78, 0.7, 0.29, 0.91, 0.21, 0.45, 0.93, 0.02, 0.6, 0.95])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([0.425, 0.577, 0.584]))


def test_WFG1():
    f = multi_objective.WFG1()

    x = np.array([0.75, 3.80, 4.39, 4.79, 1.56, 1.87, 0.81, 13.86, 10.82, 14.16, 0.23, 23.28, 20.65, 5.95, 5.45, 5.87, 10.39, 18.85, 16.70, 11.65, 25.63, 6.15, 13.49, 17.58])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([2.95134123, 0.97713161]))


def test_WFG2():
    f = multi_objective.WFG2()

    x = np.array([1.56, 0.95, 4.95, 7.73, 9.73, 5.44, 8.53, 12.41, 11.55, 14.44, 0.77, 7.16, 1.52, 24.00, 11.19, 21.76, 8.71, 12.51, 0.36, 14.33, 39.86, 9.59, 14.69, 44.05])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([1.76255531, 4.66109279]))


def test_WFG3():
    f = multi_objective.WFG3()

    x = np.array([0.87, 0.10, 3.30, 3.48, 8.41, 5.09, 5.73, 4.09, 9.31, 11.99, 6.59, 12.45, 11.21, 2.59, 15.42, 17.06, 14.48, 9.52, 15.94, 1.19, 21.56, 4.37, 19.54, 25.46])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([1.15395601, 2.98645601]))


def test_WFG4():
    f = multi_objective.WFG4()

    x = np.array([1.93, 2.19, 5.84, 5.72, 6.98, 2.59, 13.02, 0.50, 4.55, 8.70, 17.15, 21.17, 14.30, 27.13, 5.91, 23.93, 16.69, 5.62, 22.80, 39.25, 32.77, 15.76, 35.87, 19.78])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([1.8392815, 3.1138314]))


def test_WFG5():
    f = multi_objective.WFG5()

    x = np.array([0.59, 3.24, 2.10, 6.32, 5.61, 3.04, 1.47, 0.94, 12.12, 13.96, 16.12, 18.91, 1.99, 8.90, 22.23, 10.51, 15.98, 6.52, 12.29, 18.20, 22.24, 12.89, 12.74, 32.55])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([1.75640418, 3.91211285]))


def test_WFG6():
    f = multi_objective.WFG6()

    x = np.array([0.15, 3.12, 2.63, 5.79, 9.78, 6.46, 7.02, 1.15, 4.83, 10.00, 18.16, 12.19, 9.91, 1.01, 9.03, 29.14, 7.27, 16.14, 27.33, 3.00, 23.51, 38.74, 11.91, 26.89])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([2.41857116, 2.58039527]))


def test_WFG7():
    f = multi_objective.WFG7()

    x = np.array([1.98, 1.98, 1.69, 0.90, 2.35, 5.74, 0.76, 5.50, 8.16, 7.69, 17.84, 9.95, 11.86, 11.29, 9.62, 25.66, 10.36, 13.72, 21.99, 27.82, 16.16, 24.56, 8.43, 8.14])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([1.74747715, 3.03849476]))


def test_WFG8():
    f = multi_objective.WFG8()

    x = np.array([0.16, 0.41, 0.64, 1.97, 7.37, 4.05, 9.06, 11.43, 3.42, 15.40, 13.08, 21.31, 4.15, 25.76, 18.29, 17.35, 6.89, 27.67, 20.34, 17.77, 15.68, 39.13, 19.68, 31.69])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([0.91577966, 4.41026463]))


def test_WFG9():
    f = multi_objective.WFG9()

    x = np.array([1.03, 1.14, 0.34, 2.74, 6.85, 5.00, 5.97, 4.36, 5.56, 7.28, 19.42, 16.53, 4.74, 21.70, 7.31, 24.63, 4.95, 24.93, 7.23, 17.79, 20.39, 15.39, 6.46, 15.50])

    y = f(x)
    
    np.testing.assert_allclose(y, np.array([2.27553429, 2.06986023]))