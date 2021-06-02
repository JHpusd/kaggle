import math

coefficients = [1, 1]

def function(coeffs, x):
    a = coeffs[0]
    b = coeffs[1]
    return a*math.sin(b*x)

data = [[0.5,2], [0.9,0.9], [1,0.3]]

def rss(coeffs, function):
    x_vals = [row[0] for row in data]
    y_vals = [row[1] for row in data]
    rss = 0
    for i in range(len(x_vals)):
        prediction = function(coeffs, x_vals[i])
        rss += (y_vals[i] - prediction)**2
    return rss

def calc_gradient(coeffs, function, delta):
    gradients = []
    for i in range(len(coeffs)):
        coeffs_1 = list(coeffs)
        coeffs_2 = list(coeffs)
        coeffs_1[i] += 0.5*delta
        coeffs_2[i] -= 0.5*delta
        derivative = (rss(coeffs_1,function)-rss(coeffs_2,function))/delta
        gradients.append(derivative)
    return gradients

def gradient_descent(coeffs, function, alpha, delta, num_steps, debug_mode=False):
    for i in range(num_steps):
        gradients = calc_gradient(coeffs, function, delta)
        if debug_mode:
            print("Step {}:".format(i))
            print("\tGradient: "+str(gradients))
            print("\tCoeffs: "+str(coeffs))
            print("\tRSS: "+str(rss(coeffs, function))+"\n")
        for n in range(len(gradients)):
            coeffs[n] -= gradients[n] * alpha

gradient_descent(coefficients, function, 0.1, 0.1, 100, True)
