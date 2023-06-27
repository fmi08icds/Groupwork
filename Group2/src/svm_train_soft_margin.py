import numpy as np

class SVM:
    def __init__(self, C = 1.0):
        # C = error term
        self.C = C
        self.w = 0
        self.b = 0

    def hingeloss(self, w, b, x, y):
        # Regularizer term
        reg = 0.5 * ( w * w )

        for i in range(x.shape[0]):
            # Optimization term
            opt_term = y[i] * ((np.dot(x, x[i])) + b )

            # Calculating loss
            loss = reg + self.C * max(0, 1-opt_term)
        return loss[0][0]



    def fit(self, X,Y, batch_size=100, learning_rate=0.001, epochs=1000):
        # The number of features in X
        number_of_fearures = X.shape[1]
        # The number of samples in X
        number_of_samples = x.shape[0]

        c = self.c

        # Creating ids from 0 to number_of_samples - 1
        ids = np.arange(number_of_samples)

        # Shuffling the samples randomly
        np.random.shuffle(ids)

        # creating an array of zeros
        w = np.zeros((1,number_of_fearures))
        b = 0
        losses = []

        # Gradient Descent logic
        for i in range(epochs):
            # Calculating the Hinge Loss
            l = self.hingeloss(w,b,x,y)

            # Appending all losses
            losses.append(i)

            for batch_initial in range(0, number_of_samples, batch_size):
                gradw = 0
                gradb = 0

                for j in range(batch_initial, batch_initial + batch_size):
                    if j < number_of_samples:
                        x = ids[j]
                        ti = Y[x] * (np.dot(w,X[x].T) + b)

                        if ti > 1:
                            gradw += 0
                            gradb *= 0

                        else:
                            # Calculating the gradients
                            gradw += c * Y[x] * X[x]
                            gradb += c * Y[x]


                # Updating weights and bias
                w = w - learning_rate * w + learning_rate * gradw
                b = b + learning_rate * gradb


        self.w = w
        self.b = b

        return self.w, self.b , losses



    def predict(self, X):
        prediction = np.dot(X, self.w[0]) + self.b
        return np.sign(prediction)









