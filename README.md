This is week 2 of the weekly AI challenge

# Polynomial Regression with Regularisation

This is just a PoC.
I train a layer to basically evaluate a cubic model over the input vector (it's in one dim here coz i cant be bothered)

And then add the coeffiecients of degrees higher than 1 to the loss in order to combat overfitting.
But as you can guess, this may cause us to lose more optimal fits and underfit instead.
So I will just fiddle around till it works.