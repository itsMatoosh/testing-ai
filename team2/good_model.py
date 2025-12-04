import sklearn

"""
Since we dont have actual data who commited fraud, only who has been checked. 
To get proper training data it would be good to filter out who was checked due to biases and who poses a real risk.

How can we do that?
- define set o sensitive variables, (adres, women and so on)
- train bad model on everything 
- permutate vulnerable variables 
- make new training set 

Why is it better then just permutating sensitive variables?

If a decision to check was made based on the vulnerable variables changing them does not change the fact that
the true label is checked, this is a problem since it leads to some datapoints being labeled true even if they
 should not be...

whatever just permutate, smaller model  
"""

