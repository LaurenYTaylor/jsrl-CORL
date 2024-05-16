import numpy as np

guide_H = 4
min_R = 10
guide_R = -4
mu = .75

def h_term_sum(h):
    coeffs = []
    while h>0:
        if h==guide_H:
            coeffs.append(-(min_R+guide_H))
        else:
            coeffs.append(0)
        h -= 1
    coeffs.append(-((-min_R-mu*(-min_R-guide_R))-min_R))
    return coeffs

if __name__ =="__main__":
    coeffs = list(reversed(h_term_sum(guide_H)))
    p = np.polynomial.polynomial.Polynomial(coeffs)
    print(coeffs)
    print(p)
    for c in p.roots():
        if np.isreal(c):
            print(c)

    print((((.75*-4)-10)/(-4-10))**(1/4))

