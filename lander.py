import numpy
import numpy as np

print("lander model")

class GlobalVar:
    g = 8.15     # gravity acceleration[m / s2]
    t = 1        # the simulation time step[s]
    ml = 700     # own mass of the lander[kg]

    tmax = 200   # the time in which the engine can work with full power[s]
    mp0 = 1000   # initial amount of fuel[kg]
    Fmax = 15000 # maximum force[N]
    k = mp0 / (Fmax * tmax)
    ko = 1 / k


# lander model returns new state based on previous state, force, and
# global variables (phisical params)
def lander_model(state,F,var_global):
    g = var_global.g
    t = var_global.t
    ml = var_global.ml
    k = var_global.k
    ko = var_global.ko
    Fmax = var_global.Fmax

    mp=state[0]
    x=state[1]
    v=state[2]

    mc=ml+mp  # own mass + fuel mass
    if F<0:
        F=0
    if F>Fmax:
        F=Fmax

    mp1=mp-k*F*t
    if mp1<0:
        F=mp/(k*t)
        mp1=0

    if F>0.1:
        v1=ko*np.log(mc/(mc-k*F*t))+v-g*t
        x1=-(ko*ko/F)*np.log(mc/(mc-k*F*t))*(mc-k*F*t)+t/k+v*t-(g*t*t)/2+x
    else:
        v1=v-g*t
        x1=v*t-(g*t*t)/2+x


    # determination of the exact moment of impact and speed at the moment.

    if x1<0:
        x1=0
        i=1
        tpom = np.linspace(0, t, num=np.floor(t * 100))
        if F>0.1:
            while -(ko*ko/F)*np.log(mc/(mc-k*F*tpom[i]))*(mc-k*F*tpom[i])+tpom[i]/k+v*tpom[i]-(g*tpom[i]*tpom[i])/2+x>0:
                i=i+1
            v1=ko*np.log(mc/(mc-k*F*tpom[i]))+v-g*tpom[i]
        else:
            while v*tpom[i]-(g*tpom[i]*tpom[i])/2+x>0:
                i=i+1
            v1=v-g*tpom[i]

    state_next = [mp1, x1, v1]
    return state_next