import numpy as np




def rk4(f, t_span, intial_value):
    step_size = 0.01
    t0, t1 = t_span
    num_steps = int((t1-t0)/step_size)
    t = np.linspace(t0, t1, num_steps)
    
    values = np.zeros((num_steps,len(intial_value)))
    values[0] = np.asarray(intial_value)
    
    for i in range(len(t)-1):
        values[i+1] = rk4_step(step_size, f, t[i], values[i])
    return values



