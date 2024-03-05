cpdef double collision_activate_decay(
    double x,
    double lorentzian_height,
    double lorentzian_powerlaw,
    double logistic_activate,
    double logistic_gradient
)

cdef double coll_act_decay_scipy(double x, void *user_data)

cdef double coll_act_decay_kramerskronig_scipy(int n, double *xx, void *user_data)
