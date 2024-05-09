cdef double born_logpeak_model(
   double x,
    double born_height,
    double born_width,
    double logpeak_height,
    double logpeak_activate,
    double logpeak_gradient,
    double logpeak_decay
)

cdef double scipy_kramerskronig_integrand(int n, double *xx, void *user_data)

cdef double scipy_cauchy_integrand(int n, double *xx, void *user_data)
