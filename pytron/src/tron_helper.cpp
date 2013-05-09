#include "tron_helper.h"



double func_callback::fun(double *w)
{
	double out;
	c_func(w, f_py, &out, nr_variable);
	return out;
}

void func_callback::grad(double *w, double *g)
{
	c_grad(w, g);
}

void func_callback::Hv(double *s, double *Hs)
{
	c_hess(s, Hs);
}

int func_callback::get_nr_variable(void)
{
	return nr_variable;
}