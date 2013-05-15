#include "tron_helper.h"
#include <string.h>
#include <stdio.h>

double func_callback::fun(double *w)
{
	double t;
	c_func(w, py_func, &t, nr_variable);
	return t;
}

void func_callback::grad(double *w, double *g)
{
	c_grad(w, py_grad_hess, &this->py_hess, g, nr_variable);
}

void func_callback::Hv(double *s, double *Hs)
{
	c_hess(s, this->py_hess, Hs, nr_variable, this->py_args);
}

int func_callback::get_nr_variable(void)
{
	return nr_variable;
}