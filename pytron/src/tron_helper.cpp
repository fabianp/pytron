#include "tron_helper.h"
#include <stdio.h>

double func_callback::fun(double *w)
{
	double t;
	c_func(w, py_func, &t, nr_variable);
	return t;
}

void func_callback::grad(double *w, double *g)
{
	c_grad(w, py_grad, g, nr_variable);
	this->w = w;
}

void func_callback::Hv(double *s, double *Hs)
{
	double *w = this->w;
	c_hess(s, w, py_hess, Hs, nr_variable);
}

int func_callback::get_nr_variable(void)
{
	return nr_variable;
}