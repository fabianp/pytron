#include "tron_helper.h"
#include <string.h>
#include <stdio.h>

double func_callback::fun(double *w)
{
	double t;
	t = c_func(w, py_func, this->nr_variable, this->py_args);
	return t;
}

void func_callback::grad(double *w, double *g)
{
	c_grad(w, py_grad_hess, &this->py_hess, g, this->nr_variable,
	this->py_args);
}

void func_callback::Hv(double *s, double *Hs)
{
	c_hess(s, this->py_hess, Hs, this->nr_variable, this->py_args);
}

int func_callback::get_nr_variable(void)
{
	return nr_variable;
}

void func_callback::callback(double *w)
{
    if (this->py_callback != NULL)
        c_callback(w, this->py_callback, this->nr_variable, this->py_args);
}