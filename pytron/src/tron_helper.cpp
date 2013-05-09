#include "tron_helper.h"
#include <iostream>


double func_callback::fun(double *w)
{
	// I really don't know how to code properly in C++
	c_func(w, f_py, &tmp, nr_variable);
	std::cout << tmp << '\n';
	return tmp;
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