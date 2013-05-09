#include "tron.h"

typedef void (*f_cb)(double *, void *, double *, int);
typedef void (*grad_cb)(double *, double *);
typedef void (*hess_cb)(double *, double *);

class func_callback: public function
{

public:
	func_callback(void *f_py, f_cb c_func, grad_cb c_grad, 
		hess_cb c_hess, int nr_variable);
	~func_callback();

	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);

	int get_nr_variable(void);
protected:
	f_cb c_func;
	void *f_py;
	grad_cb c_grad;
	hess_cb c_hess;
	int nr_variable;
};

func_callback::func_callback(void *f_py, f_cb c_func,
	grad_cb c_grad, hess_cb c_hess, int nr_variable)
{
	this->f_py = f_py;
	this->c_func = c_func;
	this->c_grad = c_grad;
	this->c_hess = c_hess;
	this->nr_variable = nr_variable;
}

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