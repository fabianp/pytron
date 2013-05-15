#include "tron.h"

typedef void (*func_cb)(double *, void *, double *, int);
typedef void (*grad_cb)(double *, void *, void **, double *, int);
typedef void (*hess_cb)(double *, void *, double *, int, void *);

class func_callback: public function {

public:
	func_callback(double *x0, void *py_func, func_cb c_func,
	void *py_grad_hess, grad_cb c_grad, hess_cb c_hess,
	int nr_variable, void *py_args) {
		this->w = new double[nr_variable];
		this->py_func = py_func;
		this->py_grad_hess = py_grad_hess;
		this->c_func = c_func;
		this->c_grad = c_grad;
		this->c_hess = c_hess;
		this->nr_variable = nr_variable;
		this->py_args = py_args;
	};

	~func_callback() {
		delete this->w;
	};
	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);
	int get_nr_variable(void);

protected:
	double tmp;
	double *w;
	func_cb c_func;
	grad_cb c_grad;
	hess_cb c_hess;
	void *py_func;
	void *py_grad_hess;
	void *py_hess;
	void *py_args;
	int nr_variable;
};

