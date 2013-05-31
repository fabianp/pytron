#include "tron.h"

typedef double (*func_cb)(double *, void *, int, void *);
typedef int (*grad_cb)(double *, void *, void **, double *, int, void *);
typedef int (*hess_cb)(double *, void *, double *, int, void *);

class func_callback: public function {

public:
	func_callback(double *x0, void *py_func, func_cb c_func,
	void *py_grad_hess, grad_cb c_grad, hess_cb c_hess,
	void *py_callback, func_cb c_callback, int nr_variable, void *py_args) {
		this->w = new double[nr_variable];
		this->py_func = py_func;
		this->py_grad_hess = py_grad_hess;
		this->py_callback = py_callback;
		this->c_func = c_func;
		this->c_grad = c_grad;
		this->c_hess = c_hess;
		this->c_callback = c_callback;
		this->nr_variable = nr_variable;
		this->py_args = py_args;
	};

	~func_callback() {
		delete this->w;
	};
	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);
	void callback(double *w);
	int get_nr_variable(void);

protected:
	double tmp;
	double *w;
	func_cb c_func;
	grad_cb c_grad;
	hess_cb c_hess;
	func_cb c_callback;
	void *py_func;
	void *py_grad_hess;
	void *py_hess;
	void *py_args;
	void *py_callback;
	int nr_variable;
};

