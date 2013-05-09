#include "tron.h"

typedef void (*tron_cb)(double *, void *, double *, int);

class func_callback: public function {

public:
	func_callback(void *py_func, tron_cb c_func, 
	void *py_grad, tron_cb c_grad, 
	void *py_hess, tron_cb c_hess, 
	int nr_variable) {
		this->py_func = py_func;
		this->py_grad = py_grad;
		this->py_hess = py_hess;
		this->c_func = c_func;
		this->c_grad = c_grad;
		this->c_hess = c_hess;
		this->nr_variable = nr_variable;
	};

	~func_callback() {};
	double fun(double *w);
	void grad(double *w, double *g);
	void Hv(double *s, double *Hs);
	int get_nr_variable(void);

protected:
	double tmp;
	tron_cb c_func;
	tron_cb c_grad;
	tron_cb c_hess;
	void *py_func;
	void *py_grad;
	void *py_hess;
	int nr_variable;
};

