/*
 * lsqr.cc - LSQR implementation
 * 
 * Jun Korenaga, MIT/WHOI
 * January 1999
 */

#include <cmath>
#include "lsqr.h"
#include "stdlib.h"
#include <error.h>
#include <cstdlib>

// solve Ax = b
int iterativeSolver_LSQR(const SparseRectangular& A, const Array1d<double>& b,
			 Array1d<double>& x, double ATOL, int& itermax, double& chi)
{
    // check inputs
    int nnode = A.nCol();
    int ndata = A.nRow();
    if (nnode != x.size() || ndata != b.size()){
	cerr << "iterativeSolver_LSQR::size mismatch "
	     << nnode << " " << x.size() << ", "
	     << ndata << " " << b.size() << '\n';
	exit(1);
    }

    Array1d<double> v(nnode), w(nnode), u(b.size());
    Array1d<double> vtmp(nnode), col_scale(nnode), col_norm(nnode);
    for (int i=1; i<=b.size(); i++){
	u(i) = b(i);
    }

    bool enable_col_precond = true;
    {
	const char* legacy_env = getenv("TOMO2D_INV_LEGACY_BASELINE");
	if (legacy_env && atoi(legacy_env)!=0){
	    enable_col_precond = false;
	}
    }

    if (enable_col_precond){
	// diagonal column preconditioning: solve A*D*y=b, then x=D*y
	// where D(j)=1/max(||A(:,j)||_2, eps).
	const double eps_col = 1e-12;
	A.columnNorm2(col_norm);
	for (int j=1; j<=nnode; j++){
	    const double nrm = col_norm(j);
	    col_scale(j) = (nrm>eps_col) ? 1.0/nrm : 1.0;
	}
    }else{
	for (int j=1; j<=nnode; j++) col_scale(j) = 1.0;
    }
    
    // initialization
    double beta = normalize(u); 
    A.Atx(u,vtmp);
    for (int j=1; j<=nnode; j++) v(j) = col_scale(j)*vtmp(j);
    double alpha = normalize(v); 
    w = v;
    x = 0.0;
    double phibar = beta;
    double rhobar = alpha;

    double Bnorm = 0.0; // |B|
    double Dnorm = 0.0; // |D|

    int istop = 0;
    int iter = 0;

    while (istop == 0){
	iter++;
	if (iter%100==0) cerr << "LSQR iter= " << iter
			      << " nnode= " << nnode << "\r";
	
	u *= (-alpha); // update u
	for (int j=1; j<=nnode; j++) vtmp(j) = col_scale(j)*v(j);
	A.Ax(vtmp,u,false);
	beta = normalize(u); 
	v *= (-beta); // update v
	A.Atx(u,vtmp);
	for (int j=1; j<=nnode; j++) v(j) += col_scale(j)*vtmp(j);
	alpha = normalize(v);

	Bnorm += alpha*alpha+beta*beta; // |B|
	    
	double rho = sqrt(rhobar*rhobar+beta*beta);
	double c = rhobar/rho;
	double s = beta/rho;
	double theta = s*alpha;
	rhobar = -c*alpha;
	double phi = c*phibar;
	phibar = s*phibar;

	double phirho = phi/rho; // update x and w
	double thetarho = theta/rho;
	for (int i=1; i<=nnode; i++){
	    double tmp = w(i);
	    x(i) += phirho*tmp;
	    w(i) = v(i)-thetarho*tmp;

	    tmp /= rho;
	    Dnorm += tmp*tmp;
	}

	// check stoping criteria
	double rnorm = phibar; // |r|
	double Arnorm = phibar*alpha*abs(c); // |Atr|
	double Anorm = sqrt(Bnorm); // |A|
	double condA = Anorm*sqrt(Dnorm); // cond(A)
	double test2 = Arnorm/(Anorm*rnorm);
	double test3 = 1.0/condA;
	
	if (1.0+test2<=1.0) istop = 2;
	if (1.0+test3<=1.0) istop = 3;
	if (iter>itermax) istop = 4;

	if (test2 < ATOL) istop = 30;

	chi=rnorm;
    }

    cerr << "LSQR iter= " << iter
	 << " nnode= " << nnode << " ndata= " << ndata << "\n";
    for (int j=1; j<=nnode; j++) x(j) *= col_scale(j);
    itermax = iter;
    return istop;
}

double normalize(Array1d<double>& x)
{
    double norm=0;
    int n=x.size();
    for (int i=1; i<=n; i++){
	double val = x(i);
	norm += val*val;
    }
    norm = sqrt(norm);
    double rnorm = 1.0/norm;
    x *= rnorm;

    return norm;
}

