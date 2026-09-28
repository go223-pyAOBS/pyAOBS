/*
 * lsqr.cc - LSQR implementation
 *
 * Jun Korenaga, MIT/WHOI
 * January 1999
 */

#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include "lsqr.h"
#include "stdlib.h"
#include <error.h>
#include <cstdlib>

void lsqrColPrecondConfig(bool& enable, double& kappa, int& miniter)
{
    enable = false;
    kappa = 10.0;
    miniter = 20;
    const char* legacy_env = getenv("TOMO2D_INV_LEGACY_BASELINE");
    if (legacy_env && atoi(legacy_env)!=0){
	enable = false;
    }else{
	const char* p = getenv("TOMO2D_INV_LSQR_PRECOND");
	if (p && atoi(p)!=0){
	    enable = true;
	}
    }
    const char* mx = getenv("TOMO2D_INV_LSQR_PRECOND_MAX");
    if (mx && *mx){
	const double v = atof(mx);
	kappa = (v>0.0) ? v : 0.0; // 0 = 相对中位数不封顶（空列仍 D=0）
    }
    const char* mi = getenv("TOMO2D_INV_LSQR_PRECOND_MINITER");
    if (mi && *mi){
	const int v = atoi(mi);
	if (v>=0) miniter = v;
    }
}

static double median_positive(const Array1d<double>& nrm)
{
    std::vector<double> pos;
    pos.reserve(size_t(nrm.size()));
    for (int j=1; j<=nrm.size(); j++){
	if (std::isfinite(nrm(j)) && nrm(j)>0.0) pos.push_back(nrm(j));
    }
    if (pos.empty()) return 0.0;
    std::nth_element(pos.begin(), pos.begin()+pos.size()/2, pos.end());
    return pos[pos.size()/2];
}

static void fill_col_scale(const Array1d<double>& col_norm, double kappa,
			   Array1d<double>& col_scale, int& nzero, int& nclip)
{
    nzero = nclip = 0;
    const double med = median_positive(col_norm);
    const double eps_abs = 1e-12;
    const double eps_rel = (med>0.0) ? 1e-8*med : eps_abs;
    for (int j=1; j<=col_norm.size(); j++){
	const double nrm = col_norm(j);
	if (!(nrm>eps_abs) || nrm<eps_rel || med<=0.0){
	    col_scale(j) = 0.0;
	    nzero++;
	    continue;
	}
	double s = med/nrm;
	if (kappa>0.0){
	    const double lo = 1.0/kappa;
	    if (s<lo){ s = lo; nclip++; }
	    else if (s>kappa){ s = kappa; nclip++; }
	}
	col_scale(j) = s;
    }
}

// solve Ax = b
int iterativeSolver_LSQR(const SparseRectangular& A, const Array1d<double>& b,
			 Array1d<double>& x, double ATOL, int& itermax, double& chi,
			 bool allow_precond)
{
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

    bool enable_col_precond = false;
    double kappa = 10.0;
    int miniter = 20;
    lsqrColPrecondConfig(enable_col_precond, kappa, miniter);
    if (!allow_precond) enable_col_precond = false;

    int nzero = 0, nclip = 0;
    if (enable_col_precond){
	// Jacobi：A D y = b, x = D y。D_j = clip(s_med/‖A_j‖, 1/κ, κ)；
	// 空列 D=0，不再用 D=1 放大暗结点。
	A.columnNorm2(col_norm);
	fill_col_scale(col_norm, kappa, col_scale, nzero, nclip);
    }else{
	for (int j=1; j<=nnode; j++) col_scale(j) = 1.0;
	miniter = 0;
    }

    double beta = normalize(u);
    const double bnorm = beta;
    A.Atx(u,vtmp);
    for (int j=1; j<=nnode; j++) v(j) = col_scale(j)*vtmp(j);
    double alpha = normalize(v);
    w = v;
    x = 0.0;
    double phibar = beta;
    double rhobar = alpha;

    double Bnorm = 0.0;
    double Dnorm = 0.0;

    int istop = 0;
    int iter = 0;
    double test2 = 0.0;
    double Anorm = 0.0;
    double rnorm = phibar;
    const int miniter_use = enable_col_precond ? miniter : 0;

    while (istop == 0){
	iter++;
	if (iter%100==0) cerr << "LSQR iter= " << iter
			      << " nnode= " << nnode << "\r";

	u *= (-alpha);
	for (int j=1; j<=nnode; j++) vtmp(j) = col_scale(j)*v(j);
	A.Ax(vtmp,u,false);
	beta = normalize(u);
	if (beta <= 0.0){ istop = 1; break; }
	v *= (-beta);
	A.Atx(u,vtmp);
	for (int j=1; j<=nnode; j++) v(j) += col_scale(j)*vtmp(j);
	alpha = normalize(v);
	if (alpha <= 0.0){ istop = 1; break; }

	Bnorm += alpha*alpha+beta*beta;

	double rho = sqrt(rhobar*rhobar+beta*beta);
	double c = rhobar/rho;
	double s = beta/rho;
	double theta = s*alpha;
	rhobar = -c*alpha;
	double phi = c*phibar;
	phibar = s*phibar;

	double phirho = phi/rho;
	double thetarho = theta/rho;
	for (int i=1; i<=nnode; i++){
	    double tmp = w(i);
	    x(i) += phirho*tmp;
	    w(i) = v(i)-thetarho*tmp;

	    tmp /= rho;
	    Dnorm += tmp*tmp;
	}

	rnorm = phibar;
	double Arnorm = phibar*alpha*abs(c);
	Anorm = sqrt(Bnorm);
	double condA = Anorm*sqrt(Dnorm);
	test2 = (Anorm>0.0 && rnorm>0.0) ? Arnorm/(Anorm*rnorm) : 0.0;
	double test3 = 1.0/condA;

	if (1.0+test2<=1.0) istop = 2;
	if (1.0+test3<=1.0) istop = 3;
	if (iter>itermax) istop = 4;

	// 预条件后 test2 对 AD 过松，iter=1 就会 ATOL 停死。最少走完 miniter。
	if (iter>=miniter_use && test2 < ATOL) istop = 30;
	if (enable_col_precond && iter>=miniter_use && bnorm>0.0
	    && rnorm <= 1e-8*bnorm) istop = 31;
	if (!std::isfinite(alpha) || !std::isfinite(beta) || !std::isfinite(phibar))
	    istop = 5;

	chi=rnorm;
    }

    cerr << "LSQR iter= " << iter
	 << " nnode= " << nnode << " ndata= " << ndata
	 << " istop=" << istop
	 << " test2=" << test2
	 << " rnorm=" << rnorm
	 << " Anorm=" << Anorm;
    if (enable_col_precond){
	cerr << " precond=1 kappa=" << kappa
	     << " miniter=" << miniter_use
	     << " nzero=" << nzero
	     << " nclip=" << nclip;
    }else{
	cerr << " precond=0";
    }
    cerr << "\n";
    for (int j=1; j<=nnode; j++){
	x(j) *= col_scale(j);
	if (!std::isfinite(x(j))) x(j) = 0.0;
    }
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
    if (norm <= 0.0) return 0.0;
    double rnorm = 1.0/norm;
    x *= rnorm;

    return norm;
}
