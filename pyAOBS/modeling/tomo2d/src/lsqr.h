/*
 * lsqr.h - LSQR definition
 *
 * Jun Korenaga, MIT/WHOI
 * January 1999
 */

#ifndef _TOMO_LSQR_H_
#define _TOMO_LSQR_H_

#include <array.h>
#include "sparse_rect.h"

int iterativeSolver_LSQR(const SparseRectangular& A, const Array1d<double>& b,
			 Array1d<double>& x, double ATOL, int& itermax, double& chi,
			 bool allow_precond=true);
double normalize(Array1d<double>& x);

// 默认关。TOMO2D_INV_LSQR_PRECOND≠0 且未开 Legacy → enable=true。
// kappa：相对列范数中位数的夹逼上限，默认 10；TOMO2D_INV_LSQR_PRECOND_MAX<=0 不夹逼。
// miniter：预条件开启后 test2 最早可停的迭代，默认 20。
void lsqrColPrecondConfig(bool& enable, double& kappa, int& miniter);

#endif /* _TOMO_LSQR_H_ */
