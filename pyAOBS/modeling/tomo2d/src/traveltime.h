/*
 * traveltime.h - traveltime related helper functions
 *
 * Jun Korenaga, MIT/WHOI
 * January 1999
 */

#ifndef _TOMO_TRAVELTIME_H_
#define _TOMO_TRAVELTIME_H_

#include <array.h> // from mconv
#include <geom.h>
#include "smesh.h"
#include "betaspline.h"

class Interface2d;

// PPS/PSS: lid S after id1; PSS also S below conv. PSP: lid_is_s=false,
// S on/below conv (converse: interface belongs to below).
struct PsxBendSpec {
    bool active;
    int iu1, id0, id1;
    double kappa;
    bool below_is_s;
    bool lid_is_s;
    const Interface2d* conv;
    PsxBendSpec()
	: active(false), iu1(0), id0(0), id1(0), kappa(1.0),
	  below_is_s(false), lid_is_s(true), conv(0) {}
};

void ensurePsxBelowHinge(Array1d<Point2d>& path, int iu1, int& id0, int& id1,
			 const Interface2d& conv);
void seedPsxHingeFillets(Array1d<Point2d>& path, int iu1, int& id0, int& id1,
			 const Interface2d& conv);
void ensurePsxSLegBelow(Array1d<Point2d>& path, int iu1, const Interface2d& conv);
bool psxBendPathExploded(const Array1d<Point2d>& graph,
			 const Array1d<Point2d>& bent,
			 double x_margin=3.0, double z_margin=3.0);

double calcTravelTime(const SlownessMesh2d& m, const Array1d<Point2d>& path,
		      const BetaSpline2d& bs,
		      const Array1d<const Point2d*>& pp,
		      Array1d<Point2d>& Q,
		      const Interface2d* z_lo=0, const Interface2d* z_hi=0,
		      const PsxBendSpec* psx=0);
double calcTravelTime(const SlownessMesh2d& m,
		      const Array1d<Point2d>& path, int ndiv);
double calcPsxGraphTime(const SlownessMesh2d& m, const Array1d<Point2d>& path,
			int ndiv, int iu1, int id0, int id1,
			double kappa, bool below_is_s, bool lid_is_s=true,
			const Interface2d* conv=0);
//double calcTravelTime2(const SlownessMesh2d& m, const Array1d<Point2d>& path);
void calc_dTdV(const SlownessMesh2d& m, const Array1d<Point2d>& path,
	       const BetaSpline2d& bs, Array1d<Point2d>& dTdV,
	       const Array1d<const Point2d*>& pp,
	       Array1d<Point2d>& Q, Array1d<Point2d>& dQdu,
	       const Interface2d* z_lo=0, const Interface2d* z_hi=0,
	       const PsxBendSpec* psx=0);
void calc_dTdV2(const SlownessMesh2d& m, const Array1d<Point2d>& path,
		const BetaSpline2d& bs, Array1d<Point2d>& dTdV,
		const Array1d<const Point2d*>& pp,
		Array1d<Point2d>& Q, Array1d<Point2d>& dQdu);
void calc_dTdV3(const SlownessMesh2d& smesh, const Array1d<Point2d>& path,
 		const BetaSpline2d& bs, Array1d<Point2d>& dTdV,
 		const Array1d<const Point2d*>& pp,
 		Array1d<Point2d>& Q, Array1d<Point2d>& dQdu,
		const Array1d<int>& start_i,
		const Array1d<int>& end_i,
		const Array1d<const Interface2d*>& interf);
#endif /* _TOMO_TRAVELTIME_H_ */
