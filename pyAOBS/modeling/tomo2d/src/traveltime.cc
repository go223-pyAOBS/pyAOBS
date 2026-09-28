/*
 * traveltime.cc - travel time integration along a ray path
 *
 * Jun Korenaga, MIT/WHOI
 * January 1999
 */

#include "traveltime.h"
#include "interface.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fstream>

// Project spline samples into [z_lo, z_hi) so water-phase integrals never
// query the mixed cell below the seafloor (faster sediment attracts rays).
static void clip_water_col_points(Array1d<Point2d>& Q, int n,
				  const Interface2d* z_lo,
				  const Interface2d* z_hi)
{
    if (!z_lo || !z_hi) return;
    const double eps = 1e-6;
    for (int j=1; j<=n; j++){
	double x = Q(j).x();
	double z = Q(j).y();
	double lo = z_lo->z(x);
	double hi = z_hi->z(x);
	if (lo > hi){
	    double tmp = lo; lo = hi; hi = tmp;
	}
	if (z > hi - eps) z = hi - eps;
	if (z < lo) z = lo;
	Q(j) = Point2d(x, z);
    }
}

static Point2d clip_water_col_point(Point2d p,
				    const Interface2d* z_lo,
				    const Interface2d* z_hi)
{
    if (!z_lo || !z_hi) return p;
    Array1d<Point2d> one(1);
    one(1) = p;
    clip_water_col_points(one, 1, z_lo, z_hi);
    return one(1);
}

static int psx_path_seg(int spline_piece, int np)
{
    int seg = spline_piece - 2;
    if (seg < 1) seg = 1;
    if (seg > np - 1) seg = np - 1;
    return seg;
}

static bool psx_sample_is_s(const PsxBendSpec& spec, const Point2d& p, int seg)
{
    // Strictly below conv: below-leg phase (PSP/PSS → S, PPS → P).
    // On the interface, PPS cannot use below_is_s=P — whole-path refine
    // would hug the interface as P and undercut the OBS-side S lid.
    const bool has_c = spec.conv != 0;
    const double zc = has_c ? spec.conv->z(p.x()) : 0.0;
    if (has_c && p.y() > zc + 1e-3)
	return spec.below_is_s;
    if (!spec.lid_is_s)
	return has_c && p.y() >= zc - 1e-3 && spec.below_is_s;
    if (spec.below_is_s)
	return seg >= spec.iu1;
    // PPS: spline piece i maps to seg=i-2, so the OBS lid (nodes >= id1)
    // starts two pieces earlier than a raw seg>=id1 test would allow.
    return (seg + 2) >= spec.id1;
}

static double unmarked_at(const SlownessMesh2d& smesh, const Point2d& p,
			  Index2d& guess)
{
    // Dual: at() is Vs. Unmarked paths (PPP / water) must stay on Vp.
    if (smesh.inWater(p) || smesh.inAir(p) || !smesh.hasDualVs())
	return smesh.at(p, guess);
    return smesh.atVp(p, guess);
}

static double unmarked_at(const SlownessMesh2d& smesh, const Point2d& p,
			  Index2d& guess, double& dudx, double& dudz)
{
    if (smesh.inWater(p) || smesh.inAir(p) || !smesh.hasDualVs())
	return smesh.at(p, guess, dudx, dudz);
    return smesh.atVp(p, guess, dudx, dudz);
}

static bool psx_sample_below(const PsxBendSpec& spec, const Point2d& p, bool is_s)
{
    if (!spec.conv) return is_s;
    const bool geo_below = p.y() >= spec.conv->z(p.x()) + 1e-3;
    if (is_s){
	if (spec.lid_is_s && spec.below_is_s) return geo_below;
	return spec.below_is_s;
    }
    if (!spec.below_is_s && spec.lid_is_s) return geo_below;
    return false;
}

static double psx_at(const SlownessMesh2d& smesh, const Point2d& p,
		     Index2d& guess, const PsxBendSpec* psx, int seg)
{
    if (!psx || !psx->active)
	return unmarked_at(smesh, p, guess);
    if (smesh.inWater(p) || smesh.inAir(p))
	return smesh.at(p, guess);
    const bool is_s = psx_sample_is_s(*psx, p, seg);
	if (psx->conv){
	const bool sample_below = psx_sample_below(*psx, p, is_s);
	return smesh.atPsx(p, *psx->conv, is_s, sample_below, guess);
    }
    if (smesh.hasDualVs())
	return is_s ? smesh.at(p, guess) : smesh.atVp(p, guess);
    double u = smesh.at(p, guess);
    return is_s ? psx->kappa * u : u;
}

static double psx_at(const SlownessMesh2d& smesh, const Point2d& p,
		     Index2d& guess, double& dudx, double& dudz,
		     const PsxBendSpec* psx, int seg)
{
    if (!psx || !psx->active)
	return unmarked_at(smesh, p, guess, dudx, dudz);
    if (smesh.inWater(p) || smesh.inAir(p))
	return smesh.at(p, guess, dudx, dudz);
    const bool is_s = psx_sample_is_s(*psx, p, seg);
	if (psx->conv){
	const bool sample_below = psx_sample_below(*psx, p, is_s);
	return smesh.atPsx(p, *psx->conv, is_s, sample_below, guess, dudx, dudz);
    }
    if (smesh.hasDualVs())
	return is_s ? smesh.at(p, guess, dudx, dudz)
		    : smesh.atVp(p, guess, dudx, dudz);
    double u = smesh.at(p, guess, dudx, dudz);
    if (is_s){
	u *= psx->kappa;
	dudx *= psx->kappa;
	dudz *= psx->kappa;
    }
    return u;
}

static int psx_trial_nins(double span)
{
    if (span > 6.0) return 3;
    if (span > 2.0) return 2;
    return 1;
}

static double psx_trial_dip(double r, double span)
{
    double amp = 0.25 + 0.04 * span;
    if (amp > 1.2) amp = 1.2;
    const double pi = 3.141592653589793;
    return 0.15 + amp * std::sin(pi * r);
}

void ensurePsxBelowHinge(Array1d<Point2d>& path, int iu1, int& id0, int& id1,
			 const Interface2d& conv)
{
    if (iu1 < 1 || id0 <= iu1 || id0 > path.size()) return;
    double zmax_below = 0.0;
    for (int i=iu1; i<=id0; i++){
	double gap = path(i).y() - conv.z(path(i).x());
	if (gap > zmax_below) zmax_below = gap;
    }
    Point2d a = path(iu1), b = path(id0);
    double span = abs(b.x() - a.x());
    if (span < 0.4) return;
    if (zmax_below > 0.30 + 0.03 * span) return;
    int nins = psx_trial_nins(span);
    int np = path.size();
    Array1d<Point2d> tmp(np + nins);
    for (int i=1; i<=iu1; i++) tmp(i) = path(i);
    for (int k=1; k<=nins; k++){
	double r = double(k) / double(nins + 1);
	double x = (1.0-r)*a.x() + r*b.x();
	tmp(iu1+k) = Point2d(x, conv.z(x) + psx_trial_dip(r, span));
    }
    for (int i=id0; i<=np; i++) tmp(i+nins) = path(i);
    path.resize(np + nins);
    for (int i=1; i<=np+nins; i++) path(i) = tmp(i);
    id0 += nins;
    id1 += nins;
}

void seedPsxHingeFillets(Array1d<Point2d>& path, int iu1, int& id0, int& id1,
			 const Interface2d& conv)
{
    // PSS OBS-side: both legs are S, so a single interface pin is enough.
    // Three coincident pins force a beta-spline cusp (90°). Collapsing them
    // lets the spline round while the pin still slides on the conversion.
    (void)iu1;
    (void)conv;
    if (id1 <= id0) return;
    int drop = id1 - id0;
    int np = path.size();
    Array1d<Point2d> tmp(np - drop);
    for (int i=1; i<=id0; i++) tmp(i) = path(i);
    for (int i=id1+1; i<=np; i++) tmp(i - drop) = path(i);
    path.resize(np - drop);
    for (int i=1; i<=np-drop; i++) path(i) = tmp(i);
    id1 = id0;
}

bool psxBendPathExploded(const Array1d<Point2d>& graph,
			 const Array1d<Point2d>& bent,
			 double x_margin, double z_margin)
{
    if (graph.size() < 2 || bent.size() < 2) return false;
    double gx0 = 1e30, gx1 = -1e30, gz1 = -1e30;
    for (int i=1; i<=graph.size(); i++){
	double x = graph(i).x(), z = graph(i).y();
	if (x < gx0) gx0 = x;
	if (x > gx1) gx1 = x;
	if (z > gz1) gz1 = z;
    }
    for (int i=1; i<=bent.size(); i++){
	double x = bent(i).x(), z = bent(i).y();
	if (x < gx0 - x_margin || x > gx1 + x_margin) return true;
	if (z > gz1 + z_margin) return true;
	if (z < -0.5) return true;
    }
    return false;
}

void ensurePsxSLegBelow(Array1d<Point2d>& path, int iu1, const Interface2d& conv)
{
    int np = path.size();
    if (iu1 < 1 || iu1 >= np) return;
    double zmax_below = 0.0;
    for (int i=iu1; i<=np; i++){
	double gap = path(i).y() - conv.z(path(i).x());
	if (gap > zmax_below) zmax_below = gap;
    }
    Point2d a = path(iu1), b = path(np);
    double span = abs(b.x() - a.x());
    if (span < 1.0) return;
    if (zmax_below > 0.30 + 0.03 * span) return;
    int nins = psx_trial_nins(span);
    Array1d<Point2d> tmp(np + nins);
    for (int i=1; i<=iu1; i++) tmp(i) = path(i);
    for (int k=1; k<=nins; k++){
	double r = double(k) / double(nins + 1);
	double x = (1.0-r)*a.x() + r*b.x();
	tmp(iu1+k) = Point2d(x, conv.z(x) + psx_trial_dip(r, span));
    }
    for (int i=iu1+1; i<=np; i++) tmp(i+nins) = path(i);
    path.resize(np + nins);
    for (int i=1; i<=np+nins; i++) path(i) = tmp(i);
}

double calcTravelTime(const SlownessMesh2d& smesh,
		      const Array1d<Point2d>& path,
		      const BetaSpline2d& bs,
		      const Array1d<const Point2d*>& pp,
		      Array1d<Point2d>& Q,
		      const Interface2d* z_lo, const Interface2d* z_hi,
		      const PsxBendSpec* psx)
{
    int np = path.size();
    int nintp = bs.numIntp();

    Index2d guess_index = smesh.nodeIndex(smesh.nearest(*pp(1)));
    double ttime=0.0;
    for (int i=1; i<=np+1; i++){
	int j1=i;
	int j2=i+1;
	int j3=i+2;
	int j4=i+3;
	bs.interpolate(*pp(j1),*pp(j2),*pp(j3),*pp(j4),Q);
	clip_water_col_points(Q, nintp, z_lo, z_hi);
	const int seg = psx_path_seg(i, np);

	if (smesh.inWater(Q(nintp/2))){
	    // rectangular integration
	    for (int j=2; j<=nintp; j++){
		Point2d midp = 0.5*(Q(j-1)+Q(j));
		double u0 = psx_at(smesh, midp, guess_index, psx, seg);
		double dist = Q(j).distance(Q(j-1));
		ttime += u0*dist;
	    }		
	}else{
	    // trapezoidal integration along one segment
	    double u0 = psx_at(smesh, Q(1), guess_index, psx, seg);
	    for (int j=2; j<=nintp; j++){
		double u1 = psx_at(smesh, Q(j), guess_index, psx, seg);
		double dist= Q(j).distance(Q(j-1));
		ttime += 0.5*(u0+u1)*dist;
		u0 = u1;
	    }
	}
    }
    return ttime;
}

double calcTravelTime(const SlownessMesh2d& smesh,
		      const Array1d<Point2d>& path,
		      int ndiv)
{
    int np = path.size();
    Index2d guess_index = smesh.nodeIndex(smesh.nearest(path(1)));
    double dd = 1.0/ndiv;

    double ttime=0.0;
    for (int i=2; i<=np; i++){
	double dist= path(i).distance(path(i-1));

	Point2d midp = 0.5*(path(i-1)+path(i));
	if (smesh.inWater(midp)){
	    // rectangular integration
	    ttime += smesh.atWater()*dist;
	}else{
	    // trapezoidal integration
	    double ddist = dist*dd;
	    double u0 = unmarked_at(smesh, path(i-1), guess_index);
	    for (int j=1; j<=ndiv; j++){
		double ratio = j*dd;
		Point2d tmpp = (1-ratio)*path(i-1)+ratio*path(i);
		double u1 = unmarked_at(smesh, tmpp, guess_index);
		ttime += 0.5*(u0+u1)*ddist;
		u0 = u1;
	    }
	}
    }
    return ttime;
}

void calc_dTdV2(const SlownessMesh2d& smesh, const Array1d<Point2d>& path,
 		const BetaSpline2d& bs, Array1d<Point2d>& dTdV,
 		const Array1d<const Point2d*>& pp,
 		Array1d<Point2d>& Q, Array1d<Point2d>& dQdu)
 {
     int np = path.size();
     Array1d<Point2d> path2(np);
     for (int i=1; i<=np; i++) path2(i) = path(i);
     Array1d<const Point2d*> pp2;
     makeBSpoints(path2,pp2);

     double dx=0.0001;
     dTdV(1).set(0.0,0.0);  // end points are fixed
     dTdV(np).set(0.0,0.0);

     for (int i=2; i<np; i++){
 	double orig=path2(i).x();
 	path2(i).x(orig+dx);
 	double t1 = calcTravelTime(smesh,path2,bs,pp2,Q);
 	path2(i).x(orig-dx);
 	double t2 = calcTravelTime(smesh,path2,bs,pp2,Q);
 	dTdV(i).x((t1-t2)/(2*dx));
 	path2(i).x(orig);

 	orig=path2(i).y();
 	path2(i).y(orig+dx);
 	t1 = calcTravelTime(smesh,path2,bs,pp2,Q);
 	path2(i).y(orig-dx);
 	t2 = calcTravelTime(smesh,path2,bs,pp2,Q);
 	dTdV(i).y((t1-t2)/(2*dx));
 	path2(i).y(orig);
     }	
 }

void calc_dTdV3(const SlownessMesh2d& smesh, const Array1d<Point2d>& path,
 		const BetaSpline2d& bs, Array1d<Point2d>& dTdV,
 		const Array1d<const Point2d*>& pp,
 		Array1d<Point2d>& Q, Array1d<Point2d>& dQdu,
		const Array1d<int>& start_i,
		const Array1d<int>& end_i,
		const Array1d<const Interface2d*>& interf)
{
    int np = path.size();
    Array1d<Point2d> path2(np);
    for (int i=1; i<=np; i++) path2(i) = path(i);
    Array1d<const Point2d*> pp2;
    makeBSpoints(path2,pp2);

    double dx=0.0001;
    dTdV(1).set(0.0,0.0);  // end points are fixed
    dTdV(np).set(0.0,0.0);

    int a=1;
    for (int i=2; i<np; i++){
	if (i==start_i(a)){
	    double origx=path2(i).x();
	    double origy=path2(i).y();
	    for (int j=i; j<=end_i(a); j++){
		path2(j).x(origx+dx);
		path2(j).y(interf(a)->z(origx+dx));
	    }
	    double t1 = calcTravelTime(smesh,path2,bs,pp2,Q);
	    for (int j=i; j<=end_i(a); j++){
		path2(j).x(origx-dx);
		path2(j).y(interf(a)->z(origx-dx));
	    }
	    double t2 = calcTravelTime(smesh,path2,bs,pp2,Q);
	    double dtdv = (t1-t2)/(2*dx);
	    dtdv /= 3.0;
 	    cerr << "\n" << "dtdv3: t1=" << t1 << " t2=" << t2
 		 << " dtdv=" << dtdv << '\n';

	    for (int j=i; j<=end_i(a); j++){
		dTdV(j).x(dtdv);
		dTdV(j).y(0.0);
	    }
	    for (int j=i; j<=end_i(a); j++){
		path2(j).x(origx);
		path2(j).y(origy);
	    }
	    i = end_i(a);
	    a++;
	}else{
	    // move x
	    double orig=path2(i).x();
	    path2(i).x(orig+dx);
	    double t1 = calcTravelTime(smesh,path2,bs,pp2,Q);
	    path2(i).x(orig-dx);
	    double t2 = calcTravelTime(smesh,path2,bs,pp2,Q);
	    dTdV(i).x((t1-t2)/(2*dx));
	    path2(i).x(orig);

	    // move y
	    orig=path2(i).y();
	    path2(i).y(orig+dx);
	    t1 = calcTravelTime(smesh,path2,bs,pp2,Q);
	    path2(i).y(orig-dx);
	    t2 = calcTravelTime(smesh,path2,bs,pp2,Q);
	    dTdV(i).y((t1-t2)/(2*dx));
	    path2(i).y(orig);
	}
    }	
}

void calc_dTdV(const SlownessMesh2d& smesh, const Array1d<Point2d>& path,
	       const BetaSpline2d& bs, Array1d<Point2d>& dTdV,
	       const Array1d<const Point2d*>& pp,
	       Array1d<Point2d>& Q, Array1d<Point2d>& dQdu,
	       const Interface2d* z_lo, const Interface2d* z_hi,
	       const PsxBendSpec* psx)
{
    int np = path.size();
    int nintp = bs.numIntp();
    double du = 1.0/(nintp-1);
    
    dTdV(1).set(0.0,0.0); // end points are fixed
    dTdV(np).set(0.0,0.0);

    Index2d guess_index = smesh.nodeIndex(smesh.nearest(*pp(1)));
    for (int i=2; i<np; i++){
	double dTdVx=0.0, dTdVz=0.0;

	// loop over four segments affected by the point Vi
	int iseg_start = i-1;
	int iseg_end = i+2; // min(i+2,np-1);
	for (int j=iseg_start; j<=iseg_end; j++){
	    int j1=j;
	    int j2=j+1;
	    int j3=j+2;
	    int j4=j+3;
	    bs.interpolate(*pp(j1),*pp(j2),*pp(j3),*pp(j4),Q,dQdu);
	    clip_water_col_points(Q, nintp, z_lo, z_hi);
	    int i_j = i-j;
	    const int seg = psx_path_seg(j, np);

	    // trapezoidal integration along one segment
	    double dudx0, dudz0, dudx1, dudz1;
	    Point2d startp = 0.9*Q(1)+0.1*Q(2); // in order to avoid singularity
	    startp = clip_water_col_point(startp, z_lo, z_hi);
	    double u0 = psx_at(smesh, startp, guess_index, dudx0, dudz0, psx, seg);
	    double dQnorm0 = dQdu(1).norm();
	    double integx0, integz0;
	    if (dQnorm0==0.0){
		integx0 = integz0 = 0.0;
	    }else{
		double tmp1 = u0*bs.coeff_dbdu(i_j,1)/dQnorm0;
		double tmp2 = dQnorm0*bs.coeff_b(i_j,1);
		integx0 = tmp1*dQdu(1).x()+tmp2*dudx0;
		integz0 = tmp1*dQdu(1).y()+tmp2*dudz0;
	    }		

	    for (int m=2; m<=nintp; m++){
		double u1;
		if (m==nintp){
		    Point2d endp = 0.1*Q(m-1)+0.9*Q(m); // in order to avoid singularity
		    endp = clip_water_col_point(endp, z_lo, z_hi);
		    u1 = psx_at(smesh, endp, guess_index, dudx1, dudz1, psx, seg);
		}else{
		    u1 = psx_at(smesh, Q(m), guess_index, dudx1, dudz1, psx, seg);
		}
		double dQnorm1 = dQdu(m).norm();
		double integx1, integz1;
		if (dQnorm1==0.0){
		    integx1 = integz1 = 0.0;
		}else{
		    double tmp1 = u1*bs.coeff_dbdu(i_j,m)/dQnorm1;
		    double tmp2 = dQnorm1*bs.coeff_b(i_j,m);
		    integx1 = tmp1*dQdu(m).x()+tmp2*dudx1;
		    integz1 = tmp1*dQdu(m).y()+tmp2*dudz1;
		}

		dTdVx += 0.5*(integx0+integx1)*du;
		dTdVz += 0.5*(integz0+integz1)*du;
		
		integx0 = integx1;
		integz0 = integz1;
	    }
	}
	dTdV(i).set(dTdVx,dTdVz);
    }
}

double calcPsxGraphTime(const SlownessMesh2d& smesh, const Array1d<Point2d>& path,
			int ndiv, int iu1, int id0, int id1,
			double kappa, bool below_is_s, bool lid_is_s,
			const Interface2d* conv)
{
    int np = path.size();
    if (np<2) return 0.0;
    Index2d guess = smesh.nodeIndex(smesh.nearest(path(1)));
    double dd = 1.0/std::max(1, ndiv);
    double t = 0.0;
    const bool dual = smesh.hasDualVs();
    for (int i=1; i<np; i++){
	double zm = 0.5*(path(i).y()+path(i+1).y());
	double xm = 0.5*(path(i).x()+path(i+1).x());
	bool is_s;
	if (conv && dual && zm > conv->z(xm) + 1e-3)
	    is_s = below_is_s;
	else if (!lid_is_s)
	    is_s = (i >= iu1 && i < id0);
	else if (below_is_s)
	    is_s = (i >= iu1);
	else
	    is_s = (i >= id1);
	const Point2d& a = path(i);
	const Point2d& b = path(i+1);
	for (int k=0; k<ndiv; k++){
	    double r0 = k*dd, r1 = (k+1)*dd;
	    Point2d p0 = (1.0-r0)*a + r0*b;
	    Point2d p1 = (1.0-r1)*a + r1*b;
	    double u;
	    if (conv){
		bool sample_below;
		if (is_s)
		    sample_below = (lid_is_s && below_is_s)
			? (0.5*(p0.y()+p1.y()) >= conv->z(0.5*(p0.x()+p1.x())) + 1e-3)
			: below_is_s;
		else
		    sample_below = (!below_is_s && lid_is_s)
			? (0.5*(p0.y()+p1.y()) >= conv->z(0.5*(p0.x()+p1.x())) + 1e-3)
			: false;
		u = 0.5*(smesh.atPsx(p0, *conv, is_s, sample_below, guess)
			 +smesh.atPsx(p1, *conv, is_s, sample_below, guess));
		t += u*p0.distance(p1);
	    }else if (dual){
		u = is_s ? 0.5*(smesh.at(p0,guess)+smesh.at(p1,guess))
			 : 0.5*(smesh.atVp(p0,guess)+smesh.atVp(p1,guess));
		t += u*p0.distance(p1);
	    }else{
		u = 0.5*(smesh.at(p0,guess)+smesh.at(p1,guess));
		t += (is_s ? kappa : 1.0)*u*p0.distance(p1);
	    }
	}
    }
    return t;
}
