/*
 * bend.cc - ray-bending solver implementation
 *           direct minimization of travel times by conjugate gradient
 *           (rays are represented as beta-splines)
 *
 * Jun Korenaga, MIT/WHOI
 * January 1999
 */

#include "bend.h"
#include "traveltime.h"
#include "graph.h"
#include <cmath>

BendingSolver2d::BendingSolver2d(const SlownessMesh2d& s,
				 const BetaSpline2d& b,
				 double tol1, double tol2)
    : smesh(s), bs(b), nintp(bs.numIntp()),
      cg_tol(tol1), brent_tol(tol2), eps(1e-10),
      clip_lo(0), clip_hi(0),
      psx_active(false), psx_iu1(0), psx_id0(0), psx_id1(0),
      psx_kappa(1.0), psx_below_is_s(false), psx_lid_is_s(true), psx_conv(0),
      psx_moho(0),
      psx_leg_mode(false)
{
    Q.resize(nintp);
    dQdu.resize(nintp);

    int max_np = int(2*sqrt(float(smesh.numNodes())));
    new_point.reserve(max_np);
    dTdV.reserve(max_np);
    new_dTdV.reserve(max_np);
    direc.reserve(max_np);

    pp.reserve(max_np+4);
    new_pp.reserve(max_np+4);
}

void BendingSolver2d::setWaterColClip(const Interface2d* z_lo, const Interface2d* z_hi)
{
    clip_lo = z_lo;
    clip_hi = z_hi;
}

void BendingSolver2d::setPsxBend(int iu1, int id0, int id1,
				 double kappa, bool below_is_s,
				 const Interface2d* conv, bool lid_is_s)
{
    psx_active = true;
    psx_iu1 = iu1;
    psx_id0 = id0;
    psx_id1 = id1;
    psx_kappa = kappa;
    psx_below_is_s = below_is_s;
    psx_lid_is_s = lid_is_s;
    psx_conv = conv;
}

void BendingSolver2d::setPsxMoho(const Interface2d* moho)
{
    psx_moho = moho;
}

void BendingSolver2d::clearPsxBend()
{
    psx_active = false;
    psx_conv = 0;
    psx_moho = 0;
}

const PsxBendSpec* BendingSolver2d::psxArg(PsxBendSpec& spec) const
{
    if (!psx_active) return 0;
    spec.active = true;
    spec.iu1 = psx_iu1;
    spec.id0 = psx_id0;
    spec.id1 = psx_id1;
    spec.kappa = psx_kappa;
    spec.below_is_s = psx_below_is_s;
    spec.lid_is_s = psx_lid_is_s;
    spec.conv = psx_conv;
    return &spec;
}

void BendingSolver2d::clip_if_water(Array1d<Point2d>& p) const
{
    if (clip_lo && clip_hi)
	clampPathToWaterCol(p, *clip_lo, *clip_hi);
}

void BendingSolver2d::clamp_psx_below(Array1d<Point2d>& p) const
{
    // S-leg interior stays strictly below conv (type 6/8/12/13).
    // Type 12/13 Moho is type-1 style: hinge pinned on -F, legs free.
    // Sub-reflector water comes from -A, not a z=zm snap (that causes sliding).
    if (!psx_active || !psx_conv) return;
    if (!psx_below_is_s) return;
    if (psx_id0 <= psx_iu1) return;
    int i0 = psx_iu1, i1 = psx_id0;
    if (i0 < 1) i0 = 1;
    if (i1 > (int)p.size()) i1 = (int)p.size();
    if (i1 - i0 < 2) return;
    double xL = p(i0).x(), xR = p(i1).x();
    if (xL > xR){
	double tmp = xL; xL = xR; xR = tmp;
    }
    double xpad = 0.15;
    for (int i=i0+1; i<i1; i++){
	double x = p(i).x();
	if (psx_leg_mode){
	    if (x < xL - xpad) x = xL - xpad;
	    if (x > xR + xpad) x = xR + xpad;
	}
	double zc = psx_conv->z(x);
	double z = p(i).y();
	if (z < zc + 1e-3) z = zc + 1e-3;
	p(i) = Point2d(x, z);
    }
}

int BendingSolver2d::refine(Array1d<Point2d>& path,
			    double& orig_time, double& new_time)
{
    int np=check_path_size(path);
    clip_if_water(path);
    clamp_psx_below(path);
    makeBSpoints(path,pp);
    new_point.resize(np); 
    makeBSpoints(new_point,new_pp);

    dTdV.resize(np);
    new_dTdV.resize(np);
    direc.resize(np);
    PsxBendSpec spec;
    const PsxBendSpec* px = psxArg(spec);
    
    orig_time=calcTravelTime(smesh, path, bs, pp, Q, clip_lo, clip_hi, px);
    double old_time = orig_time;

    calc_dTdV(smesh, path, bs, dTdV, pp, Q, dQdu, clip_lo, clip_hi, px);
    direc = -dTdV; // p0
    // conjugate gradient search
    int iter_max = np*10;
    for (int iter=1; iter<=iter_max; iter++){
	new_time=line_min(path, direc);
	if ((old_time-new_time)<=cg_tol){
	    return iter;
	}
	
	calc_dTdV(smesh, path, bs, new_dTdV, pp, Q, dQdu, clip_lo, clip_hi, px);
	double gg=0.0, dgg=0.0;
	for (int i=1; i<=np; i++){
	    double x0=dTdV(i).x(),y0=dTdV(i).y();
	    double x1=new_dTdV(i).x(),y1=new_dTdV(i).y();
	    gg += x0*x0+y0*y0;
	    dgg += x1*x1+y1*y1;
	}
//	if (gg==0.0){
	if (abs(gg)<1e-10){
	    return iter;
	}else{
	    dgg /= gg;
	}
	for (int i=1; i<=np; i++){
	    direc(i) = -new_dTdV(i)+dgg*direc(i);
	}
	dTdV = new_dTdV;
	old_time = new_time;
    }
    return -iter_max; // too many iterations
}

int BendingSolver2d::refine(Array1d<Point2d>& path,
			    double& orig_time, double& new_time,
			    const Array1d<int>& start_i,
			    const Array1d<int>& end_i,
			    const Array1d<const Interface2d*>& interf)
{
    int np=check_path_size(path,start_i,end_i,interf);
    clip_if_water(path);
    clamp_psx_below(path);

    makeBSpoints(path,pp);
    new_point.resize(np); 
    makeBSpoints(new_point,new_pp);
    
    dTdV.resize(np);
    new_dTdV.resize(np);
    direc.resize(np);
    PsxBendSpec spec;
    const PsxBendSpec* px = psxArg(spec);

    orig_time=calcTravelTime(smesh, path, bs, pp, Q, clip_lo, clip_hi, px);
    double old_time = orig_time;

    calc_dTdV(smesh, path, bs, dTdV, pp, Q, dQdu, clip_lo, clip_hi, px);
    adjust_dTdV(dTdV, path, start_i, end_i, interf);
    for (int k=1; k<=(int)freeze_idx.size(); k++){
	int ii = freeze_idx(k);
	if (ii>=1 && ii<=np) dTdV(ii).set(0.0,0.0);
    }
    direc = -dTdV; // p0
    
    // conjugate gradient search
    int iter_max = np*10;
    for (int iter=1; iter<=iter_max; iter++){
	new_time=line_min(path, direc, start_i, end_i, interf);
	if ((old_time-new_time)<=cg_tol){
	    return iter;
	}
	
	calc_dTdV(smesh, path, bs, new_dTdV, pp, Q, dQdu, clip_lo, clip_hi, px);
	adjust_dTdV(new_dTdV, path, start_i, end_i, interf);
	for (int k=1; k<=(int)freeze_idx.size(); k++){
	    int ii = freeze_idx(k);
	    if (ii>=1 && ii<=np) new_dTdV(ii).set(0.0,0.0);
	}
	double gg=0.0, dgg=0.0;
	for (int i=1; i<=np; i++){
	    double x0=dTdV(i).x(),y0=dTdV(i).y();
	    double x1=new_dTdV(i).x(),y1=new_dTdV(i).y();
	    gg += x0*x0+y0*y0;
	    dgg += x1*x1+y1*y1;
	}
//	if (gg==0.0){ // 2000.5.8 
                      // this may be the cause for "-NaN" reported by
                      // Allegra. (due to division by extremely small gg)
	if (abs(gg)<1e-10){
	    return iter;
	}else{
	    dgg /= gg;
	}
	for (int i=1; i<=np; i++){
	    direc(i) = -new_dTdV(i)+dgg*direc(i);
	}
	dTdV = new_dTdV;
	old_time = new_time;
    }
    return -iter_max; // too many iterations
}

void BendingSolver2d::adjust_dTdV(Array1d<Point2d>& dTdV,
				  const Array1d<Point2d>& path,
				  const Array1d<int>& start_i,
				  const Array1d<int>& end_i,
				  const Array1d<const Interface2d*>& interf)
{
    for (int a=1; a<=interf.size(); a++){
	double slope = interf(a)->dzdx(path(start_i(a)).x());
	double total=0.0;
	for (int i=start_i(a); i<=end_i(a); i++){
	    total += dTdV(i).x()+slope*dTdV(i).y();
//	    total += dTdV(i).x();
	    dTdV(i).y() = 0.0;
	}
	total /= (end_i(a)-start_i(a)+1);
	for (int i=start_i(a); i<=end_i(a); i++){
	    dTdV(i).x() = total; // all points have the same direction now.
	}
    }
}

int BendingSolver2d::check_path_size(Array1d<Point2d>& path)
{
    int np=path.size();
    if (np<2) error("BendingSolver2d::invalid ray path");

    // the following is now taken care of by graph's pickPath()
//    if (np==2){
 	// since end points are fixed, this path wouldn't be
	// refined without an additional point in the middle
// 	Point2d p1=path(1),p2=path(2);
// 	path.resize(3);
// 	path(1)=p1; path(2)=0.5*(p1+p2); path(3)=p2;
//	np=3;
//    }

    return np;
}

int BendingSolver2d::check_path_size(Array1d<Point2d>& path,
				     const Array1d<int>& start_i,
				     const Array1d<int>& end_i,
				     const Array1d<const Interface2d*>& interf)
{
    int np=path.size();
    if (np<2) error("BendingSolver2d::invalid ray path");

    int nintf = interf.size();
    if (start_i.size() != nintf || end_i.size() != nintf)
	error("BendingSolver2d::refine - size mismatch in interface spec");
    for (int i=1; i<=nintf; i++){
	if (start_i(i) > end_i(i) ||
	    start_i(i) < 1 || start_i(i) > np ||
	    end_i(i) < 1 || end_i(i) > np)
	    error("BendingSolver2d::refine - invalid interface points");
    }
    return np;
}
    
int BendingSolver2d::refine(Array1d<Point2d>& path,
			    double& orig_time, double& new_time,
			    int nfac)
{
    if (nfac<1) error("BendingSolver2d::illegal nfac input");
    if (nfac==1){
	return refine(path,orig_time,new_time);
    }
    
    int np=path.size();
    Array1d<Point2d> old_path(np);
    old_path = path;

    int n2=(nfac-1)*(np-1);
    double frac=1.0/nfac;
    path.resize(np+n2);
    for (int i=1; i<np; i++){
	int orig_i = nfac*(i-1)+1;
	path(orig_i) = old_path(i);
	for (int j=1; j<nfac; j++){
	    double ratio=frac*j;
	    path(orig_i+j) = (1-ratio)*old_path(i)+ratio*old_path(i+1);
	}
    }
    path.back()=old_path.back();

    return refine(path,orig_time,new_time);
}

double BendingSolver2d::line_min(Array1d<Point2d>& path, const Array1d<Point2d>& direc)
{
    point_p = &path; // for f1dim()
    direc_p = &direc;

    double ax = 0.0;
    double xx = 1.0;
    double bx, fa, fx, fb, xmin;
    mnbrak(&ax, &xx, &bx, &fa, &fx, &fb, &BendingSolver2d::f1dim);
    double new_val = brent(ax,xx,bx,&xmin, &BendingSolver2d::f1dim);
    for (int i=1; i<=path.size(); i++){
	path(i) += xmin*direc(i);
    }
    clip_if_water(path);
    return new_val;
}

double BendingSolver2d::f1dim(double x)
{
    for (int i=1; i<=point_p->size(); i++){
	new_point(i) = (*point_p)(i)+x*(*direc_p)(i);
    }
    clip_if_water(new_point);
    PsxBendSpec spec;
    double d = calcTravelTime(smesh, new_point, bs, new_pp, Q, clip_lo, clip_hi, psxArg(spec));
    return d;
}

double BendingSolver2d::line_min(Array1d<Point2d>& path, const Array1d<Point2d>& direc,
				 const Array1d<int>& start_i, const Array1d<int>& end_i,
				 const Array1d<const Interface2d*>& interf)
{
    point_p = &path; // for f1dim_interf()
    direc_p = &direc;
    start_i_p = &start_i;
    end_i_p = &end_i;
    interf_p = &interf;

//    double ax = 0.0;
//    double xx = 1.0;
    double ax = -0.1;
    double xx = 0.1;
    double bx, fa, fx, fb, xmin;
    mnbrak(&ax, &xx, &bx, &fa, &fx, &fb, &BendingSolver2d::f1dim_interf);
    brent(ax,xx,bx,&xmin, &BendingSolver2d::f1dim_interf);
    for (int i=1; i<=path.size(); i++){
	path(i) += xmin*direc(i);
    }
    for (int a=1; a<=interf.size(); a++){
	double new_z = interf(a)->z(path(start_i(a)).x());
	for (int i=start_i(a); i<=end_i(a); i++){
	    path(i).y() = new_z;
	}
    }
    clip_if_water(path);
    clamp_psx_below(path);
    PsxBendSpec spec;
    return calcTravelTime(smesh, path, bs, pp, Q, clip_lo, clip_hi, psxArg(spec));
}

double BendingSolver2d::f1dim_interf(double x)
{
    for (int i=1; i<=point_p->size(); i++){
	new_point(i) = (*point_p)(i)+x*(*direc_p)(i);
    }
    for (int a=1; a<=interf_p->size(); a++){
	double new_z = (*interf_p)(a)->z(new_point((*start_i_p)(a)).x());
	for (int i=(*start_i_p)(a); i<=(*end_i_p)(a); i++){
	    new_point(i).y() = new_z;
	}
    }
    clip_if_water(new_point);
    clamp_psx_below(new_point);
    PsxBendSpec spec;
    double d = calcTravelTime(smesh, new_point, bs, new_pp, Q, clip_lo, clip_hi, psxArg(spec));
    return d;
}

int BendingSolver2d::refinePsxPhaseLegs(Array1d<Point2d>& path,
					double& orig_time, double& new_time,
					int i0, int i1, int iu0, int iu1,
					int id0, int id1,
					const Interface2d* bathy, const Interface2d* conv,
					double kappa, bool below_is_s, bool lid_is_s)
{
    const int np = (int)path.size();
    if (!conv || iu1 < 2 || iu1 > np || id1 > np){
	setPsxBend(iu1, id0, id1, kappa, below_is_s, conv, lid_is_s);
	int it = refine(path, orig_time, new_time);
	clearPsxBend();
	return it;
    }

    const bool one_hinge = (id0 <= iu1);
    const int lid0 = one_hinge ? iu1 : id1;
    Point2d p_shot = path(1);
    Point2d p_obs = path(np);

    auto copy_span = [](const Array1d<Point2d>& src, int a, int b,
			Array1d<Point2d>& dst){
	int n = b - a + 1;
	dst.resize(n);
	for (int i=1; i<=n; i++) dst(i) = src(a+i-1);
    };
    auto paste_span = [](Array1d<Point2d>& dst, int a, const Array1d<Point2d>& src){
	for (int i=1; i<=(int)src.size(); i++) dst(a+i-1) = src(i);
    };

    auto span_time = [&](int a, int b, bool all_s, bool clamp_below) -> double {
	if (b <= a) return 0.0;
	Array1d<Point2d> leg;
	copy_span(path, a, b, leg);
	int nloc = (int)leg.size();
	if (all_s)
	    setPsxBend(1, clamp_below ? nloc : 1, 1, kappa, true, conv, true);
	else
	    setPsxBend(nloc, 0, 0, kappa, false, conv, false);
	makeBSpoints(leg, pp);
	PsxBendSpec spec;
	return calcTravelTime(smesh, leg, bs, pp, Q, clip_lo, clip_hi, psxArg(spec));
    };

    // Bend one leg with both ends fixed. Only the water-entry pin (if any)
    // stays interface-pinned; coincident hinge duplicates inside the leg are
    // frozen via freeze_idx so the conversion points are left to slide_hinge.
    auto bend_span = [&](int a, int b, bool all_s, bool clamp_below) -> int {
	if (b - a < 2) return 0;
	Array1d<Point2d> leg;
	copy_span(path, a, b, leg);
	int nloc = (int)leg.size();

	int s = 0, e = -1;
	if (bathy && i0 > 0 && i1 > 0 && i1 >= a && i0 <= b){
	    s = i0 < a ? a : i0;
	    e = i1 > b ? b : i1;
	}
	const int npin = (s <= e) ? 1 : 0;
	Array1d<int> start_i(npin), end_i(npin);
	Array1d<const Interface2d*> interf(npin);
	if (npin){
	    start_i(1) = s - a + 1;
	    end_i(1) = e - a + 1;
	    interf(1) = bathy;
	}

	freeze_idx.resize(0);
	for (int g=iu0; g<iu1; g++)
	    if (g > a && g < b) freeze_idx.push_back(g - a + 1);

	if (all_s)
	    setPsxBend(1, clamp_below ? nloc : 1, 1, kappa, true, conv, true);
	else
	    setPsxBend(nloc, 0, 0, kappa, false, conv, false);

	double t0, t1;
	int it = refine(leg, t0, t1, start_i, end_i, interf);
	freeze_idx.resize(0);
	paste_span(path, a, leg);
	return it;
    };

    // Slide one hinge along the conversion interface, minimizing the JOINT
    // time of the two adjacent legs. The stationary point of t(x_hinge) along
    // the interface is exactly the generalized Snell condition (P<->S or
    // S<->S), so the conversion point leaves the graph's grid-locked position
    // (e.g. the vertical lid S leg) without any hard-coded Snell formula.
    auto slide_hinge = [&](int h0, int h1,
			   int aL, int bL, bool ls, bool lclamp,
			   int aR, int bR, bool rs, bool rclamp,
			   double xn1, double xn2){
	// Original legs; the trial shift is blended into interior nodes
	// (weight 0 at the fixed far end, 1 at the hinge) so the objective
	// is smooth in x instead of kinking the spline at the old hinge.
	if (h0 < 1 || h1 > np || h1 < h0 || aL < 1 || bL > np || aR < 1 || bR > np)
	    return;
	Array1d<Point2d> legL, legR;
	copy_span(path, aL, bL, legL);
	copy_span(path, aR, bR, legR);
	if (legL.size()==0 || legR.size()==0) return;
	const double x0 = path(h1).x();
	if (!std::isfinite(x0) || !std::isfinite(xn1) || !std::isfinite(xn2))
	    return;
	double lo = xn1 < xn2 ? xn1 : xn2;
	double hi = xn1 < xn2 ? xn2 : xn1;
	double pad = 0.2*(hi-lo) + 0.3;
	lo -= pad; hi += pad;
	if (x0 < lo) lo = x0 - 0.05;
	if (x0 > hi) hi = x0 + 0.05;
	const double xmin = conv->xmin();
	const double xmax = conv->xmax();
	if (lo < xmin) lo = xmin;
	if (hi > xmax) hi = xmax;
	if (lo > hi) return;

	auto obj = [&](double x){
	    if (!std::isfinite(x)) return 1e30;
	    if (x < xmin) x = xmin;
	    if (x > xmax) x = xmax;
	    double dxs = x - x0;
	    for (int i=aL; i<=bL; i++){
		double w = (bL > aL) ? double(i-aL)/double(bL-aL) : 1.0;
		path(i) = legL(i-aL+1) + Point2d(w*dxs, 0.0);
	    }
	    for (int i=aR; i<=bR; i++){
		double w = (bR > aR) ? double(bR-i)/double(bR-aR) : 1.0;
		path(i) = legR(i-aR+1) + Point2d(w*dxs, 0.0);
	    }
	    Point2d H(x, conv->z(x));
	    for (int i=h0; i<=h1; i++) path(i) = H;
	    return span_time(aL, bL, ls, lclamp) + span_time(aR, bR, rs, rclamp);
	};
	const double RG = 0.6180339887498948482;
	double ga=lo, gb=hi;
	double c = gb - RG*(gb-ga), d = ga + RG*(gb-ga);
	double fc = obj(c), fd = obj(d);
	for (int it=0; it<26; it++){
	    if (fc < fd){ gb=d; d=c; fd=fc; c=gb-RG*(gb-ga); fc=obj(c); }
	    else { ga=c; c=d; fc=fd; d=ga+RG*(gb-ga); fd=obj(d); }
	}
	obj(0.5*(ga+gb));
    };

    orig_time = span_time(1, iu1, false, false);
    if (!one_hinge)
	orig_time += span_time(iu1, id0, below_is_s, true);
    orig_time += span_time(lid0, np, lid_is_s, false);

    psx_leg_mode = true;
    int last = 0;
    for (int pass=0; pass<2; pass++){
	last = bend_span(1, iu1, false, false);
	if (!one_hinge)
	    last = bend_span(iu1, id0, below_is_s, true);
	last = bend_span(lid0, np, lid_is_s, false);

	slide_hinge(iu0, iu1, 1, iu1, false, false,
		    iu1, one_hinge ? np : id0,
		    one_hinge ? lid_is_s : below_is_s, !one_hinge,
		    path(1).x(), one_hinge ? path(np).x() : path(id0).x());
	if (!one_hinge)
	    slide_hinge(id0, id1, iu1, id0, below_is_s, true,
			lid0, np, lid_is_s, false,
			path(iu1).x(), path(np).x());
    }
    // final relax of leg interiors with hinges at their Snell positions
    last = bend_span(1, iu1, false, false);
    if (!one_hinge)
	last = bend_span(iu1, id0, below_is_s, true);
    last = bend_span(lid0, np, lid_is_s, false);
    psx_leg_mode = false;

    path(1) = p_shot;
    path(np) = p_obs;
    {
	double x = path(iu1).x();
	Point2d B(x, conv->z(x));
	for (int i=iu0; i<=iu1; i++)
	    if (i>=1 && i<=np) path(i) = B;
    }
    if (!one_hinge){
	double x = path(id0).x();
	Point2d C(x, conv->z(x));
	for (int i=id0; i<=id1; i++)
	    if (i>=1 && i<=np) path(i) = C;
    }

    new_time = span_time(1, iu1, false, false);
    if (!one_hinge)
	new_time += span_time(iu1, id0, below_is_s, true);
    new_time += span_time(lid0, np, lid_is_s, false);
    clearPsxBend();
    return last;
}

