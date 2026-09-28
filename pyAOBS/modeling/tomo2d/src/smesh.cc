/*
 * smesh.cc - slowness mesh implementation
 *
 * Jun Korenaga, MIT/WHOI
 * December 1998
 */

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cstdio>
#include "smesh.h"
#include "interface.h"
#include "d_nr.h"
#include "error.h"

// 慢度下限（对应 vmax=30 km/s）。NaN / 非正 / 过小一律钳到此值，避免 1/p 爆炸或负走时。
static const double kSlownessPmin = 1.0/30.0;

static double clamp_nonneg_ttime(int nsrc, int nrcv, double ttime)
{
    if (ttime >= 0.0) return ttime;
    static int nwarn = 0;
    if (nwarn < 20){
	cerr << "SlownessMesh2d::calc_ttime - negative traveltime ("
	     << nsrc << ", " << nrcv << ", " << ttime
	     << "); clamped to 0 (not aborting)\n";
	nwarn++;
	if (nwarn == 20){
	    cerr << "SlownessMesh2d::calc_ttime - further negative-ttime warnings suppressed\n";
	}
    }
    return 0.0;
}

SlownessMesh2d::SlownessMesh2d(const char* fn)
    : dual_vs(false), mix_kappa(1.0), mix_kappa_below(1.0), eps(1e-6)
{
    ifstream s(fn);
    if (!s){
	cerr << "SlownessMesh2d::cannot open " << fn << "\n";
	exit(1);
    }

    s >> nx >> nz >> p_water >> p_air;
    if (p_water <= 0.0)
	cerr << "SlownessMesh2d::invalid water velocity\n";
    if (p_air <= 0.0)
	cerr << "SlownessMesh2d::invalid air velocity\n";
    p_water = 1.0/p_water;
    p_air = 1.0/p_air;
    nnodes = nx*nz;
    ncells = (nx-1)*(nz-1);

    pgrid.resize(nx,nz);
    vgrid.resize(nx,nz);
    ser_index.resize(nx,nz);
    node_index.resize(nnodes);
    cell_index.resize(ncells);
    index2cell.resize(nx-1,nz-1);
    xpos.resize(nx);
    topo.resize(nx);
    zpos.resize(nz);

    for (int i=1; i<=nx; i++) s >> xpos(i);
    for (int i=1; i<=nx; i++) s >> topo(i);
    for (int k=1; k<=nz; k++) s >> zpos(k);
    
    int N=1;
    vgrid = 0.0;
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
	    double vel;
	    s >> vel;
	    vgrid(i,k) = vel;
	    pgrid(i,k) = 1.0/vel;
	    ser_index(i,k) = N;
	    node_index(N).set(i,k);
	    N++;
	}
    }
    int icell=1;
    for (int i=1; i<nx; i++){
	for (int k=1; k<nz; k++){
	    index2cell(i,k) = icell;
	    cell_index(icell++) = ser_index(i,k);
	    
	}
    }

    // sanity check
    for (int i=2; i<=nx; i++){
	if (xpos(i)<=xpos(i-1)){
	    cerr << "SlownessMesh2d: illegal ordering of x nodes at i=" << i << '\n';
	    exit(1);
	}
    }
    for (int i=2; i<=nz; i++){
	if (zpos(i)<=zpos(i-1)){
	    cerr << "SlownessMesh2d: illegal ordering of z nodes at i=" << i << '\n';
	    exit(1);
	}
    }
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
	    if (vgrid(i,k)<=0){
		cerr << "SlownessMesh2d: non-positive velocity is found at (i,k)="
		     << i << "," << k << '\n';
		exit(1);
	    }
	}
    }

    dx_vec.resize(nx-1); rdx_vec.resize(nx-1);
    dz_vec.resize(nz-1); rdz_vec.resize(nz-1);
    b_vec.resize(nx-1);
    for (int i=1; i<nx; i++){
	dx_vec(i) = xpos(i+1)-xpos(i);
	rdx_vec(i) = 1.0/dx_vec(i);
	b_vec(i) = topo(i+1)-topo(i);
    }
    for (int k=1; k<nz; k++){
	dz_vec(k) = zpos(k+1)-zpos(k);
	rdz_vec(k) = 1.0/dz_vec(k);
    }

    Sm_H1.resize(4,4); Sm_H2.resize(4,4); Sm_V.resize(4,4);
    T_common.resize(4,4);
    commonGradientKernel();
    commonNormKernel();
}

void SlownessMesh2d::set(const Array1d<double>& u)
{
    if (u.size() != nnodes) error("SlownessMesh2d::set - size mismatch");

    int N=1;
    int nclip=0;
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
	    double p = u(N);
	    if (!(p > kSlownessPmin)){
		p = kSlownessPmin;
		nclip++;
	    }
	    pgrid(i,k) = p;
	    vgrid(i,k) = 1.0/p;
	    N++;
	}
    }
    if (nclip>0){
	cerr << "SlownessMesh2d::set - clamped " << nclip
	     << " nodes to pmin=" << kSlownessPmin << " (vmax=30 km/s)\n";
    }
}

void SlownessMesh2d::get(Array1d<double>& u) const
{
    if (u.size() != nnodes) error("SlownessMesh2d::set - size mismatch");

    int N=1;
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
	    u(N) = pgrid(i,k);
	    N++;
	}
    }
}

void SlownessMesh2d::vget(Array1d<double>& v) const
{
    if (v.size() != nnodes) error("SlownessMesh2d::set - size mismatch");

    int N=1;
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
	    v(N) = vgrid(i,k);
	    N++;
	}
    }
}

void SlownessMesh2d::setVp(const Array1d<double>& u)
{
    if (!dual_vs){
	set(u);
	return;
    }
    if (u.size() != nnodes) error("SlownessMesh2d::setVp - size mismatch");

    int N=1;
    int nclip=0;
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
	    double p = u(N);
	    if (!(p > kSlownessPmin)){
		p = kSlownessPmin;
		nclip++;
	    }
	    pgrid_vp(i,k) = p;
	    vgrid_vp(i,k) = 1.0/p;
	    N++;
	}
    }
    if (nclip>0){
	cerr << "SlownessMesh2d::setVp - clamped " << nclip
	     << " nodes to pmin=" << kSlownessPmin << " (vmax=30 km/s)\n";
    }
}

void SlownessMesh2d::getVp(Array1d<double>& u) const
{
    if (!dual_vs){
	get(u);
	return;
    }
    if (u.size() != nnodes) error("SlownessMesh2d::getVp - size mismatch");

    int N=1;
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
	    u(N) = pgrid_vp(i,k);
	    N++;
	}
    }
}

bool parseVpVsKappa(const char* s, double& k_lid, double& k_below)
{
    if (!s || s[0]=='\0') return false;
    double a=0.0, b=0.0;
    int n = sscanf(s, "%lf/%lf", &a, &b);
    if (n==1 && a>0.0){ k_lid = k_below = a; return true; }
    if (n==2 && a>0.0 && b>0.0){ k_lid = a; k_below = b; return true; }
    return false;
}

void SlownessMesh2d::setMixKappa(double k)
{
    setMixKappa(k, k);
}

void SlownessMesh2d::setMixKappa(double k_lid, double k_below)
{
    if (k_lid<=0.0 || k_below<=0.0)
	error("SlownessMesh2d::setMixKappa - kappa must be positive.");
    if (!dual_vs){
	mix_kappa = k_lid;
	mix_kappa_below = k_below;
    }
}

void SlownessMesh2d::enableDualVs(double kappa)
{
    enableDualVs(kappa, kappa, 0);
}

void SlownessMesh2d::enableDualVs(double k_lid, double k_below,
				  const Interface2d* conv)
{
    if (k_lid<=0.0 || k_below<=0.0)
	error("SlownessMesh2d::enableDualVs - kappa must be positive.");
    if (dual_vs) return;
    mix_kappa = k_lid;
    mix_kappa_below = k_below;

    pgrid_vp.resize(nx,nz);
    vgrid_vp.resize(nx,nz);
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
	    pgrid_vp(i,k) = pgrid(i,k);
	    vgrid_vp(i,k) = vgrid(i,k);
	    Point2d p = nodePos(ser_index(i,k));
	    if (inWater(p) || inAir(p)) continue;
	    if (abs(pgrid(i,k) - p_water) < 1e-8) continue;
	    const bool below = conv && (p.y() >= conv->z(p.x()) - 1e-6);
	    double kappa = below ? k_below : k_lid;
	    double ps = kappa*pgrid_vp(i,k);
	    if (!(ps > kSlownessPmin)) ps = kSlownessPmin;
	    pgrid(i,k) = ps;
	    vgrid(i,k) = 1.0/ps;
	}
    }
    dual_vs = true;
}

void SlownessMesh2d::resetVsFromVpBelow(double kappa, const Interface2d& conv)
{
    if (!dual_vs || kappa<=0.0) return;
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
	    Point2d p = nodePos(ser_index(i,k));
	    if (inWater(p) || inAir(p)) continue;
	    if (p.y() < conv.z(p.x()) - 1e-6) continue;
	    double ps = kappa*pgrid_vp(i,k);
	    if (!(ps > kSlownessPmin)) ps = kSlownessPmin;
	    pgrid(i,k) = ps;
	    vgrid(i,k) = 1.0/ps;
	}
    }
}

void SlownessMesh2d::resetVsFromVp(double k_lid, double k_below,
				   const Interface2d* conv)
{
    if (!dual_vs) return;
    if (k_lid<=0.0) k_lid = 1.0;
    if (k_below<=0.0) k_below = k_lid;
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
	    Point2d p = nodePos(ser_index(i,k));
	    if (inWater(p) || inAir(p)) continue;
	    if (abs(pgrid_vp(i,k) - p_water) < 1e-8) continue;
	    const bool below = conv && (p.y() >= conv->z(p.x()) - 1e-6);
	    double kappa = below ? k_below : k_lid;
	    double ps = kappa*pgrid_vp(i,k);
	    if (!(ps > kSlownessPmin)) ps = kSlownessPmin;
	    pgrid(i,k) = ps;
	    vgrid(i,k) = 1.0/ps;
	}
    }
}

void SlownessMesh2d::loadDualVs(const char* vsfn)
{
    if (dual_vs) return;
    SlownessMesh2d vs(vsfn);
    if (vs.Nx() != nx || vs.Nz() != nz)
	error("SlownessMesh2d::loadDualVs - Vs mesh nx/nz mismatch");

    pgrid_vp.resize(nx,nz);
    vgrid_vp.resize(nx,nz);
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
	    pgrid_vp(i,k) = pgrid(i,k);
	    vgrid_vp(i,k) = vgrid(i,k);
	    Point2d p = nodePos(ser_index(i,k));
            if (inWater(p) || inAir(p)) continue;
	    if (abs(pgrid(i,k) - p_water) < 1e-8) continue;
	    double ps = vs.pgrid(i,k);
	    double vel = vs.vgrid(i,k);
	    if (!(vel > 0.0) || !(ps > kSlownessPmin)) continue;
	    pgrid(i,k) = ps;
	    vgrid(i,k) = vel;
	}
    }
    dual_vs = true;
}

double SlownessMesh2d::atVp(const Point2d& pos) const
{
    Index2d guess = nodeIndex(nearest(pos));
    return atVp(pos,guess);
}

double SlownessMesh2d::atVp(const Point2d& pos, Index2d& guess) const
{
    if (!dual_vs) return at(pos,guess);
    upperleft(pos,guess);
    if (in_water(pos,guess)) return p_water;
    if (in_air(pos,guess)) return p_air;

    int i=guess.i(), k=guess.k();
    if (i < 1) i = 1;
    if (k < 1) k = 1;
    if (i > nx-1) i = nx-1;
    if (k > nz-1) k = nz-1;
    guess.set(i, k);
    double r,s,rr,ss;
    calc_local(pos,i,k,r,s,rr,ss);
    double u=
	rr*ss*vgrid_vp(i,k)+rr*s*vgrid_vp(i,k+1)
	+r*s*vgrid_vp(i+1,k+1)+r*ss*vgrid_vp(i+1,k);
    return 1.0/u;
}

double SlownessMesh2d::atVp(const Point2d& pos, Index2d& guess,
			    double& dudx, double& dudz) const
{
    if (!dual_vs) return at(pos, guess, dudx, dudz);
    upperleft(pos,guess);
    if (in_water(pos,guess)){
	dudx=dudz=0.0;
	return p_water;
    }
    if (in_air(pos,guess)){
	dudx=dudz=0.0;
	return p_air;
    }

    int i=guess.i(), k=guess.k();
    if (i < 1) i = 1;
    if (k < 1) k = 1;
    if (i > nx-1) i = nx-1;
    if (k > nz-1) k = nz-1;
    guess.set(i, k);
    double r,s,rr,ss;
    calc_local(pos,i,k,r,s,rr,ss);

    double u1 = vgrid_vp(i,k);
    double u2 = vgrid_vp(i,k+1);
    double u3 = vgrid_vp(i+1,k+1);
    double u4 = vgrid_vp(i+1,k);
    double u = rr*ss*u1+rr*s*u2+r*s*u3+r*ss*u4;

    double rDX = rdx_vec(i);
    double rDZ = rdz_vec(k);
    double B = b_vec(i);
    double a = ss*u1+s*u2;
    double b = s*u3+ss*u4-a;
    double a_br = a+b*r;
    double dudr = -b/(a_br*a_br);
    double c = rr*u1+r*u4;
    double d = rr*u2+r*u3-c;
    double c_ds = c+d*s;
    double duds = -d/(c_ds*c_ds);
    dudz = rDZ*duds;
    dudx = rDX*(dudr-B*dudz);
    return 1.0/u;
}

double SlownessMesh2d::xmin() const { return xpos.front(); }

double SlownessMesh2d::xmax() const { return xpos.back(); }

double SlownessMesh2d::zmin() const
{
    double tmin=1e30;
    for (int i=1; i<=topo.size(); i++)
	if (topo(i)<tmin) tmin = topo(i);

    return tmin+zpos.front();
}

double SlownessMesh2d::zmax() const
{
    double tmax=-1e30;
    for (int i=1; i<=topo.size(); i++)
	if (topo(i)>tmax) tmax = topo(i);
    return tmax+zpos.back();
}

const Index2d&
SlownessMesh2d::nodeIndex(int i) const { return node_index(i); }

int SlownessMesh2d::nodeIndex(int i, int k) const { return ser_index(i,k); }

Point2d SlownessMesh2d::nodePos(int i) const
{
    int node_i = node_index(i).i();
    int node_k = node_index(i).k();
    Point2d p(xpos(node_i),zpos(node_k)+topo(node_i));

    return p;
}

double SlownessMesh2d::calc_ttime(int nsrc, int nrcv) const
{
    return calc_ttime_on(nsrc, nrcv, pgrid);
}

double SlownessMesh2d::calc_ttime_vp(int nsrc, int nrcv) const
{
    if (!dual_vs) return calc_ttime(nsrc, nrcv);
    return calc_ttime_on(nsrc, nrcv, pgrid_vp);
}

double SlownessMesh2d::calc_ttime_converse_dual(int nsrc, int nrcv,
					       const Interface2d& conv) const
{
    if (!dual_vs) return calc_ttime(nsrc, nrcv);
    return calc_ttime_on(nsrc, nrcv, pgrid, &conv);
}

double SlownessMesh2d::calc_ttime_psx(int nsrc, int nrcv,
				     const Interface2d& conv, bool want_s) const
{
    return calc_ttime_psx(nsrc, nrcv, conv, want_s ? 2 : 1);
}

double SlownessMesh2d::calc_ttime_psx(int nsrc, int nrcv,
				     const Interface2d& conv, int psx_mode) const
{
    return calc_ttime_on(nsrc, nrcv, pgrid, &conv, psx_mode);
}

double SlownessMesh2d::p_on_node(int i, int k, const Array2d<double>& pg,
				const Interface2d* mix_conv, int psx_leg) const
{
    if (psx_leg != 0 && mix_conv){
	Point2d p(xpos(i), zpos(k)+topo(i));
	const bool want_s = (psx_leg == 2 || psx_leg == 3);
	if (inWater(p) || inAir(p)){
	    if (!want_s && dual_vs) return pgrid_vp(i,k);
	    return pgrid(i,k);
	}
	bool sample_below;
	if (psx_leg == 4)
	    sample_below = p.y() >= mix_conv->z(p.x()) + 1e-3;
	else if (psx_leg == 2)
	    sample_below = true;
	else
	    sample_below = false;
	double v = psx_corner_vel(i, k, *mix_conv, want_s, sample_below);
	if (v > 0.0) return 1.0/v;
	return (!want_s && dual_vs) ? pgrid_vp(i,k) : pgrid(i,k);
    }
    if (mix_conv && dual_vs){
	Point2d p(xpos(i), zpos(k)+topo(i));
	if (inWater(p) || inAir(p)) return pgrid(i,k);
	if (p.y() >= mix_conv->z(p.x()) - 1e-6) return pgrid(i,k);
	return pgrid_vp(i,k);
    }
    return pg(i,k);
}

double SlownessMesh2d::calc_ttime_on(int nsrc, int nrcv,
				    const Array2d<double>& pg,
				    const Interface2d* mix_conv, int psx_leg) const
{
    if (nsrc<1 || nsrc>nnodes || nrcv<1 || nrcv>nnodes)
	error("SlownessMesh2d::calc_ttime - invalid input");
    Index2d src=node_index(nsrc);
    Index2d rcv=node_index(nrcv);
    int isrc=src.i(), ksrc=src.k();
    int ircv=rcv.i(), krcv=rcv.k();
    int di = ircv-isrc;
    int dk = krcv-ksrc;
    int dii = di > 0 ? 1 : -1;
    int dkk = dk > 0 ? 1 : -1;
    
    double ttime=0.0;
    if (di==0){ // ray path on a vertical grid
	int k1=ksrc, k2=ksrc+dkk;
	while (dk!=0){
	    double dz=abs(zpos(k1)-zpos(k2));
	    double pave = 0.5*(p_on_node(isrc,k1,pg,mix_conv,psx_leg)
			       +p_on_node(isrc,k2,pg,mix_conv,psx_leg));
	    ttime += dz*pave;
	    dk -= dkk; k1 += dkk; k2 += dkk;
	}
    }else if (abs(di)==1){ // ray path within a single sheared column
	double dx = abs(xpos(isrc)-xpos(ircv));
	double pave1 = p_on_node(isrc,ksrc,pg,mix_conv,psx_leg);
	double dtopo = abs(topo(isrc)-topo(ircv));
	if (dk==0){
	    double dist = sqrt(dtopo*dtopo+dx*dx);
	    double pave2 = p_on_node(ircv,ksrc,pg,mix_conv,psx_leg);
	    double pave = 0.5*(pave1+pave2);
	    ttime += dist*pave;
	}else{
	    double ddx = dx/abs(dk);
	    double ddx2 = ddx*ddx;
	    double dratio = 1.0/abs(dk);

	    double ratio = dratio;
	    int k1=ksrc, k2=ksrc+dkk;
	    while (dk!=0){
		double dz=abs(zpos(k1)-zpos(k2))+dtopo;
		double dist = sqrt(dz*dz+ddx2);
		double pave2 = (1.0-ratio)*p_on_node(isrc,k2,pg,mix_conv,psx_leg)
		    +ratio*p_on_node(ircv,k2,pg,mix_conv,psx_leg);
		double pave = 0.5*(pave1+pave2);
		ttime += dist*pave;
		dk -= dkk; k1 += dkk; k2 += dkk;
		ratio += dratio;
		pave1 = pave2; 
	    }
	}
    }else{
	double xs=xpos(isrc), xr=xpos(ircv);
	double dx = xr-xs;
	double zs=topo(isrc)+zpos(ksrc), zr=topo(ircv)+zpos(krcv);
	double dz = zr-zs;
	double x1=xs, z1=zs;
	double pave1 = p_on_node(isrc,ksrc,pg,mix_conv,psx_leg);
	int i2 = isrc+dii;
	double pave2;
	while (di!=0){
	    double x2 = xpos(i2);
	    double z2 = zs+dz*(x2-xs)/dx;
	    if (topo(i2)+zpos(1) > z2){ // above the model domain
		pave2 = p_water;
	    }else if (topo(i2)+zpos(nz) < z2){ // below the model domain
		pave2 = p_on_node(i2,nz,pg,mix_conv,psx_leg);
	    }else{
		// search for enclosing nodes
		int k1=-1, k2=-1;
		double zsrc = topo(i2)+zpos(ksrc);
		if (abs(zsrc-z2)<eps){
		    k1 = k2 = ksrc;
		}else if (zsrc<z2){ // search downward
		    if (ksrc<nz){
			for (int k=ksrc+1; k<=nz; k++){
			    double ztest = topo(i2)+zpos(k);
			    if (abs(ztest-z2)<eps){
				k1 = k2 = k;
				break;
			    }else if (ztest>z2){
				k2 = k;
				k1 = k-1;
				break;
			    }
			}
		    }
		}else{		// search upward
		    if (ksrc>1){
			for (int k=ksrc-1; k>-1; k--){
			    double ztest = topo(i2)+zpos(k);
			    if (abs(ztest-z2)<eps){
				k1 = k2 = k;
				break;
			    }else if (ztest<z2){
				k1 = k;
				k2 = k+1;
				break;
			    }
			}
		    }
		}
		if (k1==k2){
		    pave2 = p_on_node(i2,k1,pg,mix_conv,psx_leg);
		}else{
		    double ztest1=topo(i2)+zpos(k1);
		    double ztest2=topo(i2)+zpos(k2);
		    double dztest=ztest2-ztest1;
		    double ratio = (z2-ztest1)/dztest;
		    pave2 = ratio*p_on_node(i2,k1,pg,mix_conv,psx_leg)
			+(1.0-ratio)*p_on_node(i2,k2,pg,mix_conv,psx_leg);
		}
	    }
	    double ddx=x2-x1;
	    double ddz=z2-z1;
	    double dist = sqrt(ddx*ddx+ddz*ddz);
	    double pave = 0.5*(pave1+pave2);
	    ttime += dist*pave;

	    di-=dii; i2+=dii;
	    x1=x2; z1=z2;
	    pave1=pave2;
	}
    }

    if (ttime<0){
	ttime = clamp_nonneg_ttime(nsrc, nrcv, ttime);
    }
    return ttime;
}

#ifdef notdef

double SlownessMesh2d::calc_ttime3(int nsrc, int nrcv) const
{
    const double tol_vdiff=1e-4;
    
    if (nsrc<1 || nsrc>nnodes || nrcv<1 || nrcv>nnodes)
	error("SlownessMesh2d::calc_ttime - invalid input");
    Index2d src=node_index(nsrc);
    Index2d rcv=node_index(nrcv);
    int isrc=src.i(), ksrc=src.k();
    int ircv=rcv.i(), krcv=rcv.k();
    int di = ircv-isrc;
    int dk = krcv-ksrc;
    int dii = di > 0 ? 1 : -1;
    int dkk = dk > 0 ? 1 : -1;
    
    double ttime=0.0;
    if (di==0){ // ray path on a vertical grid
	int k1=ksrc, k2=ksrc+dkk;
	while (dk!=0){
	    double dz=abs(zpos(k1)-zpos(k2));
	    double v1=vgrid(isrc,k1), v2=vgird(isrc,k2);
	    ttime += almost_exact_ttime(v1,v2,dz);
	    dk -= dkk; k1 += dkk; k2 += dkk;
	}
    }else if (abs(di)==1){ // ray path within a single sheared column
	double dx = abs(xpos(isrc)-xpos(ircv));
	double v1 = vgrid(isrc,ksrc);
	double dtopo = abs(topo(isrc)-topo(ircv));
	if (dk==0){
	    double dist = sqrt(dtopo*dtopo+dx*dx);
	    double v2 = vgrid(ircv,ksrc);
	    ttime += almost_exact_ttime(v1,v2,dist);
	}else{
	    double ddx = dx/abs(dk);
	    double ddx2 = ddx*ddx;
	    double dratio = 1.0/abs(dk);

	    double ratio = dratio;
	    int k1=ksrc, k2=ksrc+dkk;
	    while (dk!=0){
		double dz=abs(zpos(k1)-zpos(k2))+dtopo;
		double dist = sqrt(dz*dz+ddx2);
		double v2 = (1.0-ratio)*vgrid(isrc,k2)+ratio*vgrid(ircv,k2);
		ttime += almost_exact_ttime(v1,v2,dist);
		dk -= dkk; k1 += dkk; k2 += dkk;
		ratio += dratio;
	    }
	}
    }else{
	double xs=xpos(isrc), xr=xpos(ircv);
	double dx = xr-xs;
	double zs=topo(isrc)+zpos(ksrc), zr=topo(ircv)+zpos(krcv);
	double dz = zr-zs;
	double x1=xs, z1=zs;
	double v1 = vgrid(isrc,ksrc);
	double i2 = isrc+dii;
	double pave2;
	while (di!=0){
	    double x2 = xpos(i2);
	    double z2 = zs+dz*(x2-xs)/dx;
	    if (topo(i2)+zpos(1) > z2){ // above the model domain
		pave2 = p_water;
	    }else if (topo(i2)+zpos(nz) < z2){ // below the model domain
		pave2 = pgrid(i2,nz);
	    }else{
		// search for enclosing nodes
		int k1=-1, k2=-1;
		double zsrc = topo(i2)+zpos(ksrc);
		if (abs(zsrc-z2)<eps){
		    k1 = k2 = ksrc;
		}else if (zsrc<z2){ // search downward
		    if (ksrc<nz){
			for (int k=ksrc+1; k<=nz; k++){
			    double ztest = topo(i2)+zpos(k);
			    if (abs(ztest-z2)<eps){
				k1 = k2 = k;
				break;
			    }else if (ztest>z2){
				k2 = k;
				k1 = k-1;
				break;
			    }
			}
		    }
		}else{		// search upward
		    if (ksrc>1){
			for (int k=ksrc-1; k>-1; k--){
			    double ztest = topo(i2)+zpos(k);
			    if (abs(ztest-z2)<eps){
				k1 = k2 = k;
				break;
			    }else if (ztest<z2){
				k1 = k;
				k2 = k+1;
				break;
			    }
			}
		    }
		}
		if (k1==k2){
		    pave2 = pgrid(i2,k1);
		}else{
		    double ztest1=topo(i2)+zpos(k1);
		    double ztest2=topo(i2)+zpos(k2);
		    double dztest=ztest2-ztest1;
		    double ratio = (z2-ztest1)/dztest;
		    pave2 = ratio*pgrid(i2,k1)+(1.0-ratio)*pgrid(i2,k2);
		}
	    }
	    double ddx=x2-x1;
	    double ddz=z2-z1;
	    double dist = sqrt(ddx*ddx+ddz*ddz);
	    double pave = 0.5*(pave1+pave2);
	    ttime += dist*pave;

	    di-=dii; i2+=dii;
	    x1=x2; z1=z2; pave1=pave2;
	}
    }

    if (ttime<0){
	ttime = clamp_nonneg_ttime(nsrc, nrcv, ttime);
    }
    return ttime;
}

double SlownessMesh2d::almost_exact_ttime(double v1, double v2,
					  double dpath) const
{
    const double tol_vdiff=1e-4;
    
    double dv = v2-v1;
    if (abs(dv)<tol_vdiff){
	return dpath*log(v2/v1)/dv;
    }else{
	return dpath/v1;
    }
}

#endif

void SlownessMesh2d::upperleft(const Point2d& p, Index2d& guess) const
{
    // note: this code guarantees that the final guess index is bounded by
    //       valid range (1...nx-1)(1...nz-1).
    int i_guess=guess.i(), k_guess=guess.k();
    if (i_guess < 1) i_guess = 1;
    if (i_guess > nx) i_guess = nx;
    if (k_guess < 1) k_guess = 1;
    if (k_guess > nz) k_guess = nz;

    if (xpos(i_guess)<=p.x()){
	if (i_guess < nx) i_guess++;
	while (xpos(i_guess)<=p.x() && i_guess<nx) i_guess++;
	i_guess--;
    }else{
	while (xpos(i_guess)>p.x() && i_guess>1) i_guess--;
    }
    double r = (p.x()-xpos(i_guess))*rdx_vec(i_guess);
    double dz = r*b_vec(i_guess)+topo(i_guess);

    if (zpos(k_guess)+dz<=p.y()){
	if (k_guess < nz) k_guess++;
	while (zpos(k_guess)+dz<=p.y() && k_guess<nz) k_guess++;
	k_guess--;
    }else{
	while (zpos(k_guess)+dz>p.y() && k_guess>1) k_guess--;
    }

    guess.set(i_guess,k_guess);
}

void SlownessMesh2d::calc_local(const Point2d& pos, int i, int k,
				double& r, double& s, double& rr, double& ss) const
{
    double rDX = rdx_vec(i);
    double rDZ = rdz_vec(k);
    double B = b_vec(i);

    // global coordinates with an origin at the upperleft node
    double x = pos.x()-xpos(i);
    double z = pos.y()-(zpos(k)+topo(i));
	
    // local coordinates
    r = x*rDX;
    s = (z-B*r)*rDZ;
    rr=1-r;
    ss=1-s;
}


double SlownessMesh2d::at(const Point2d& pos, Index2d& guess) const
{
    upperleft(pos,guess);
    if (in_water(pos,guess)) return p_water;
    if (in_air(pos,guess)) return p_air;
	
    int i=guess.i(), k=guess.k();
    if (i < 1) i = 1;
    if (k < 1) k = 1;
    if (i > nx-1) i = nx-1;
    if (k > nz-1) k = nz-1;
    guess.set(i, k);
    double r,s,rr,ss;
    calc_local(pos,i,k,r,s,rr,ss);

    double u=
	rr*ss*vgrid(i,k)+rr*s*vgrid(i,k+1)
	+r*s*vgrid(i+1,k+1)+r*ss*vgrid(i+1,k);
//     if (u<0){
// 	cerr << "at i=" << i << " k=" << k << " r=" << r
// 	     << " s=" << s << " x=" << pos.x() << " z=" << pos.y()
// 	     << " grid x = " << xpos(i) << "," << xpos(i+1)
// 	     << " grid z = " << topo(i)+zpos(k) << "," << topo(i)+zpos(k+1)
// 	     << " " << topo(i+1)+zpos(k) << "," << topo(i+1)+zpos(k+1) << '\n';
//     }
    return 1.0/u;
}

double SlownessMesh2d::at(const Point2d& pos) const
{
    Index2d guess = nodeIndex(nearest(pos));
    return at(pos,guess);
}

double SlownessMesh2d::at(const Point2d& pos, Index2d& guess,
			  double& dudx, double& dudz) const
{
    upperleft(pos,guess);
    if (in_water(pos,guess)){
	dudx=dudz=0.0;
	return p_water;
    }
    if (in_air(pos,guess)){
	dudx=dudz=0.0;
	return p_air;
    }
    
    int i=guess.i(), k=guess.k();
    if (i < 1) i = 1;
    if (k < 1) k = 1;
    if (i > nx-1) i = nx-1;
    if (k > nz-1) k = nz-1;
    guess.set(i, k);
    double r,s,rr,ss;
    calc_local(pos,i,k,r,s,rr,ss);

    double u1 = vgrid(i,k);
    double u2 = vgrid(i,k+1);
    double u3 = vgrid(i+1,k+1);
    double u4 = vgrid(i+1,k);
    double u = rr*ss*u1+rr*s*u2+r*s*u3+r*ss*u4;

    double rDX = rdx_vec(i);
    double rDZ = rdz_vec(k);
    double B = b_vec(i);
//    dudz = rDZ*(rr*(pgrid(i,k+1)-pgrid(i,k))+r*(pgrid(i+1,k+1)-pgrid(i+1,k)));
//    dudx = rDX*(ss*(pgrid(i+1,k)-pgrid(i,k))+s*(pgrid(i+1,k+1)-pgrid(i,k+1)) - B*dudz);
//    return 1.0/u;
    
    double a = ss*u1+s*u2;
    double b = s*u3+ss*u4-a;
    double a_br = a+b*r;
    double dudr = -b/(a_br*a_br);
    double c = rr*u1+r*u4;
    double d = rr*u2+r*u3-c;
    double c_ds = c+d*s;
    double duds = -d/(c_ds*c_ds);
    dudz = rDZ*duds;
    dudx = rDX*(dudr-B*dudz);
    
    return 1.0/u;
}

double SlownessMesh2d::psx_node_phase_vel(int i, int k, const Interface2d& conv,
					  bool want_s) const
{
    Point2d p(xpos(i), zpos(k)+topo(i));
    if (inWater(p) || inAir(p)){
	if (!want_s && dual_vs) return vgrid_vp(i, k);
	return vgrid(i, k);
    }
    if (dual_vs)
	return want_s ? vgrid(i, k) : vgrid_vp(i, k);

    // Folded mixed: generate the missing field at this node (locked dual).
    const bool below = p.y() >= conv.z(p.x()) - 1e-6;
    const double v = vgrid(i, k);
    if (!(v > 0.0) || mix_kappa <= 0.0 || mix_kappa_below <= 0.0) return v;
    if (want_s) return below ? v : v / mix_kappa;
    return below ? v * mix_kappa_below : v;
}

double SlownessMesh2d::psx_corner_vel(int i, int k, const Interface2d& conv,
				      bool want_s, bool sample_below) const
{
    Point2d p(xpos(i), zpos(k)+topo(i));
    const double zc = conv.z(p.x());
    const bool node_below = p.y() >= zc - 1e-6;
    if (sample_below){
	if (!node_below){
	    for (int kk=k; kk<=nz; ++kk){
		Point2d q(xpos(i), zpos(kk)+topo(i));
		if (inWater(q) || inAir(q)) continue;
		if (q.y() >= zc - 1e-6)
		    return psx_node_phase_vel(i, kk, conv, want_s);
	    }
	}
	return psx_node_phase_vel(i, k, conv, want_s);
    }
    if (node_below){
	for (int kk=k-1; kk>=1; --kk){
	    Point2d q(xpos(i), zpos(kk)+topo(i));
	    if (inWater(q) || inAir(q)) continue;
	    if (q.y() < zc - 1e-6)
		return psx_node_phase_vel(i, kk, conv, want_s);
	}
    }
    return psx_node_phase_vel(i, k, conv, want_s);
}

double SlownessMesh2d::atPsx_interp(const Point2d& pos, const Interface2d& conv,
				    bool want_s, bool sample_below, Index2d& guess,
				    double* dudx, double* dudz) const
{
    upperleft(pos, guess);
    if (in_water(pos, guess)){
	if (dudx && dudz) *dudx = *dudz = 0.0;
	return p_water;
    }
    if (in_air(pos, guess)){
	if (dudx && dudz) *dudx = *dudz = 0.0;
	return p_air;
    }
    int i=guess.i(), k=guess.k();
    if (i < 1) i = 1;
    if (k < 1) k = 1;
    if (i > nx-1) i = nx-1;
    if (k > nz-1) k = nz-1;
    guess.set(i, k);
    double r,s,rr,ss;
    calc_local(pos,i,k,r,s,rr,ss);
    double u1 = psx_corner_vel(i,   k,   conv, want_s, sample_below);
    double u2 = psx_corner_vel(i,   k+1, conv, want_s, sample_below);
    double u3 = psx_corner_vel(i+1, k+1, conv, want_s, sample_below);
    double u4 = psx_corner_vel(i+1, k,   conv, want_s, sample_below);
    double u = rr*ss*u1+rr*s*u2+r*s*u3+r*ss*u4;
    if (dudx && dudz){
	double rDX = rdx_vec(i);
	double rDZ = rdz_vec(k);
	double B = b_vec(i);
	double a = ss*u1+s*u2;
	double b = s*u3+ss*u4-a;
	double a_br = a+b*r;
	double dudr = -b/(a_br*a_br);
	double c = rr*u1+r*u4;
	double d = rr*u2+r*u3-c;
	double c_ds = c+d*s;
	double duds = -d/(c_ds*c_ds);
	*dudz = rDZ*duds;
	*dudx = rDX*(dudr-B*(*dudz));
    }
    return 1.0/u;
}

double SlownessMesh2d::atPsx(const Point2d& pos, const Interface2d& conv,
			     bool want_s, bool sample_below, Index2d& guess) const
{
    return atPsx_interp(pos, conv, want_s, sample_below, guess, 0, 0);
}

double SlownessMesh2d::atPsx(const Point2d& pos, const Interface2d& conv,
			     bool want_s, bool sample_below, Index2d& guess,
			     double& dudx, double& dudz) const
{
    return atPsx_interp(pos, conv, want_s, sample_below, guess, &dudx, &dudz);
}

void SlownessMesh2d::cellNodes(int icell, int& j1, int& j2, int& j3, int& j4) const
{
    j1 = cell_index(icell);
    Index2d index=node_index(j1);
    int i=index.i(), k=index.k();
    j2 = ser_index(i,k+1);
    j3 = ser_index(i+1,k+1);
    j4 = ser_index(i+1,k);
}

void SlownessMesh2d::cellGradientKernel(int icell, Array2d<double>& R,
					double Lh2, double Lv2) const
{
    Index2d index=node_index(cell_index(icell));
    int i=index.i(), k=index.k();
    double dx=dx_vec(i), dz=dz_vec(k), b=b_vec(i);
    double rdx=rdx_vec(i), rdz=rdz_vec(k);

    Array2d<double> Sm(4,4);
    Sm = (Lh2*dz*rdx)*Sm_H1 - (Lh2*b*rdx)*Sm_H2
	+ (Lh2*b*b*rdx*rdz+Lv2*dx*rdz)*Sm_V;

    Array2d<double> Dm(4,4);
    Array1d<double> w(4);
    int nrot;
    d_jacobi(Sm.toRecipe(), 4, w.toRecipe(), Dm.toRecipe(), &nrot);
    for (int m=1; m<=4; m++){
	if (abs(w(m))<1e-10) w(m) = 0.0;
	double tmp = sqrt(w(m));
	for (int n=1; n<=4; n++) R(m,n) = tmp*Dm(n,m);
    }
}

void SlownessMesh2d::commonGradientKernel()
{
    Sm_H1(1,1) =  2; Sm_H1(1,2) =  1; Sm_H1(1,3) = -1; Sm_H1(1,4) = -2;
    Sm_H1(2,1) =  1; Sm_H1(2,2) =  2; Sm_H1(2,3) = -2; Sm_H1(2,4) = -1;
    Sm_H1(3,1) = -1; Sm_H1(3,2) = -2; Sm_H1(3,3) =  2; Sm_H1(3,4) =  1;
    Sm_H1(4,1) = -2; Sm_H1(4,2) = -1; Sm_H1(4,3) =  1; Sm_H1(4,4) =  2;
    Sm_H1 *= (1.0/6.0);

    Sm_H2(1,1) =  1; Sm_H2(1,2) =  0; Sm_H2(1,3) = -1; Sm_H2(1,4) =  0;
    Sm_H2(2,1) =  0; Sm_H2(2,2) = -1; Sm_H2(2,3) =  0; Sm_H2(2,4) =  1;
    Sm_H2(3,1) = -1; Sm_H2(3,2) =  0; Sm_H2(3,3) =  1; Sm_H2(3,4) =  0;
    Sm_H2(4,1) =  0; Sm_H2(4,2) =  1; Sm_H2(4,3) =  0; Sm_H2(4,4) = -1;
    Sm_H2 *= (1.0/2.0);

    Sm_V(1,1) =  2; Sm_V(1,2) = -2; Sm_V(1,3) = -1; Sm_V(1,4) =  1;
    Sm_V(2,1) = -2; Sm_V(2,2) =  2; Sm_V(2,3) =  1; Sm_V(2,4) = -1;
    Sm_V(3,1) = -1; Sm_V(3,2) =  1; Sm_V(3,3) =  2; Sm_V(3,4) = -2;
    Sm_V(4,1) =  1; Sm_V(4,2) = -1; Sm_V(4,3) = -2; Sm_V(4,4) =  2;
    Sm_V *= (1.0/6.0);
}

void SlownessMesh2d::cellNormKernel(int icell, Array2d<double>& T) const
{
    Index2d index=node_index(cell_index(icell));
    int i=index.i(), k=index.k();
    double dx=dx_vec(i), dz=dz_vec(k);
    T = T_common*(dx*dz);
}

void SlownessMesh2d::commonNormKernel()
{
    T_common(1,1) = 4; T_common(1,2) = 2; T_common(1,3) = 1; T_common(1,4) = 2;
    T_common(2,1) = 2; T_common(2,2) = 4; T_common(2,3) = 2; T_common(2,4) = 1;
    T_common(3,1) = 1; T_common(3,2) = 2; T_common(3,3) = 4; T_common(3,4) = 2;
    T_common(4,1) = 2; T_common(4,2) = 1; T_common(4,3) = 1; T_common(4,4) = 4;
    T_common *= (1.0/36.0);

    Array1d<double> p(4);
    d_choldc(T_common.toRecipe(), 4, p.toRecipe());
    for (int i=1; i<=4; i++){
	T_common(i,i) = p(i);
	for (int j=i+1; j<=4; j++){
	    T_common(i,j) = T_common(j,i);
	}
    }
}

int SlownessMesh2d::locateInCell(const Point2d& p, Index2d& guess,
				 int& j1, int& j2, int& j3 , int& j4,
				 double& r, double& s, double& rr, double& ss) const
{
    upperleft(p,guess);
    if (in_water(p,guess)) return -1;
    if (in_air(p,guess)) return -2;
    
    int i=guess.i(), k=guess.k();
    j1 = ser_index(i,k);
    j2 = ser_index(i,k+1);
    j3 = ser_index(i+1,k+1);
    j4 = ser_index(i+1,k);
    calc_local(p,i,k,r,s,rr,ss);
    return index2cell(i,k);
}

void SlownessMesh2d::nodalCellVolume(Array1d<double>& dx, Array1d<double>& dz,
				     Array1d<Point2d>& center) const
{
    //
    // determine cell dimensions and center positions
    //
    dx.resize(nnodes);
    dz.resize(nnodes);
    center.resize(nnodes);
    // first x
    for (int k=1; k<=nz; k++){
	center(ser_index(1,k)).x(xpos(1)+0.25*dx_vec(1)); // left
	dx(ser_index(1,k)) = 0.5*dx_vec(1);
	center(ser_index(nx,k)).x(xpos(nx)-0.25*dx_vec(nx-1)); // right
	dx(ser_index(nx,k)) = 0.5*dx_vec(nx-1);
	for (int i=2; i<nx; i++){
	    center(ser_index(i,k)).x(xpos(i)+0.5*(dx_vec(i)-dx_vec(i-1)));
	    dx(ser_index(i,k)) = 0.5*(dx_vec(i)+dx_vec(i-1));
	}
    }
    // then z
    for (int i=1; i<=nx; i++){
	center(ser_index(i,1)).y(topo(i)+zpos(1)+0.25*dz_vec(1)); // top
	dz(ser_index(i,1)) = 0.5*dz_vec(1);
	center(ser_index(i,nz)).y(topo(i)+zpos(nz)-0.25*dz_vec(nz-1)); // bottom
	dz(ser_index(i,nz)) = 0.5*dz_vec(nz-1);
	for (int k=2; k<nz; k++){
	    center(ser_index(i,k)).y(topo(i)+zpos(k)+0.5*(dz_vec(k)-dz_vec(k-1)));
	    dz(ser_index(i,k)) = 0.5*(dz_vec(k)+dz_vec(k-1));
	}
    }
}
 
int SlownessMesh2d::nearest(const Point2d& src) const
{
    // returns the serial index of the node nearest to
    // the given source location
    int inear=-1, knear=-1;
    double srcx=src.x(), srcz=src.y();

    for (int i=1; i<nx; i++){
	double xnode=xpos(i);
	double xnode2=xpos(i+1);
	double hdx=0.5*(xnode2-xnode);

	if (i==1 && (srcx-xnode)<=eps){ // to the left
	    inear = 1;
	}else if (i==(nx-1) && (srcx-xnode2)>=eps){ // to the right
	    inear = nx;
	}else if (abs(srcx-xnode)<eps){ // on a grid
	    inear = i; 
	}else if (abs(srcx-xnode2)<eps){ // on a grid
	    inear = i+1; 
	}else if (srcx > xnode && srcx < xnode2){
	    if (abs(srcx-xnode) < hdx){
		inear = i;
	    }else{
		inear = i+1;
	    }
	}

	if (inear > 0){
	    for (int k=1; k<nz; k++){
		double znode=zpos(k)+topo(inear);
		double znode2=zpos(k+1)+topo(inear);
		double hdz=0.5*(znode2-znode);

		if (k==1 && (srcz-znode)<=eps){ // above the domain
		    knear = 1;
		}else if (k==(nz-1) && (srcz-znode2)>=eps){ // below the domain
		    knear = nz;
		}else if (abs(srcz-znode)<eps){ // on a grid
		    knear = k; 
		}else if (abs(srcz-znode2)<eps){ // on a grid
		    knear = k+1; 
		}else if (srcz > znode && srcz < znode2){
		    if (abs(srcz-znode) < hdz){
			knear = k;
		    }else{
			knear = k+1;
		    }
		}
		if (knear > 0) break;
	    }
	}

	if (inear > 0 && knear > 0) break;
    }

    return ser_index(inear, knear);
}

void SlownessMesh2d::nearest(const Interface2d& itf, Array1d<int>& inodes) const
{
    int nx = xpos.size();
    inodes.resize(nx);
    for (int i=1; i<=nx; i++){
	double x=xpos(i);
	double z=itf.z(x);
	Point2d p(x,z);
	inodes(i) = nearest(p);
    }
}

void SlownessMesh2d::outMesh(ostream& os) const
{
    os << nx << " " << nz << " "
       << 1.0/p_water << " " << 1.0/p_air << '\n';

    for (int i=1; i<=nx; i++) os << xpos(i) << " ";
    os << '\n';
    for (int i=1; i<=nx; i++) os << topo(i) << " ";
    os << '\n';
    for (int k=1; k<=nz; k++) os << zpos(k) << " ";
    os << '\n';
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
//	    os.precision(20);
	    os << 1.0/pgrid(i,k) << " ";
	}
	os << '\n';
    }
}

void SlownessMesh2d::outMeshVp(ostream& os) const
{
    if (!dual_vs){
	outMesh(os);
	return;
    }
    os << nx << " " << nz << " "
       << 1.0/p_water << " " << 1.0/p_air << '\n';
    for (int i=1; i<=nx; i++) os << xpos(i) << " ";
    os << '\n';
    for (int i=1; i<=nx; i++) os << topo(i) << " ";
    os << '\n';
    for (int k=1; k<=nz; k++) os << zpos(k) << " ";
    os << '\n';
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++) os << 1.0/pgrid_vp(i,k) << " ";
	os << '\n';
    }
}

bool SlownessMesh2d::in_water(const Point2d& pos, const Index2d& guess) const
{
    if (pos.y()<0.0) return false; // negative z means above horizon
    
    int i=guess.i();
    double r = (pos.x()-xpos(i))/dx_vec(i);
    double zsurf = topo(i)+b_vec(i)*r;
    if (pos.y()<zsurf){
	return true;
    }else{
	return false;
    }
}

bool SlownessMesh2d::in_air(const Point2d& pos, const Index2d& guess) const
{
    if (pos.y()>=0.0) return false; // positive z means below horizon
    
    int i=guess.i();
    double r = (pos.x()-xpos(i))/dx_vec(i);
    double zsurf = topo(i)+b_vec(i)*r;
    if (pos.y()<zsurf){
	return true;
    }else{
	return false;
    }
}

bool SlownessMesh2d::inWater(const Point2d& pos) const
{
    Index2d guess(1,1);
    upperleft(pos,guess);
    return in_water(pos,guess);
}

bool SlownessMesh2d::inAir(const Point2d& pos) const
{
    Index2d guess(1,1);
    upperleft(pos,guess);
    return in_air(pos,guess);
}

void SlownessMesh2d::printElements(ostream& os) const
{
    for (int i=1; i<nx; i++){
	for (int k=1; k<nz; k++){
	    os << ">\n";
	    os << xpos(i) << " " << zpos(k)+topo(i) << '\n';
	    os << xpos(i) << " " << zpos(k+1)+topo(i) << '\n';
	    os << xpos(i+1) << " " << zpos(k+1)+topo(i+1) << '\n';
	    os << xpos(i+1) << " " << zpos(k)+topo(i+1) << '\n';
	}
    }
}

void SlownessMesh2d::printVGrid(ostream& os, bool printAW) const
{
    double min_topo=0.0;
    for (int i=1; i<=nx; i++){
	if (topo(i)<min_topo) min_topo=topo(i);
    }
    min_topo -= 1.0;		// upper bound for courtesy grid making
				// (extra 1km)
    double dz=dz_vec(1)*0.5;
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
	    os << xpos(i) << " "
	       << zpos(k)+topo(i) << " "
	       << 1.0/pgrid(i,k) << '\n';
	}
	if (printAW){
	     // some extra grid points for grid file 
	     double z = topo(i)-dz;
	     while(z>=0){
		  os << xpos(i) << " "
		     << z << " " << 1.0/p_water << '\n';
		  z-=dz;
	     }
	     while(z>=min_topo){
		  os << xpos(i) << " "
		     << z << " " << 1.0/p_air << '\n';
		  z-=dz;
	     }
	}
    }
}

void SlownessMesh2d::printVGrid(ostream& os,
				double x0, double x1,
				double z0, double z1,
				double dx, double dz) const
{
    Index2d guess = nodeIndex(nearest(Point2d(x0,z0)));

    int nxx = int((x1-x0)/dx+1);
    int nzz = int((z1-z0)/dz+1);
//    cerr << nxx << " " << nzz << '\n';
    for (int ix=1; ix<=nxx; ix++){
	double x = x0+(ix-1)*dx;
	for (int iz=1; iz<=nzz; iz++){
	    double z = z0+(iz-1)*dz;
	    double p = at(Point2d(x,z),guess);
	    os << x << " " << z << " " << 1.0/p << '\n';;
	}
    }
}

void SlownessMesh2d::printMaskGrid(ostream& os,
				   const Array1d<int>& valid_node) const
{
    if (valid_node.size() != nnodes)
	error("SlownessMesh2d::printMaskGrid - size mismatch");

    os << nx << " " << nz << " "
       << 1.0/p_water << " " << 1.0/p_air << '\n';

    for (int i=1; i<=nx; i++) os << xpos(i) << " ";
    os << '\n';
    for (int i=1; i<=nx; i++) os << topo(i) << " ";
    os << '\n';
    for (int k=1; k<=nz; k++) os << zpos(k) << " ";
    os << '\n';
    int inode=1;
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
	    double val=0.0;
	    if (valid_node(inode)>0){
		val = 1.0/pgrid(i,k);
	    }
	    os << val << " ";
	    inode++;
	}
	os << '\n';
    }
}

// output DWS
void SlownessMesh2d::printMaskGrid(ostream& os,
				   const Array1d<double>& dws) const
{
    if (dws.size() != nnodes)
	error("SlownessMesh2d::printMaskGrid - size mismatch");

    int inode=1;
    for (int i=1; i<=nx; i++){
	double x=xpos(i), t=topo(i);
	for (int k=1; k<=nz; k++){
	    double z=zpos(k)+t;
	    os << x << " " << z << " " << dws(inode) << "\n";
	    inode++;
	}
    }
}


