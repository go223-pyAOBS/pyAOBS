/*
 * inverse.cc
 * 
 * Jun Korenaga, MIT/WHOI
 * January 1999
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <unordered_map>
#include <vector>
#include <map>
#include <utility>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include <geom.h>
#include <util.h>
#include "inverse.h"
#include "sparse_rect.h"
#include "lsqr.h"
#include "traveltime.h"
#ifdef _OPENMP
#include <omp.h>
#endif

// LSQR 迭代上限。TOMO2D_INV_LSQR_MAXITER>0 时取与 fallback 的较小值。
static int inv_lsqr_itermax(int fallback)
{
    int m = fallback;
    const char* p = getenv("TOMO2D_INV_LSQR_MAXITER");
    if (p && *p){
	const int v = atoi(p);
	if (v > 0 && (m <= 0 || v < m)) m = v;
    }
    return m;
}

static double inv_lsqr_atol(double fallback)
{
    const char* p = getenv("TOMO2D_INV_LSQR_ATOL");
    if (p && *p){
	const double v = atof(p);
	if (v > 0.0) return v;
    }
    return fallback;
}

static inline double inv_wall_seconds()
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return double(clock())/CLOCKS_PER_SEC;
#endif
}

static void inv_omp_progress_line(int iter, int done, int nsrc)
{
    /* 并行区内不用 iostream：cerr 非线程安全，管道下可导致 SIGSEGV */
    char buf[192];
    int n = std::snprintf(
	buf, sizeof(buf),
	"TomographicInversion2d::iter=%d ray tracing %d/%d sources (OMP)\n",
	iter, done, nsrc);
    if (n <= 0) return;
    if (n >= (int)sizeof(buf)) n = (int)sizeof(buf) - 1;
#ifdef _WIN32
    _write(2, buf, (unsigned)n);
#else
    ssize_t w = write(STDERR_FILENO, buf, (size_t)n);
    (void)w;
#endif
}

// 轻量监视通道：TOMO2D_INV_STATUS_JSONL=<path> 时每轮追加一行 JSON
// （供 GUI 准实时读取；不替代 -L 日志）
static void append_inv_status_jsonl(int iter, int iset,
				    double rms_tot, double chi_tot,
				    double pred_chi, double dv_norm, double dd_norm,
				    double Lmvh, double Lmvv, double Lmd,
				    int is_final, const char* out_root)
{
    const char* path = getenv("TOMO2D_INV_STATUS_JSONL");
    if (path == 0 || path[0] == '\0') return;
    ofstream os(path, ios::app);
    if (!os) return;
    os.setf(ios::scientific);
    os.precision(10);
    os << "{\"iter\":" << iter
       << ",\"iset\":" << iset
       << ",\"rms\":" << rms_tot
       << ",\"chi2\":" << chi_tot
       << ",\"pred_chi\":" << pred_chi
       << ",\"dv_norm\":" << dv_norm
       << ",\"dd_norm\":" << dd_norm
       << ",\"rough_v\":" << (Lmvh + Lmvv)
       << ",\"rough_d\":" << Lmd
       << ",\"is_final\":" << (is_final ? "true" : "false");
    if (out_root != 0 && out_root[0] != '\0') {
	os << ",\"smesh\":\"" << out_root << ".smesh." << iter << "." << iset << "\"";
    }
    os << "}\n";
    os.flush();
}

//
// TomographicInversion2d
//
TomographicInversion2d::TomographicInversion2d(SlownessMesh2d& m, const char* datafn,
					       int xorder, int zorder, double crit_len,
					       int nintp, double cg_tol, double br_tol)

    : smesh(m), graph(smesh,xorder,zorder),
      betasp(1,0,nintp), bend(smesh,betasp,cg_tol,br_tol),
      nrefl(0), nnoded(0), do_full_refl(false),do_convert(false), freeze_refl(false),
      invert_water_only(false), invert_crust_only(false),
      invert_vs_psx(false), invert_joint_vpvs(false), joint_staged(false), joint_vp_only_iters(0),
      freeze_psx_lid(false), freeze_below(false), pss_below_only(false), vpvs_kappa(0.0), vpvs_kappa_below(0.0),
      do_ttdiff(false), ttdiff_skip_pss_abs(false), ttdiff_pps_only(false), ndata_abs(0),
      do_strategy(false), strategy_force(false), strategy_full_vp(false),
      strategy_skip_update(false), strategy_rescale(false), strategy_stage(0),
      strategy_dstar(0.0), strategy_n_pick(0), strategy_n_corr(0),
      nnodev(smesh.numNodes()), nx(smesh.Nx()), nz(smesh.Nz()),
      itermax_LSQR(nnodev*100), LSQR_ATOL(1e-3), 
      smooth_velocity(false), logscale_vel(false), do_filter(false),
      wsv_min(0.0), wsv_max(0.0), dwsv(1.0), wsv_vs(-1.0),
      weight_s_v(0.0), weight_s_vs(0.0),
      corr_vel_p(0), uboundp(0),
      smooth_depth(false), logscale_dep(false), wsd_min(0.0),
      wsd_max(0.0), dwsd(1.0), weight_s_d(0.0), corr_dep_p(0),
      damp_velocity(false), damp_depth(false),
      target_dv(0.0), target_dd(0.0),
      damping_is_fixed(false), weight_d_v(0.0), weight_d_d(0.0),
      do_squeezing(false), damping_wt_p(0),
      jumping(false),
      robust(false), crit_chi(1e3), refl_weight(1.0), target_chisq(0.0),
      sens_weight(false), sens_kappa(10.0), sens_eps(0.05),
      line_search(false), ls_c(1e-4), ls_rho(0.5), ls_amin(0.03125),
      use_lm(false), lm_lambda(1.0), lm_up(4.0), lm_down(0.5),
      lm_lmin(1.0), lm_lmax(256.0), lm_rho_accept(0.1), lm_rho_good(0.5),
      printLog(false), verbose_level(-1),
      printFinal(false), printTransient(false), out_mask(false), out_level(0),
      out_root(0), dump_iter(0), dump_iset(0), dump_is_final(false),
      gravity(false), out_grav_dws(false), ginv(0), ngravdata(0),
      log_os_p(0), vmesh_os_p(0), grav_dws_osp(0), reflp(0), convp(0), seafloorp(0), dump_os_p(0)
{
    if (crit_len>0) graph.refineIfLong(crit_len);

    // seafloor + surface + Moho (raytype 4/5); 0/1 still use at most two.
    start_i.reserve(3); end_i.reserve(3);
    interp.reserve(3);
    bathyp = new Interface2d(smesh);
    
    read_file(datafn);

    const int max_np = int(2*sqrt(float(nnodev)));
    path.reserve(max_np);
    pp.reserve(max_np+4);
    
    A.resize(ndata); 
    Rv_h.resize(nnodev); Rv_v.resize(nnodev);
    Tv.resize(nnodev);
    Rv_h_vs.resize(nnodev); Rv_v_vs.resize(nnodev);
    Tv_vs.resize(nnodev);
    data_vec.resize(ndata);
    tmp_data.resize(ndata);
    total_data_vec.reserve(2*ndata+3*nnodev+2*max_np);
    r_dt_vec.resize(ndata);
    int idata=1;
    for (int i=1; i<=obs_dt.size(); i++){
	for (int j=1; j<=obs_dt(i).size(); j++){
	    r_dt_vec(idata) = 1.0/obs_dt(i)(j);
	    idata++;
	}
    }
    modelv.resize(nnodev);
    mvscale.resize(nnodev);
    dmodel_total.resize(nnodev);
    nnode_total = nnodev;
    path_length.resize(ndata);

    Q.resize(nintp);

    nodev_hit.resize(nnodev); 
    tmp_node.resize(nnodev);
    tmp_nodev.resize(nnodev);
}

void TomographicInversion2d::read_file(const char* datafn)
{
    ifstream in(datafn);
    if (!in){
	cerr << "TomographicInversion2d::cannot open " << datafn << "\n";
	exit(1);
    }

    int iline=0;

    int nsrc;
    in >> nsrc; iline++;
    if (nsrc<=0) error("TomographicInversion2d::invalid nsrc");
    src.resize(nsrc); rcv.resize(nsrc);
    raytype.resize(nsrc); obs_ttime.resize(nsrc); obs_dt.resize(nsrc);
    res_ttime.resize(nsrc);
    
    int isrc=0;
    while(in){
	char flag;
	double x, y;
	int nrcv;

	in >> flag >> x >> y >> nrcv; iline++;
	if (flag!='s'){
	    cerr << "TomographicInversion2d::bad input (s) at l."
		 << iline << '\n';
	    exit(1);
	}
	isrc++;
	src(isrc).set(x,y);

	rcv(isrc).resize(nrcv); raytype(isrc).resize(nrcv);
	obs_ttime(isrc).resize(nrcv); obs_dt(isrc).resize(nrcv);
	res_ttime(isrc).resize(nrcv);
	for (int ircv=1; ircv<=nrcv; ircv++){
	    int n;
	    double t_val, dt_val;
	    in >> flag >> x >> y >> n >> t_val >> dt_val; iline++;
	    if (flag!='r'){
		cerr << "TomographicInversion2d::bad input (r) at l."
		     << iline << '\n';
		exit(1);
	    }
	    if (dt_val < 1e-6){
		cerr << "TomographicInversion2d::bad input (zero obs_dt) at l."
		     << iline << '\n';
		exit(1);
	    }
	    rcv(isrc)(ircv).set(x,y);
	    raytype(isrc)(ircv) = n;
	    obs_ttime(isrc)(ircv) = t_val;
	    obs_dt(isrc)(ircv) = dt_val;
	}
	if (isrc==nsrc) break;
    }
    if (isrc != nsrc) error("TomographicInversion2d::mismatch in nsrc");

    ndata = 0;
    for (int isrc=1; isrc<=nsrc; isrc++){
	ndata += rcv(isrc).size();
    }
}

void TomographicInversion2d::doRobust(double a)
{
    if (a>0){
	robust = true;
	crit_chi = a;
    }
}

void TomographicInversion2d::removeOutliers()
{
    Array1d<double> Adm(ndata_valid);
    SparseRectangular sparseA(A,tmp_data,tmp_node,nnode_total);
    sparseA.Ax(dmodel_total,Adm);
    Array1d<int> tmp_data2(tmp_data.size());

    tmp_data2 = 0;
    int ndata_valid2=0;
    char fn[MaxStr];
    fn[0] = '\0';
    const bool dump = (out_root != 0 && out_root[0] != '\0');

    int isrc=1, ircv=1;
    int nrej=0;
    std::vector<int> rej_isrc, rej_ircv, rej_ray;
    std::vector<double> rej_sx, rej_rx, rej_tres, rej_lin;
    for (int i=1; i<=tmp_data.size(); i++){
	int j=tmp_data(i);
	if (j>0){
	    double lin = Adm(j)-data_vec(i);
	    if (abs(lin)<=crit_chi){
		tmp_data2(i) = ++ndata_valid2;
	    }else{
		nrej++;
		if (dump && isrc<=src.size() && ircv<=rcv(isrc).size()){
		    rej_isrc.push_back(isrc);
		    rej_ircv.push_back(ircv);
		    rej_sx.push_back(src(isrc).x());
		    rej_rx.push_back(rcv(isrc)(ircv).x());
		    rej_ray.push_back(raytype(isrc)(ircv));
		    rej_tres.push_back(res_ttime(isrc)(ircv));
		    rej_lin.push_back(lin);
		}
	    }
	}
	ircv++;
	if (isrc<=src.size() && ircv>rcv(isrc).size()){
	    isrc++;
	    ircv=1;
	}
    }
    if (dump && nrej>0){
	sprintf(fn, "%s.outliers.%d.%d", out_root, dump_iter, dump_iset);
	ofstream os_out(fn);
	if (os_out){
	    os_out << "# isrc ircv src_x rcv_x raytype tres_s lin_res\n";
	    os_out << "# lin_res=(A*dm-d) after first LSQR; drop if |lin_res|>crit_chi="
		   << crit_chi << "\n";
	    for (size_t k=0; k<rej_isrc.size(); k++){
		os_out << rej_isrc[k] << " " << rej_ircv[k] << " "
		       << rej_sx[k] << " " << rej_rx[k] << " "
		       << rej_ray[k] << " " << rej_tres[k] << " "
		       << rej_lin[k] << '\n';
	    }
	}
	if (dump_is_final){
	    char fnf[MaxStr];
	    sprintf(fnf, "%s.outliers.final", out_root);
	    ofstream os_final(fnf);
	    if (os_final){
		os_final << "# isrc ircv src_x rcv_x raytype tres_s lin_res\n";
		os_final << "# last-iter copy; crit_chi=" << crit_chi << "\n";
		for (size_t k=0; k<rej_isrc.size(); k++){
		    os_final << rej_isrc[k] << " " << rej_ircv[k] << " "
			     << rej_sx[k] << " " << rej_rx[k] << " "
			     << rej_ray[k] << " " << rej_tres[k] << " "
			     << rej_lin[k] << '\n';
		}
	    }
	}
    }
    if (printLog && log_os_p){
	*log_os_p << "# outliers iter=" << dump_iter << " iset=" << dump_iset
		  << " n=" << nrej << " crit_chi=" << crit_chi;
	if (fn[0]) *log_os_p << " file=" << fn;
	*log_os_p << '\n';
    }

    // redefine data node vector and related stats
    ndata_valid = ndata_valid2;
    tmp_data = tmp_data2;
    rms_tres[0]=rms_tres[1]=0;
    init_chi[0]=init_chi[1]=0;
    ndata_in[0]=ndata_in[1]=0;
    double sum_res2=0, sum_chi=0;
    int idata=1;
    for (int isrc=1; isrc<=src.size(); isrc++){
	for (int ircv=1; ircv<=rcv(isrc).size(); ircv++){
	    if (tmp_data(idata)>0){
		double res=res_ttime(isrc)(ircv);
		int icode = raytype(isrc)(ircv);
		double res2=res*r_dt_vec(idata);
		double res22=res2*res2;
		if (icode==0 || icode==1){
		    rms_tres[icode] += res*res;
		    init_chi[icode] += res22;
		    ++ndata_in[icode];
		}
		sum_res2 += res*res;
		sum_chi += res22;
	    }
	    idata++;
	}
    }
    rms_tres_total = sqrt(sum_res2/ndata_valid);
    init_chi_total = sum_chi/ndata_valid;
    for (int i=0; i<=1; i++){
	rms_tres[i] = ndata_in[i]>0 ? sqrt(rms_tres[i]/ndata_in[i]) : 0.0;
	init_chi[i] = ndata_in[i]>0 ? init_chi[i]/ndata_in[i] : 0.0;
    }
}

void TomographicInversion2d::setLSQR_TOL(double a)
{
    if (a<=0){
	cerr << "TomographicInversion2d::setLSQR_TOL - invalid TOL (ignored)\n";
	return;
    }
    LSQR_ATOL = a;
}

void TomographicInversion2d::setLogfile(const char* fn)
{
    printLog = true;
    log_os_p = new ofstream(fn);
    if (!(*log_os_p)){
	cerr << "TomographicInversion2d::setLogfile - can't open "
	     << fn << '\n';
	exit(1);
    }
}

void TomographicInversion2d::setVerbose(int i){ verbose_level=i; }

void TomographicInversion2d::outStepwise(const char* fn, int i)
{
    printTransient = true;
    out_root = fn;
    out_level = i;
}

void TomographicInversion2d::outFinal(const char* fn, int i)
{
    printFinal = true;
    out_root = fn;
    out_level = i;
}

void TomographicInversion2d::targetChisq(double c)
{
    target_chisq = c;
}

void TomographicInversion2d::outMask(const char* fn)
{
    out_mask = true;
    vmesh_os_p = new ofstream(fn);
    if (!(*vmesh_os_p)){
	cerr << "TomographicInversion2d::outMask - can't open "
	     << fn << '\n';
	exit(1);
    }
}

void TomographicInversion2d::addonGravity(double _z, const Array1d<double>& _x,
					  const Array1d<double>& _g,
					  AddonGravityInversion2d* p, double w,
					  const char *dwsfn)
{
    out_grav_dws = true;
    grav_dws_osp = new ofstream(dwsfn);
    if (!(*grav_dws_osp)){
	cerr << "TomographicInversion2d::addonGravity - can't open "
	     << dwsfn << '\n';
	exit(1);
    }
    addonGravity(_z,_x,_g,p,w);
}

void TomographicInversion2d::addonGravity(double _z, const Array1d<double>& _x,
					  const Array1d<double>& _g,
					  AddonGravityInversion2d* p, double w)
{
    gravity = true;
    grav_z0 = _z;
    ngravdata = _x.size();
    grav_x.resize(ngravdata); grav_x = _x;
    obs_grav.resize(ngravdata); obs_grav = _g;
    res_grav.resize(ngravdata);
    B.resize(ngravdata);
    tmp_gravdata.resize(ngravdata);
    for (int i=1; i<=ngravdata; i++) tmp_gravdata(i) = i;
    ginv = p;
    weight_grav = w;
    if (w<0){
	error("TomographicInversion2d::addonGravity - negative weight detected.");
    }
}

void TomographicInversion2d::enableTraveltimeDiff()
{
    do_ttdiff = true;
}

void TomographicInversion2d::enableFreezeLid()
{
    freeze_psx_lid = true;
}

void TomographicInversion2d::enablePssBelowOnly()
{
    pss_below_only = true;
}

void TomographicInversion2d::enableFreezeBelow()
{
    freeze_below = true;
}

void TomographicInversion2d::enableStrategy()
{
    strategy_force = true;
}

void TomographicInversion2d::setupTraveltimeDiffRows()
{
    if (ndata_abs > 0) return;
    ndata_abs = ndata;
    std::vector<TtDiffPair> tmp;
    const int nsrc = src.size();
    Array1d<int> gid0(nsrc);
    int cursor = 1;
    for (int isrc=1; isrc<=nsrc; isrc++){
	gid0(isrc) = cursor;
	cursor += rcv(isrc).size();
    }
    const int pairs[3][2] = {{6,0},{7,0},{8,6}};
    for (int isrc=1; isrc<=nsrc; isrc++){
	std::map<int,int> ix[9];
	for (int ircv=1; ircv<=rcv(isrc).size(); ircv++){
	    int c = raytype(isrc)(ircv);
	    if (c<0 || c>8) continue;
	    int xk = int(floor(rcv(isrc)(ircv).x()*1000.0+0.5));
	    ix[c][xk] = ircv;
	}
	for (int ip=0; ip<3; ip++){
	    int ca = pairs[ip][0], cb = pairs[ip][1];
	    if (ttdiff_pps_only && !(ca==7 && cb==0)) continue;
	    for (std::map<int,int>::const_iterator p=ix[ca].begin();
		 p!=ix[ca].end(); ++p){
		std::map<int,int>::const_iterator q = ix[cb].find(p->first);
		if (q==ix[cb].end()) continue;
		TtDiffPair d;
		d.isrc = isrc;
		d.ia = p->second;
		d.ib = q->second;
		d.gid_a = gid0(isrc) + d.ia - 1;
		d.gid_b = gid0(isrc) + d.ib - 1;
		tmp.push_back(d);
	    }
	}
    }
    ttdiff_pairs.resize(int(tmp.size()));
    for (int i=0; i<int(tmp.size()); i++) ttdiff_pairs(i+1) = tmp[i];
    ndata = ndata_abs + ttdiff_pairs.size();
    A.resize(ndata);
    data_vec.resize(ndata);
    tmp_data.resize(ndata);
    r_dt_vec.resize(ndata);
    path_length.resize(ndata);
    if (verbose_level>=0)
	cerr << "[tt_inverse] ttdiff pairs=" << ttdiff_pairs.size()
	     << (ttdiff_pps_only ? "  PPS-PPP only" : "  PSP-PPP + PPS-PPP + PSS-PSP")
	     << (ttdiff_skip_pss_abs ? "  skip abs PSS\n" : "  abs PSS on\n");
}

static int strategy_xkey(double x)
{
    return int(floor(x*1000.0+0.5));
}

static double strategy_rms(const std::vector<double>& a)
{
    if (a.empty()) return 0.0;
    double s = 0.0;
    for (size_t i=0; i<a.size(); i++) s += a[i]*a[i];
    return sqrt(s/double(a.size()));
}

static double strategy_pct(std::vector<double> xs, double q)
{
    if (xs.empty()) return 0.0;
    std::sort(xs.begin(), xs.end());
    if (xs.size()==1) return xs[0];
    const double t = (double(xs.size())-1.0)*q/100.0;
    const int i = int(t);
    const double f = t-double(i);
    if (i+1 >= int(xs.size())) return xs.back();
    return xs[i]*(1.0-f) + xs[i+1]*f;
}

void TomographicInversion2d::strategyRebuildDataArrays()
{
    ndata = 0;
    for (int i=1; i<=rcv.size(); i++) ndata += int(rcv(i).size());
    ndata_abs = 0;
    A.resize(ndata);
    data_vec.resize(ndata);
    tmp_data.resize(ndata);
    r_dt_vec.resize(ndata);
    path_length.resize(ndata);
    int idata=1;
    for (int i=1; i<=obs_dt.size(); i++){
	for (int j=1; j<=obs_dt(i).size(); j++){
	    const double dt = obs_dt(i)(j);
	    r_dt_vec(idata) = (dt>0.0) ? 1.0/dt : 1.0/0.05;
	    idata++;
	}
    }
}

void TomographicInversion2d::strategyEnsurePspSlots()
{
    ray_use.resize(src.size());
    ray_kind.resize(src.size());
    int nadd = 0;
    for (int is=1; is<=src.size(); is++){
	std::map<int,int> has6;
	std::vector<int> pss_ir;
	for (int ir=1; ir<=rcv(is).size(); ir++){
	    const int c = raytype(is)(ir);
	    const int xk = strategy_xkey(rcv(is)(ir).x());
	    if (c==6) has6[xk] = ir;
	    if (c==8) pss_ir.push_back(ir);
	}
	std::vector<int> add;
	for (size_t i=0; i<pss_ir.size(); i++){
	    const int ir = pss_ir[i];
	    const int xk = strategy_xkey(rcv(is)(ir).x());
	    if (has6.find(xk)==has6.end()) add.push_back(ir);
	}
	const int n0 = int(rcv(is).size());
	const int n1 = n0 + int(add.size());
	if (!add.empty()){
	    rcv(is).resize(n1);
	    raytype(is).resize(n1);
	    obs_ttime(is).resize(n1);
	    obs_dt(is).resize(n1);
	    res_ttime(is).resize(n1);
	    for (int j=0; j<int(add.size()); j++){
		const int ir = add[j];
		const int in = n0+j+1;
		rcv(is)(in) = rcv(is)(ir);
		raytype(is)(in) = 6;
		obs_ttime(is)(in) = 0.0;
		obs_dt(is)(in) = 0.05;
		res_ttime(is)(in) = 0.0;
	    }
	    nadd += int(add.size());
	}
	ray_use(is).resize(rcv(is).size());
	ray_kind(is).resize(rcv(is).size());
	for (int ir=1; ir<=rcv(is).size(); ir++){
	    ray_use(is)(ir) = 1;
	    ray_kind(is)(ir) = (ir>n0) ? 1 : 0;
	}
    }
    if (nadd>0) strategyRebuildDataArrays();
    cerr << "[tt_inverse] strategy: added " << nadd
	 << " PSP placeholders for PSS-only pairs\n";
}

void TomographicInversion2d::strategySetRayUseByCode(int c0, int c1, int c2)
{
    for (int is=1; is<=raytype.size(); is++){
	if (ray_use(is).size() != raytype(is).size())
	    ray_use(is).resize(raytype(is).size());
	for (int ir=1; ir<=raytype(is).size(); ir++){
	    const int c = raytype(is)(ir);
	    ray_use(is)(ir) = (c==c0 || c==c1 || c==c2) ? 1 : 0;
	}
    }
}

bool TomographicInversion2d::strategyRayOn(int isrc, int ircv) const
{
    if (!do_strategy) return true;
    if (isrc<1 || isrc>int(ray_use.size())) return true;
    if (ircv<1 || ircv>int(ray_use(isrc).size())) return true;
    return ray_use(isrc)(ircv) != 0;
}

void TomographicInversion2d::configureStrategyStage(int stage)
{
    strategy_stage = stage;
    strategy_rescale = true;
    do_convert = false;
    invert_joint_vpvs = false;
    joint_vp_only_iters = 0;
    if (stage==1){
	strategy_full_vp = true;
	invert_vs_psx = false;
	do_ttdiff = false;
	freeze_psx_lid = false;
	freeze_below = false;
	strategySetRayUseByCode(0, 1);
    }else if (stage==2){
	strategy_full_vp = false;
	invert_vs_psx = true;
	do_ttdiff = true;
	ttdiff_pps_only = true;
	ttdiff_skip_pss_abs = true;
	freeze_psx_lid = false;
	freeze_below = false;
	strategySetRayUseByCode(0, 7);
	if (ndata_abs==0) setupTraveltimeDiffRows();
    }else if (stage==3){
	strategy_full_vp = false;
	invert_vs_psx = true;
	do_ttdiff = false;
	freeze_psx_lid = true;
	freeze_below = false;
    }
}

void TomographicInversion2d::strategyAfterPpp()
{
    const double k_lid = vpvs_kappa>0.0 ? vpvs_kappa : 1.73;
    const double k_bel = vpvs_kappa_below>0.0 ? vpvs_kappa_below : k_lid;
    if (!smesh.hasDualVs()){
	if (vpvs_kappa<=0.0)
	    error("TomographicInversion2d:: strategy needs -k or -U after PPP");
	smesh.enableDualVs(k_lid, k_bel, convp);
	cerr << "[tt_inverse] strategy: enable dual Vs = Vp/k after PPP"
	     << " lid=" << k_lid << " below=" << k_bel << "\n";
    }else{
	smesh.resetVsFromVp(k_lid, k_bel, convp);
	cerr << "[tt_inverse] strategy: reset Vs = rec_Vp/k after PPP"
	     << " lid=" << k_lid << " below=" << k_bel << "\n";
    }
    if (nrefl>0 && !freeze_refl){
	freezeRefl();
	cerr << "[tt_inverse] strategy: freeze Moho after PPP+PmP\n";
    }
    if (out_root && out_root[0]!='\0'){
	char fn[MaxStr];
	sprintf(fn, "%s.vp.smesh", out_root);
	ofstream os(fn);
	if (os) smesh.outMeshVp(os);
    }
}

void TomographicInversion2d::applyStrategyCorr()
{
    double thresh = 0.06;
    int min_far = 8;
    double sig_floor = 0.05;
    if (const char* p = getenv("STRATEGY_DSTAR_RMS")) thresh = atof(p);
    if (const char* p = getenv("STRATEGY_DSTAR_MINN")) min_far = atoi(p);
    if (const char* p = getenv("STRATEGY_CORR_SIG")) sig_floor = atof(p);
    if (thresh<=0.0) thresh = 0.06;
    if (min_far<1) min_far = 8;
    if (sig_floor<=0.0) sig_floor = 0.05;

    std::vector<std::pair<double,double> > overlap;
    std::vector<double> pss_dx;
    strategy_n_pick = 0;
    strategy_n_corr = 0;
    int n_skip = 0;

    struct Row {
	int is, ir6, ir8;
	double dx, hat, tsyn6, tsyn8;
	bool pick, have_corr;
    };
    std::vector<Row> rows;

    for (int is=1; is<=src.size(); is++){
	std::map<int,int> i6, i8;
	for (int ir=1; ir<=rcv(is).size(); ir++){
	    const int c = raytype(is)(ir);
	    const int xk = strategy_xkey(rcv(is)(ir).x());
	    if (c==6) i6[xk] = ir;
	    if (c==8) i8[xk] = ir;
	}
	for (std::map<int,int>::const_iterator p=i6.begin(); p!=i6.end(); ++p){
	    const int ir6 = p->second;
	    const int ir8 = (i8.find(p->first)==i8.end()) ? 0 : i8[p->first];
	    const double dx = abs(rcv(is)(ir6).x()-src(is).x());
	    if (dx<=1e-6) continue;
	    Row row;
	    row.is = is; row.ir6 = ir6; row.ir8 = ir8; row.dx = dx;
	    row.hat = 0.0; row.tsyn6 = 0.0; row.tsyn8 = 0.0;
	    row.pick = (int(ray_kind(is).size())>=ir6 && ray_kind(is)(ir6)==0);
	    row.have_corr = false;
	    if (ir8>0){
		row.tsyn6 = obs_ttime(is)(ir6) - res_ttime(is)(ir6);
		row.tsyn8 = obs_ttime(is)(ir8) - res_ttime(is)(ir8);
		row.hat = obs_ttime(is)(ir8) - (row.tsyn8 - row.tsyn6);
		row.have_corr = true;
		pss_dx.push_back(dx);
		if (row.pick) overlap.push_back(std::make_pair(dx, obs_ttime(is)(ir6)-row.hat));
	    }
	    rows.push_back(row);
	}
    }

    const char* how = "pss-q60";
    double dstar_rms = 0.0;
    int n_far = 0;
    double dstar = 0.0;
    if (int(overlap.size())>=8){
	std::vector<double> dxs;
	for (size_t i=0; i<overlap.size(); i++) dxs.push_back(overlap[i].first);
	std::sort(dxs.begin(), dxs.end());
	dxs.erase(std::unique(dxs.begin(), dxs.end()), dxs.end());
	bool found = false;
	for (size_t i=0; i<dxs.size(); i++){
	    std::vector<double> far;
	    for (size_t j=0; j<overlap.size(); j++)
		if (overlap[j].first+1e-9 >= dxs[i]) far.push_back(overlap[j].second);
	    if (int(far.size())<min_far) continue;
	    const double r = strategy_rms(far);
	    if (r<=thresh){
		dstar = dxs[i]; how = "overlap-rms"; dstar_rms = r;
		n_far = int(far.size());
		found = true;
		break;
	    }
	}
	if (!found){
	    std::vector<double> odx;
	    for (size_t i=0; i<overlap.size(); i++) odx.push_back(overlap[i].first);
	    dstar = strategy_pct(odx, 60.0);
	    how = "overlap-q60";
	    std::vector<double> far;
	    for (size_t j=0; j<overlap.size(); j++)
		if (overlap[j].first+1e-9 >= dstar) far.push_back(overlap[j].second);
	    dstar_rms = strategy_rms(far);
	    n_far = int(far.size());
	}
    }else{
	dstar = strategy_pct(pss_dx, 60.0);
	how = "pss-q60";
	dstar_rms = 0.0;
	for (size_t i=0; i<pss_dx.size(); i++)
	    if (pss_dx[i]+1e-9 >= dstar) n_far++;
    }
    strategy_dstar = dstar;

    std::vector<double> far_d;
    for (size_t i=0; i<overlap.size(); i++)
	if (overlap[i].first+1e-9 >= dstar) far_d.push_back(overlap[i].second);
    const double sig_corr = std::max(sig_floor, far_d.empty() ? sig_floor : strategy_rms(far_d));

    for (int is=1; is<=raytype.size(); is++)
	for (int ir=1; ir<=raytype(is).size(); ir++)
	    ray_use(is)(ir) = 0;

    for (size_t i=0; i<rows.size(); i++){
	const Row& r = rows[i];
	if (r.pick){
	    ray_use(r.is)(r.ir6) = 1;
	    strategy_n_pick++;
	}else if (r.have_corr && r.dx+1e-9 >= dstar){
	    obs_ttime(r.is)(r.ir6) = r.hat;
	    obs_dt(r.is)(r.ir6) = sig_corr;
	    ray_use(r.is)(r.ir6) = 1;
	    strategy_n_corr++;
	}else{
	    n_skip++;
	}
    }
    if (strategy_n_pick+strategy_n_corr==0){
	for (size_t i=0; i<rows.size(); i++){
	    const Row& r = rows[i];
	    if (!r.have_corr) continue;
	    obs_ttime(r.is)(r.ir6) = r.hat;
	    obs_dt(r.is)(r.ir6) = sig_corr;
	    ray_use(r.is)(r.ir6) = 1;
	    strategy_n_corr++;
	    n_skip--;
	}
	if (verbose_level>=0)
	    cerr << "[tt_inverse] strategy: no far/pick rows, keep all finite corr\n";
    }

    int idata=1;
    for (int is=1; is<=obs_dt.size(); is++){
	for (int ir=1; ir<=obs_dt(is).size(); ir++){
	    const double dt = obs_dt(is)(ir);
	    r_dt_vec(idata) = (dt>0.0) ? 1.0/dt : 1.0/sig_corr;
	    idata++;
	}
    }

    cerr << "[tt_inverse] strategy corr: d*=" << dstar << " km  rule=" << how
	 << "  overlap_n=" << overlap.size() << "  far_n=" << n_far
	 << "  d*_rms=" << dstar_rms << "  sig_corr=" << sig_corr
	 << "  n_pick=" << strategy_n_pick
	 << "  n_corr=" << strategy_n_corr
	 << "  n_skip=" << n_skip << "\n";
}

void TomographicInversion2d::solve(int niter)
{
    typedef map<int,double>::iterator mapIterator;
    
    const int bend_nfac=1;
    {
	bool has_psx=false, has78=false, has_other=false;
	bool has0=false, has7=false, has8=false, has1213=false;
	{
	    const char* tde = getenv("TOMO2D_INV_TTDIFF");
	    if (tde && atoi(tde)!=0) do_ttdiff = true;
	    // default: PSS absolute times enter A. Opt out:
	    // TOMO2D_INV_TTDIFF_SKIP_PSSABS=1
	    const char* skip_pss = getenv("TOMO2D_INV_TTDIFF_SKIP_PSSABS");
	    if (skip_pss && atoi(skip_pss)!=0) ttdiff_skip_pss_abs = true;
	    const char* pssbel = getenv("TOMO2D_INV_PSS_BELOW");
	    if (pssbel && atoi(pssbel)!=0) pss_below_only = true;
	    const char* fbel = getenv("TOMO2D_INV_FREEZE_BELOW");
	    if (fbel && atoi(fbel)!=0) freeze_below = true;
	    const char* swt = getenv("TOMO2D_INV_SENS_WEIGHT");
	    if (swt && atoi(swt)!=0) sens_weight = true;
	    const char* sk = getenv("TOMO2D_INV_SENS_KAPPA");
	    if (sk && atof(sk)>0.0) sens_kappa = atof(sk);
	    const char* seps = getenv("TOMO2D_INV_SENS_EPS");
	    if (seps && atof(seps)>0.0) sens_eps = atof(seps);
	    const char* lse = getenv("TOMO2D_INV_LINESEARCH");
	    if (lse && atoi(lse)!=0) line_search = true;
	    const char* lsc = getenv("TOMO2D_INV_LS_C");
	    if (lsc && atof(lsc)>0.0) ls_c = atof(lsc);
	    const char* lsr = getenv("TOMO2D_INV_LS_RHO");
	    if (lsr && atof(lsr)>0.0 && atof(lsr)<1.0) ls_rho = atof(lsr);
	    const char* lsa = getenv("TOMO2D_INV_LS_AMIN");
	    if (lsa && atof(lsa)>0.0 && atof(lsa)<=1.0) ls_amin = atof(lsa);
	    const char* lme = getenv("TOMO2D_INV_LM");
	    if (lme && atoi(lme)!=0) use_lm = true;
	    const char* lml = getenv("TOMO2D_INV_LM_LAMBDA");
	    if (lml && atof(lml)>0.0) lm_lambda = atof(lml);
	    const char* lmu = getenv("TOMO2D_INV_LM_UP");
	    if (lmu && atof(lmu)>1.0) lm_up = atof(lmu);
	    const char* lmdn = getenv("TOMO2D_INV_LM_DOWN");
	    if (lmdn && atof(lmdn)>0.0 && atof(lmdn)<1.0) lm_down = atof(lmdn);
	    const char* lmmax = getenv("TOMO2D_INV_LM_LMAX");
	    if (lmmax && atof(lmmax)>1.0) lm_lmax = atof(lmmax);
	    const char* lmra = getenv("TOMO2D_INV_LM_RHO_ACCEPT");
	    if (lmra && atof(lmra)>0.0 && atof(lmra)<1.0) lm_rho_accept = atof(lmra);
	    const char* lmrg = getenv("TOMO2D_INV_LM_RHO_GOOD");
	    if (lmrg && atof(lmrg)>0.0 && atof(lmrg)<=1.0) lm_rho_good = atof(lmrg);
	    if (use_lm && !damping_is_fixed && !damp_velocity && !damp_depth){
		use_lm = false;
		if (verbose_level>=0)
		    cerr << "[tt_inverse] lm disabled (no -D/-T damping to scale)\n";
	    }
	    if (use_lm && line_search){
		line_search = false;
		if (verbose_level>=0)
		    cerr << "[tt_inverse] lm ON: linesearch ignored (re-LSQR, not shrink step)\n";
	    }
	}
	for (int is=1; is<=src.size(); is++){
	    for (int ir=1; ir<=raytype(is).size(); ir++){
		int c = raytype(is)(ir);
		if (c==6 || c==7 || c==8 || c==9 || c==10 || c==11
		    || c==12 || c==13 || c==14 || c==15) has_psx=true;
		else has_other=true;
		if (c==7 || c==8 || c==10 || c==11 || c==13 || c==14 || c==15) has78=true;
		if (c==0) has0=true;
		if (c==7) has7=true;
		if (c==8) has8=true;
		if (c==12 || c==13) has1213=true;
	    }
	}
	{
	    bool want = strategy_force;
	    const char* se = getenv("TOMO2D_INV_STRATEGY");
	    if (se && atoi(se)==0) want = false;
	    else if (se && atoi(se)>0) want = true;
	    else if (!strategy_force && has0 && has7 && has8
		     && convp && seafloorp
		     && (vpvs_kappa>0.0 || smesh.hasDualVs()))
		want = true;
	    if (want){
		if (!has0 || !has7 || !has8)
		    error("TomographicInversion2d:: strategy needs raytype 0+7+8");
		if (convp==0 || seafloorp==0)
		    error("TomographicInversion2d:: strategy needs -B and -Y");
		if (vpvs_kappa<=0.0 && !smesh.hasDualVs())
		    error("TomographicInversion2d:: strategy needs -k or -U");
		do_strategy = true;
	    }
	}
	if (do_strategy){
	    invert_joint_vpvs = false;
	    joint_vp_only_iters = 0;
	    do_ttdiff = false;
	    invert_vs_psx = false;
	    freeze_psx_lid = false;
	    freeze_below = false;
	    do_convert = false;
	    strategyEnsurePspSlots();
	    configureStrategyStage(1);
	    cerr << "[tt_inverse] strategy ON: PPP+PmP(Vp) -> PPS+PPS-PPP(lid Vs)"
		 << " -> far PSS corr PSP -> freeze lid, below Vs\n"
		 << "  TOMO2D_INV_STRATEGY=0 disables; -ts forces\n";
	}else if (has_psx){
	    if (convp==0)
		error("TomographicInversion2d:: raytype 6/7/8/10/11/12/13/14/15 requires conversion interface (-B)");
	    if (seafloorp==0)
		error("TomographicInversion2d:: raytype 6/7/8/10/11/12/13/14/15 Vs inversion requires seafloor (-Y)");
	    if (has78 && !smesh.hasDualVs() && vpvs_kappa<=0.0)
		error("TomographicInversion2d:: single-field 7/8/10/11/13/14/15 needs -k<k> or -k<k_lid>/<k_below> (Vp→Vs)");
	    if (has1213 && (nrefl==0 || reflp==0))
		error("TomographicInversion2d:: raytype 12/13 requires Moho (-F)");
	    invert_vs_psx = true;
	    joint_staged = has_other && !do_ttdiff
		&& (smesh.hasDualVs() || vpvs_kappa>0.0);
	    invert_joint_vpvs = false;
	    freeze_psx_lid = !has78;
	    if (joint_staged) invert_vs_psx = false;
	}
	if (!do_strategy){
	    const char* flid = getenv("TOMO2D_INV_FREEZE_LID");
	    if (flid && atoi(flid)!=0) freeze_psx_lid = true;
	}
	if (has_psx && !do_strategy){
	    if (!smesh.hasDualVs() && vpvs_kappa>0.0 && !joint_staged){
		smesh.enableDualVs(vpvs_kappa,
				   vpvs_kappa_below>0.0 ? vpvs_kappa_below : vpvs_kappa,
				   convp);
		if (verbose_level>=0)
		    cerr << "TomographicInversion2d:: PSP/PPS/PSS Vs inversion: kappa lid="
			 << vpvs_kappa << " below="
			 << (vpvs_kappa_below>0.0 ? vpvs_kappa_below : vpvs_kappa)
			 << " (Vp frozen, Vs=Vp/k init; output smesh is Vs)\n";
	    }else if (joint_staged){
		do_convert = false;
		cerr << "[tt_inverse] joint: PPP first as single-field Vp LSQR "
		     << "(graph.solve; dual and Vs-only after Vp-only)\n";
	    }else if (verbose_level>=0){
		if (smesh.hasDualVs())
		    cerr << "TomographicInversion2d:: raytype 6/7/8: invert Vs\n";
		else
		    cerr << "TomographicInversion2d:: raytype 6 mixed fold: invert Vs below conv\n";
	    }
	    if (smesh.hasDualVs() && out_root && out_root[0]!='\0'){
		char fn[MaxStr];
		sprintf(fn, "%s.vp.smesh", out_root);
		ofstream os(fn);
		if (os) smesh.outMeshVp(os);
	    }
	    if (verbose_level>=0){
		if (joint_staged)
		    cerr << "TomographicInversion2d:: joint staged: each stage "
			 << nnodev << " unknowns (Vp then Vs, never [Vp|Vs])\n";
		else if (freeze_psx_lid)
		    cerr << "TomographicInversion2d:: Vs lid frozen "
			 << "(kernel/smooth/damp/dm; PSS/PSP write below only)\n";
		else
		    cerr << "TomographicInversion2d:: PPS/PSS present "
			 << (pss_below_only
			     ? "(lid free via PPS; PSS kernel below only)\n"
			     : "(lid not frozen)\n");
	    }
	}else if (!do_strategy && vpvs_kappa>0.0 && verbose_level>=0){
		cerr << "TomographicInversion2d:: -k ignored (no raytype 6/7/8/9/10/11/12/13/14/15 in data)\n";
	}
	if (has1213 && verbose_level>=0)
	    cerr << "TomographicInversion2d:: raytype 12/13 S-Moho kernel "
		 << "(depth on -F with S slowness)\n";
	if (do_strategy){
	    // FREEZE_* / -td / joint staging are owned by strategy stages.
	}else if (joint_staged){
	    do_convert = false;
	}else if (freeze_below){
	    if (convp==0)
		error("TomographicInversion2d:: FREEZE_BELOW requires conversion interface (-B)");
	    do_convert = false;
	    if (verbose_level>=0)
		cerr << "TomographicInversion2d:: Vp below conv frozen "
		     << "(lid-only kernel/smooth/damp/dm; type 0 graph unchanged)\n";
	}else if (freeze_psx_lid && !invert_vs_psx){
	    if (convp==0)
		error("TomographicInversion2d:: FREEZE_LID (PPP below-only) requires conversion interface (-B)");
	    do_convert = false;
	    if (verbose_level>=0)
		cerr << "TomographicInversion2d:: Vp lid frozen "
		     << "(below-only kernel/smooth/damp/dm; type 0 graph unchanged)\n";
	}else if (freeze_psx_lid && convp){
	    do_convert = false;
	}
	if (wsv_vs>=0.0 && !invert_joint_vpvs && verbose_level>=0)
	    cerr << "TomographicInversion2d:: -Ss ignored (not joint Vp+Vs)\n";
	if (do_ttdiff){
	    if (!has_psx)
		error("TomographicInversion2d:: -td requires raytype 6/7/8");
	    invert_vs_psx = true;
	    setupTraveltimeDiffRows();
	    if (verbose_level>=0)
		cerr << "TomographicInversion2d:: ttdiff: freeze Vp, "
		     << "type 0/1 traced for residuals only"
		     << (ttdiff_skip_pss_abs ? ", skip abs PSS\n" : ", abs PSS on\n");
	}
	joint_vp_only_iters = 0;
	if (joint_staged){
	    // 先只改 Vp，避免 PSS 在 Vp 还错时把面下 Vs 打成贴面高速皮。
	    // 前段 min(8,I)，后段再跑 I 次 Vs（总迭代 min(8,I)+I）。
	    joint_vp_only_iters = (niter<8) ? niter : 8;
	    if (joint_vp_only_iters<1) joint_vp_only_iters = 1;
	    cerr << "[tt_inverse] joint staged: first " << joint_vp_only_iters
		 << " iter(s) Vp only, then " << niter << " Vs-only\n";
	}
    }

	if (printLog){
	    *log_os_p << "# strategy jumping=" << jumping
		      << " robust=" << robust << " crit_chi=" << crit_chi << '\n';
	    *log_os_p << "# ray_trace -N xorder=" << graph.xOrder()
		      << " zorder=" << graph.zOrder()
		      << " clen=" << graph.critLength()
		      << " nintp=" << betasp.numIntp()
		      << " bend_tol=" << bend.tolerance() << '\n';
	    *log_os_p << "# smooth_vel -SV on=" << int(smooth_velocity)
		      << " wmin=" << wsv_min << " wmax=" << wsv_max
		      << " dw=" << dwsv
		      << " -Ss=" << wsv_vs
		      << " log10(-XV)=" << int(logscale_vel) << '\n';
	    *log_os_p << "# smooth_dep -SD on=" << int(smooth_depth)
		      << " wmin=" << wsd_min << " wmax=" << wsd_max
		      << " dw=" << dwsd
		      << " log10(-XD)=" << int(logscale_dep) << '\n';
	    if (damping_is_fixed){
		*log_os_p << "# damping: MODE=fixed  using=-D/-DV/-DD  auto_-T=OFF"
			  << "  -DV=" << weight_d_v << " -DD=" << weight_d_d << '\n';
		if (do_squeezing){
		    *log_os_p << "# damping: -DQ=ON wdv=" << weight_d_v << '\n';
		}
	    }else if (damp_velocity || damp_depth){
		*log_os_p << "# damping: MODE=auto  using=-T/-TV/-TD  fixed_-D=OFF"
			  << "  -TV_percent=" << (target_dv * 100.0)
			  << " -TV_frac=" << target_dv
			  << "  -TD_percent=" << (target_dd * 100.0)
			  << " -TD_frac=" << target_dd << '\n';
	    }else{
		*log_os_p << "# damping: MODE=none  (neither -T nor -D)\n";
	    }
	    *log_os_p << "# filter_-s: " << (do_filter ? "ON" : "OFF")
		      << "  (ON=2D filter after each iter; OFF=no -s)\n";
	    *log_os_p << "# water: invert_only=" << int(invert_water_only)
		      << " crust_only=" << int(invert_crust_only)
		      << " seafloor=" << (seafloorp ? "Y" : (reflp ? "F" : "none"))
		      << '\n';
	    *log_os_p << "# psx: invert_vs=" << int(invert_vs_psx)
		      << " joint_vpvs=" << int(invert_joint_vpvs)
		      << " freeze_lid=" << int(freeze_psx_lid) << '\n';
	    if (do_strategy)
		*log_os_p << "# inv_strategy=ON  PPP+PmP->lid Vs->corr PSP->below Vs"
			  << "  (TOMO2D_INV_STRATEGY=0 off; -ts force)\n";
	    *log_os_p << flush;
	    *log_os_p << "# ndata " << ndata << '\n';
	    if (gravity){
		*log_os_p << "# grav_data " << obs_grav.size() << " " << weight_grav << '\n';
	    }
	    *log_os_p << "# nnodes nnodev=" << nnodev
		      << " nvel=" << (invert_joint_vpvs ? 2*nnodev : nnodev)
		      << " nnoded(refl)=" << nnoded
		      << (freeze_refl ? " freeze_-u=ON" : " freeze_-u=OFF")
		      << " refl_weight=" << refl_weight << '\n';
	    *log_os_p << "# LSQR atol=" << LSQR_ATOL
		      << " itermax=" << itermax_LSQR
		      << " cap=" << inv_lsqr_itermax(itermax_LSQR)
		      << "  (TOMO2D_INV_LSQR_MAXITER)\n";
	    {
		bool precond_on = true;
		double precond_max = 10.0;
		int precond_miniter = 20;
		lsqrColPrecondConfig(precond_on, precond_max, precond_miniter);
		*log_os_p << "# lsqr_precond: " << (precond_on ? "ON" : "OFF")
			  << " maxD=" << precond_max
			  << "  (ON=TOMO2D_INV_LSQR_PRECOND!=0; default OFF; Legacy forces OFF;"
			  << " maxD=kappa vs col-norm median, empty D=0, miniter="
			  << precond_miniter << ")\n";
		*log_os_p << "# sens_weight: " << (sens_weight ? "ON" : "OFF")
			  << " kappa=" << sens_kappa << " eps=" << sens_eps
			  << "  (ON=TOMO2D_INV_SENS_WEIGHT!=0; default OFF;"
			  << " T*=w, R couple 2/(wi+wj); median per Vp/Vs-lid/Vs-below/Moho)\n";
		*log_os_p << "# linesearch: " << (line_search ? "ON" : "OFF")
			  << " c=" << ls_c << " rho=" << ls_rho << " amin=" << ls_amin
			  << "  (ON=TOMO2D_INV_LINESEARCH!=0; default OFF;"
			  << " Armijo on true chi2 after re-trace)\n";
		*log_os_p << "# lm: " << (use_lm ? "ON" : "OFF")
			  << " lambda=" << lm_lambda
			  << " up=" << lm_up << " down=" << lm_down
			  << " lmax=" << lm_lmax
			  << " rho_acc=" << lm_rho_accept
			  << " rho_good=" << lm_rho_good
			  << "  (ON=TOMO2D_INV_LM!=0; default OFF;"
			  << " trust-region: scale T and re-LSQR on true chi2)\n";
		bool log_legacy = false;
		const char* legacy_env = getenv("TOMO2D_INV_LEGACY_BASELINE");
		if (legacy_env && atoi(legacy_env)!=0) log_legacy = true;
		bool log_reuse = false;
		double log_reuse_th = 0.0;
		const char* reuse_env = getenv("TOMO2D_INV_REUSE_FORWARD");
		const char* thresh_env = getenv("TOMO2D_INV_REUSE_THRESH");
		if (reuse_env && atoi(reuse_env)!=0) log_reuse = true;
		if (thresh_env) log_reuse_th = atof(thresh_env);
		if (log_reuse_th<=0.0) log_reuse = false;
		if (log_legacy) log_reuse = false;
		bool log_c2f = false;
		const char* c2f_env = getenv("TOMO2D_INV_COARSE2FINE");
		if (c2f_env && atoi(c2f_env)!=0) log_c2f = true;
		if (log_legacy) log_c2f = false;
		*log_os_p << "# accel: reuse=" << (log_reuse ? "ON" : "OFF")
			  << " thresh=" << log_reuse_th
			  << " c2f=" << (log_c2f ? "ON" : "OFF")
			  << " legacy=" << (log_legacy ? "ON" : "OFF") << '\n';
	    }
	}
	// 同步打到 stderr，GUI 进程日志不打开 -L 也能看见用的 -T 还是 -D、是否 -s
	if (damping_is_fixed){
	    cerr << "[tt_inverse] damping MODE=fixed using=-D/-DV/-DD auto_-T=OFF"
		 << " -DV=" << weight_d_v << " -DD=" << weight_d_d << '\n';
	}else if (damp_velocity || damp_depth){
	    cerr << "[tt_inverse] damping MODE=auto using=-T/-TV/-TD fixed_-D=OFF"
		 << " -TV_percent=" << (target_dv * 100.0)
		 << " -TV_frac=" << target_dv
		 << " -TD_percent=" << (target_dd * 100.0)
		 << " -TD_frac=" << target_dd << '\n';
	}else{
	    cerr << "[tt_inverse] damping MODE=none (neither -T nor -D)\n";
	}
	cerr << "[tt_inverse] filter_-s " << (do_filter ? "ON" : "OFF") << endl;
	if (sens_weight){
	    cerr << "[tt_inverse] sens_weight ON kappa=" << sens_kappa
		 << " eps=" << sens_eps
		 << "  (T*=w, R couple 2/(wi+wj);"
		 << " per-block Vp / Vs-lid / Vs-below / Moho;"
		 << " TOMO2D_INV_SENS_WEIGHT=0 off)\n";
	}
	if (line_search){
	    cerr << "[tt_inverse] linesearch ON c=" << ls_c
		 << " rho=" << ls_rho << " amin=" << ls_amin
		 << "  (Armijo on true chi2; TOMO2D_INV_LINESEARCH=0 off)\n";
	}
	if (use_lm){
	    cerr << "[tt_inverse] lm ON lambda=" << lm_lambda
		 << " up=" << lm_up << " down=" << lm_down
		 << " lmax=" << lm_lmax
		 << " rho_acc=" << lm_rho_accept
		 << " rho_good=" << lm_rho_good
		 << "  (trust-region re-LSQR; TOMO2D_INV_LM=0 off)\n";
	}
	if (invert_water_only){
	    cerr << "[tt_inverse] invert_water_only=-y (nodes below seafloor frozen)\n";
	}
	if (invert_crust_only){
	    cerr << "[tt_inverse] invert_crust_only=-w (nodes above seafloor frozen)\n";
	}
	{
	    const int lsqr_cap = inv_lsqr_itermax(itermax_LSQR);
	    if (lsqr_cap < itermax_LSQR)
		cerr << "[tt_inverse] LSQR itermax cap=" << lsqr_cap
		     << " (TOMO2D_INV_LSQR_MAXITER; default " << itermax_LSQR << ")\n";
	}
	if (convp){
	    if (do_strategy){
		cerr << "[tt_inverse] conversion -B: strategy pin only "
		     << "(PPP ignores conv mask; type 0 graph unchanged)\n";
	    }else if (joint_staged){
		cerr << "[tt_inverse] conversion -B: joint PPP uses graph.solve "
		     << "(single-field Vp LSQR); Vs-only after unlock, lid "
		     << (freeze_psx_lid ? "frozen" : "free") << "\n";
	    }else if (invert_vs_psx && !freeze_psx_lid){
		cerr << "[tt_inverse] conversion -B: pin only "
		     << "(PSP/PPS/PSS Vs inversion; lid not frozen; "
		     << "smooth/damp split at conv)\n";
	    }else if (freeze_below){
		cerr << "[tt_inverse] conversion -B: nodes on/below conv frozen "
		     << "(lid-only Vp; type 0 graph unchanged)\n";
	    }else if (freeze_psx_lid && !invert_vs_psx){
		cerr << "[tt_inverse] conversion -B: nodes strictly above conv frozen "
		     << "(below-only Vp; type 0 graph unchanged)\n";
	    }else{
		cerr << "[tt_inverse] conversion -B: nodes strictly above conv frozen "
		     << "(data kernel, -SV/-TV, and dm)\n";
	    }
	}
	{
	    bool precond_on = true;
	    double precond_max = 10.0;
	    int precond_miniter = 20;
	    lsqrColPrecondConfig(precond_on, precond_max, precond_miniter);
	    cerr << "[tt_inverse] lsqr_precond " << (precond_on ? "ON" : "OFF")
		 << " maxD=" << precond_max
		 << " miniter=" << precond_miniter << endl;
	    bool log_legacy = false;
	    const char* legacy_env = getenv("TOMO2D_INV_LEGACY_BASELINE");
	    if (legacy_env && atoi(legacy_env)!=0) log_legacy = true;
	    bool log_reuse = false;
	    double log_reuse_th = 0.0;
	    const char* reuse_env = getenv("TOMO2D_INV_REUSE_FORWARD");
	    const char* thresh_env = getenv("TOMO2D_INV_REUSE_THRESH");
	    if (reuse_env && atoi(reuse_env)!=0) log_reuse = true;
	    if (thresh_env) log_reuse_th = atof(thresh_env);
	    if (log_reuse_th<=0.0) log_reuse = false;
	    if (log_legacy) log_reuse = false;
	    bool log_c2f = false;
	    const char* c2f_env = getenv("TOMO2D_INV_COARSE2FINE");
	    if (c2f_env && atoi(c2f_env)!=0) log_c2f = true;
	    if (log_legacy) log_c2f = false;
	    cerr << "[tt_inverse] accel reuse=" << (log_reuse ? "ON" : "OFF")
		 << " thresh=" << log_reuse_th
		 << " c2f=" << (log_c2f ? "ON" : "OFF")
		 << " legacy=" << (log_legacy ? "ON" : "OFF") << endl;
	}

    // construct index mappers
    if (invert_joint_vpvs){
	modelvp.resize(nnodev);
	mvscale_vp.resize(nnodev);
	tmp_nodev_vs.resize(nnodev);
	const int nvel = 2*nnodev;
	nnode_total = nvel + ((nrefl>0 && !freeze_refl) ? nnoded : 0);
	dmodel_total.resize(nnode_total);
	tmp_node.resize(nnode_total);
	if (itermax_LSQR < nnode_total*100)
	    itermax_LSQR = nnode_total*100;
    }
    for (int i=1; i<=nnodev; i++){
	tmp_node(i) = i;
	tmp_nodev(i) = i;
	if (invert_joint_vpvs) tmp_nodev_vs(i) = i+nnodev;
    }
    if (invert_joint_vpvs){
	for (int i=1; i<=nnodev; i++) tmp_node(i+nnodev) = i+nnodev;
    }
    if (nrefl>0 && !freeze_refl){
	const int nvel = nVelUnknowns();
	for (int i=1; i<=nnoded; i++){
	    int i_total = i+nvel;
	    tmp_node(i+nvel) = i_total;
	    tmp_nodedc(i) = i_total;
	    tmp_nodedr(i) = i;
	}
    }

    Array1d<int> kstart;
    Array1d<double> filtered_model, dx_vec, dxr_vec, dxn_vec;
    if (do_filter){
	filtered_model.resize(dmodel_total.size());
	dx_vec.resize(dmodel_total.size());
	dxr_vec.resize(dmodel_total.size());
	dxn_vec.resize(dmodel_total.size());
	kstart.resize(nx);
	for (int i=1; i<=nx; i++){
	    Point2d p = smesh.nodePos(smesh.nodeIndex(i,1));
	    double boundz = uboundp->z(p.x());
	    kstart(i) = 1;
	    for (int k=1; k<=nz; k++){
		if (smesh.nodePos(smesh.nodeIndex(i,k)).y()>boundz) break;
		kstart(i) = k;
	    }
	}
    }
    
    // pre-allocate temporary arrays
    if (jumping) dmodel_total_sum.resize(dmodel_total.size());

    bool enable_src_parallel = false;
    bool request_src_parallel = false;
    bool compiled_with_openmp = false;
#ifdef _OPENMP
    compiled_with_openmp = true;
    {
	const char* omp_env = getenv("TOMO2D_INV_OMP");
	request_src_parallel = (omp_env && atoi(omp_env)!=0);
	enable_src_parallel = request_src_parallel;
    }
#else
    {
	const char* omp_env = getenv("TOMO2D_INV_OMP");
	request_src_parallel = (omp_env && atoi(omp_env)!=0);
    }
#endif
    Array1d<int> idata_start(src.size());
    int idata_cursor = 1;
    for (int isrc=1; isrc<=src.size(); isrc++){
	idata_start(isrc) = idata_cursor;
	idata_cursor += rcv(isrc).size();
    }

    bool legacy_baseline = false;
    {
	const char* legacy_env = getenv("TOMO2D_INV_LEGACY_BASELINE");
	legacy_baseline = (legacy_env && atoi(legacy_env)!=0);
    }

    bool enable_forward_reuse = false;
    double forward_reuse_thresh = 0.0;
    {
	const char* reuse_env = getenv("TOMO2D_INV_REUSE_FORWARD");
	const char* thresh_env = getenv("TOMO2D_INV_REUSE_THRESH");
	enable_forward_reuse = (reuse_env && atoi(reuse_env)!=0);
	if (thresh_env){
	    forward_reuse_thresh = atof(thresh_env);
	}
	if (forward_reuse_thresh<=0.0){
	    enable_forward_reuse = false;
	}
	if (legacy_baseline){
	    enable_forward_reuse = false;
	}
    }
    Array1d<double> prev_forward_modelv(modelv.size());
    Array1d<double> prev_forward_modelvp;
    if (invert_joint_vpvs) prev_forward_modelvp.resize(nnodev);
    Array1d<double> prev_forward_modeld(modeld.size());
    Array1d<double> path_length_prev(path_length.size());
    sparseMat A_prev(A.size());
    Array1d< Array1d<double> > res_ttime_prev(res_ttime.size());
    for (int isrc=1; isrc<=res_ttime.size(); isrc++){
	res_ttime_prev(isrc).resize(res_ttime(isrc).size());
    }
    bool has_prev_forward = false;

    bool enable_c2f = false;
    double c2f_smooth_start = 3.0;
    double c2f_smooth_end = 1.0;
    double c2f_damp_start = 3.0;
    double c2f_damp_end = 1.0;
    {
	const char* c2f_env = getenv("TOMO2D_INV_COARSE2FINE");
	enable_c2f = (c2f_env && atoi(c2f_env)!=0);
	const char* s0_env = getenv("TOMO2D_INV_C2F_SMOOTH_START");
	const char* s1_env = getenv("TOMO2D_INV_C2F_SMOOTH_END");
	const char* d0_env = getenv("TOMO2D_INV_C2F_DAMP_START");
	const char* d1_env = getenv("TOMO2D_INV_C2F_DAMP_END");
	if (s0_env) c2f_smooth_start = atof(s0_env);
	if (s1_env) c2f_smooth_end = atof(s1_env);
	if (d0_env) c2f_damp_start = atof(d0_env);
	if (d1_env) c2f_damp_end = atof(d1_env);
	if (c2f_smooth_start<=0.0) c2f_smooth_start = 1.0;
	if (c2f_smooth_end<=0.0) c2f_smooth_end = 1.0;
	if (c2f_damp_start<=0.0) c2f_damp_start = 1.0;
	if (c2f_damp_end<=0.0) c2f_damp_end = 1.0;
	if (legacy_baseline){
	    enable_c2f = false;
	}
    }

    bool isFinal=false;
    int iter=1;
    const int niter_user = niter;
    int niter_loop = niter_user;
    int it_end1 = 0, it_end2 = 0, it_fwd = 0;
	if (do_strategy){
	    int n_ppp = (niter_user<8) ? niter_user : 8;
	    if (n_ppp<1) n_ppp = 1;
	    it_end1 = n_ppp;
	    it_end2 = n_ppp + niter_user;
	    it_fwd = it_end2 + 1;
	    niter_loop = it_fwd + niter_user;
	    cerr << "[tt_inverse] strategy iters: PPP+PmP=" << n_ppp
		 << " lid=" << niter_user
		 << " fwd+below=" << 1+niter_user
		 << "  total=" << niter_loop << "\n";
    }else if (joint_staged && joint_vp_only_iters>0){
	    niter_loop = joint_vp_only_iters + niter_user;
	    cerr << "[tt_inverse] joint iters: Vp=" << joint_vp_only_iters
		 << " Vs=" << niter_user
		 << "  total=" << niter_loop << "\n";
    }
    bool ls_eval = false;
    bool ls_restart = false;
    double ls_phi0 = 0.0, ls_slope = 0.0, ls_alpha = 1.0;
    int ls_ntry = 0;
    Array1d<double> ls_modelv, ls_modelvp, ls_modeld;
    bool lm_resolve = false;
    double lm_pred = 0.0;
    sparseMat lm_A;
    Array1d<double> lm_data;
    Array1d<int> lm_tmp_data;

    while(iter<=niter_loop){
	ls_restart = false;
	if (!ls_eval && !lm_resolve && do_strategy){
	    if (iter<=it_end1){
		if (strategy_stage!=1){
		    configureStrategyStage(1);
		    has_prev_forward = false;
		}
	    }else if (iter<=it_end2){
		if (strategy_stage!=2){
		    strategyAfterPpp();
		    configureStrategyStage(2);
		    has_prev_forward = false;
		}
	    }else if (iter==it_fwd){
		strategy_full_vp = false;
		invert_vs_psx = true;
		do_ttdiff = false;
		freeze_psx_lid = true;
		freeze_below = false;
		do_convert = false;
		strategySetRayUseByCode(6, 8);
		strategy_skip_update = true;
		strategy_stage = 3;
		strategy_rescale = true;
		has_prev_forward = false;
		cerr << "[tt_inverse] strategy stage=fwd PSS/PSP (no LSQR)\n";
	    }else{
		strategy_skip_update = false;
	    }
	    if (strategy_rescale) has_prev_forward = false;
	}
	const bool joint_hold_vs = joint_staged && iter<=joint_vp_only_iters;
	const bool joint_unlock_vs = !ls_eval && !lm_resolve && joint_staged && iter==joint_vp_only_iters+1;
	if (!ls_eval && !lm_resolve && joint_staged){
	    do_convert = false;
	    strategy_full_vp = joint_hold_vs;
	    invert_vs_psx = !joint_hold_vs;
	}
	if (joint_unlock_vs){
	    const double k_lid = vpvs_kappa>0.0 ? vpvs_kappa : 1.73;
	    const double k_bel = vpvs_kappa_below>0.0 ? vpvs_kappa_below : k_lid;
	    if (!smesh.hasDualVs()){
		if (vpvs_kappa<=0.0)
		    error("TomographicInversion2d:: joint unlock needs -k or -U");
		smesh.enableDualVs(k_lid, k_bel, convp);
		cerr << "[tt_inverse] joint: enable dual Vs=Vp/k after PPP"
		     << " lid=" << k_lid << " below=" << k_bel << "\n";
	    }else if (convp && vpvs_kappa>0.0){
		smesh.resetVsFromVpBelow(k_bel, *convp);
		cerr << "[tt_inverse] joint: reset below-conv Vs = Vp/"
		     << k_bel << " (keep lid Vs), then Vs-only\n";
	    }
	    strategy_full_vp = false;
	    strategy_rescale = true;
	    has_prev_forward = false;
	}
	if (joint_unlock_vs && nrefl>0 && !freeze_refl){
	    freezeRefl();
	    cerr << "[tt_inverse] joint: freeze Moho after PPP+PmP (Vs-only)\n";
	}
	if (!ls_eval && !lm_resolve && verbose_level>=0){
	    cerr << "TomographicInversion2d::iter="
		 << iter << "(" << niter_loop << ")";
	    if (do_strategy){
		if (strategy_skip_update) cerr << "  [strategy fwd]";
		else if (strategy_stage==1) cerr << "  [strategy PPP Vp]";
		else if (strategy_stage==2) cerr << "  [strategy lid Vs]";
		else if (strategy_stage==3) cerr << "  [strategy below Vs]";
	    }
	    if (joint_hold_vs) cerr << "  [joint Vp-only]";
	    else if (joint_staged) cerr << "  [joint Vs-only, Vp frozen]";
	    cerr << "\n";
	}
	if (!ls_eval && !lm_resolve && iter==niter_loop) isFinal=true;
	double graph_time=0.0, bend_time=0.0;
	int c2f_i = iter, c2f_n = niter_user;
	if (do_strategy){
	    if (iter<=it_end1){ c2f_i = iter; c2f_n = it_end1; }
	    else if (iter<=it_end2){ c2f_i = iter-it_end1; c2f_n = niter_user; }
	    else if (iter==it_fwd){ c2f_i = 1; c2f_n = 1; }
	    else { c2f_i = iter-it_fwd; c2f_n = niter_user; }
	}else if (joint_staged && joint_vp_only_iters>0){
	    if (iter<=joint_vp_only_iters){
		c2f_i = iter; c2f_n = joint_vp_only_iters;
	    }else{
		c2f_i = iter-joint_vp_only_iters; c2f_n = niter_user;
	    }
	}
	double c2f_progress = (c2f_n>1) ? double(c2f_i-1)/double(c2f_n-1) : 1.0;
	double smooth_stage_factor = 1.0;
	double damp_stage_factor = 1.0;
	if (enable_c2f){
	    // Use log interpolation so factor changes smoothly across orders of magnitude.
	    smooth_stage_factor = exp(log(c2f_smooth_start)*(1.0-c2f_progress)
				      + log(c2f_smooth_end)*c2f_progress);
	    damp_stage_factor = exp(log(c2f_damp_start)*(1.0-c2f_progress)
				    + log(c2f_damp_end)*c2f_progress);
	}
	if (!ls_eval && !lm_resolve && enable_c2f && verbose_level>=0){
	    cerr << "TomographicInversion2d:: coarse-to-fine factors: smooth="
		 << smooth_stage_factor << " damp=" << damp_stage_factor
		 << " (iter " << iter << "/" << niter << ")\n";
	}

	if (strategy_full_vp && smesh.hasDualVs() && !invert_joint_vpvs)
	    smesh.getVp(modelv);
	else
	    smesh.get(modelv);
	if (invert_joint_vpvs) smesh.getVp(modelvp);
 
	if (nrefl==1) reflp->get(modeld);
	if (!ls_eval && !lm_resolve && (iter==1 || !jumping || strategy_rescale)){
	    mvscale = modelv;
	    if (invert_joint_vpvs) mvscale_vp = modelvp;
	    if (nrefl==1) mdscale = modeld;
	    strategy_rescale = false;
	}
	if (!lm_resolve){
	reset_kernel(); nodev_hit=0;
	bool reuse_forward = false;
	double model_rel_change = -1.0;
	if (enable_forward_reuse && has_prev_forward && !ls_eval){
	    double dm2 = 0.0;
	    double m2 = 0.0;
	    for (int i=1; i<=modelv.size(); i++){
		const double dv = modelv(i)-prev_forward_modelv(i);
		dm2 += dv*dv;
		m2 += prev_forward_modelv(i)*prev_forward_modelv(i);
	    }
	    if (invert_joint_vpvs && modelvp.size()==prev_forward_modelvp.size()){
		for (int i=1; i<=modelvp.size(); i++){
		    const double dv = modelvp(i)-prev_forward_modelvp(i);
		    dm2 += dv*dv;
		    m2 += prev_forward_modelvp(i)*prev_forward_modelvp(i);
		}
	    }
	    if (nrefl==1 && modeld.size()==prev_forward_modeld.size()){
		for (int i=1; i<=modeld.size(); i++){
		    const double dd = modeld(i)-prev_forward_modeld(i);
		    dm2 += dd*dd;
		    m2 += prev_forward_modeld(i)*prev_forward_modeld(i);
		}
	    }
	    model_rel_change = (m2>0.0) ? sqrt(dm2/m2) : sqrt(dm2);
	    reuse_forward = (model_rel_change<=forward_reuse_thresh);
	}
	const bool parallel_src = enable_src_parallel
	    && !do_full_refl;
	if (iter==1 && request_src_parallel && !parallel_src){
	    cerr << "TomographicInversion2d:: OMP requested but source-wise parallel disabled. "
		 << "compiled_with_openmp=" << compiled_with_openmp
		 << " do_full_refl=" << do_full_refl
		 << " printTransient=" << printTransient
		 << " printFinal=" << printFinal
		 << " isFinal=" << isFinal
		 << " verbose_level=" << verbose_level
		 << "\n";
	    cerr.flush();
	}
	if (parallel_src){
	    int nthr = 1;
#ifdef _OPENMP
	    nthr = omp_get_max_threads();
#endif
	    if (iter==1){
		cerr << "TomographicInversion2d:: parallel ray tracing enabled (source-wise, OMP)"
		     << " threads=" << nthr
		     << " nsrc=" << src.size()
		     << "\n";
		cerr.flush();
		if (legacy_baseline){
		    cerr << "TomographicInversion2d:: legacy baseline mode enabled (rollback active)\n";
		    cerr.flush();
		}
	    }
	    cerr << "TomographicInversion2d::iter=" << iter << "(" << niter_loop << ")"
		 << " OMP tracing " << src.size() << " sources, threads=" << nthr
		 << "\n";
	    cerr.flush();
	}
	if (reuse_forward){
	    A = A_prev;
	    path_length = path_length_prev;
	    for (int isrc=1; isrc<=src.size(); isrc++){
		res_ttime(isrc) = res_ttime_prev(isrc);
	    }
	    if (verbose_level>=0){
		cerr << "TomographicInversion2d:: reusing forward kernels (relative model change="
		     << model_rel_change << ", threshold=" << forward_reuse_thresh << ")\n";
	    }
	}else{
	int nsrc_done = 0;
#pragma omp parallel if(parallel_src)
	{
	    GraphSolver2d graph_local(smesh,graph.xOrder(),graph.zOrder());
	    if (graph.critLength()>0) graph_local.refineIfLong(graph.critLength());
	    if (graph.reflDownward()) graph_local.do_refl_downward();
	    BendingSolver2d bend_local(smesh,betasp,bend.tolerance(),bend.brentTolerance());
	    Array1d<const Point2d*> pp_local_buf;
	    Array1d<Point2d> Q_local_buf(betasp.numIntp());
	    double graph_time_local = 0.0;
	    double bend_time_local = 0.0;
#pragma omp for schedule(dynamic,1)
	    for (int isrc=1; isrc<=src.size(); isrc++){
	    Array1d<Point2d> path_local;
	    Array1d<int> start_i_local, end_i_local;
	    Array1d<const Interface2d*> interp_local;
	    start_i_local.reserve(6); end_i_local.reserve(6); interp_local.reserve(6);
	    const int nrcv_src = rcv(isrc).size();
	    std::vector< std::unordered_map<int,double> > A_local_entries_hash;
	    std::vector< std::vector< std::pair<int,double> > > A_local_entries_legacy;
	    if (legacy_baseline){
		A_local_entries_legacy.resize(nrcv_src);
	    }else{
		A_local_entries_hash.resize(nrcv_src);
	    }
	    std::vector<double> path_length_local(nrcv_src, 0.0);

	    auto add_kernel_local = [&](int ircv_local, const Array1d<Point2d>& cur_path,
					bool freeze_above_conv=false, int psx_code=0,
					int iu1=0, int id0=0, int id1=0){
		const int np = cur_path.size();
		const int nintp_local = betasp.numIntp();
		if (legacy_baseline){
		    std::vector< std::pair<int,double> >& A_i = A_local_entries_legacy[ircv_local-1];
		    if (A_i.empty()){
			const size_t seg_count = size_t(np + 1) * size_t(std::max(1, nintp_local - 1));
			const size_t est_nnz = seg_count * size_t(4) + size_t(8);
			A_i.reserve(est_nnz);
		    }
		}else{
		    std::unordered_map<int,double>& A_i = A_local_entries_hash[ircv_local-1];
		    if (A_i.empty()){
			// Pre-reserve nnz contributions for this ray to reduce reallocation overhead.
			const size_t seg_count = size_t(np + 1) * size_t(std::max(1, nintp_local - 1));
			const size_t est_nnz = seg_count * size_t(4) + size_t(8);
			A_i.reserve(est_nnz);
		    }
		}
		makeBSpoints(cur_path, pp_local_buf);

		const bool psx_below_s = (psx_code==6 || psx_code==8
					  || psx_code==9 || psx_code==11
					  || psx_code==12 || psx_code==13 || psx_code==15);
		const bool psx_lid_s = (psx_code==7 || psx_code==8 || psx_code==10
					|| psx_code==11 || psx_code==13
					|| psx_code==14 || psx_code==15);
		const bool is_psx_code = (psx_code==6 || psx_code==7 || psx_code==8
					  || psx_code==9 || psx_code==10 || psx_code==11
					  || psx_code==12 || psx_code==13
					  || psx_code==14 || psx_code==15);

		double path_len = 0.0;
		Index2d guess_index = smesh.nodeIndex(smesh.nearest(*pp_local_buf(1)));
		for (int i=1; i<=np+1; i++){
		    int j1=i;
		    int j2=i+1;
		    int j3=i+2;
		    int j4=i+3;
		    betasp.interpolate(*pp_local_buf(j1),*pp_local_buf(j2),*pp_local_buf(j3),*pp_local_buf(j4),Q_local_buf);

		    for (int j=2; j<=nintp_local; j++){
			Point2d midp = 0.5*(Q_local_buf(j-1)+Q_local_buf(j));
			double dist = Q_local_buf(j).distance(Q_local_buf(j-1));
			path_len += dist;
			int psx_seg = 0; // 1=lid S, 2=below S
			if (is_psx_code){
			    const double zc = convp ? convp->z(midp.x()) : 1e30;
			    const double zsf = seafloorp ? seafloorp->z(midp.x()) : -1e30;
			    if (midp.y() > zc + 1e-6){
				if (psx_below_s) psx_seg = 2;
			    }else if (midp.y() >= zsf - 1e-6){
				int path_i = 1;
				double best = 1e30;
				for (int k=1; k<=np; k++){
				    double d = midp.distance(cur_path(k));
				    if (d<best){ best=d; path_i=k; }
				}
				// 7/8/13 OBS-side lid S, including on-conv. +2 matches
				// spline piece vs pin index (same as psx_sample_is_s).
				if (psx_lid_s && path_i + 2 >= id1)
				    psx_seg = 1;
				// 盖层已由 PPS 收住并冻结时，PSS/13 只留面下 S。
				if (psx_seg==1 && (freeze_psx_lid
						   || (pss_below_only && (psx_code==8 || psx_code==13))))
				    continue;
			    }
			    // P 段：Vs-only 本来就不写；联合也只让 0/1 写 Vp。
			    // PSS 初至残差远大于 PPP 时，P 段若进 Vp 会在转换面拧出
			    // 高速薄层，射线贴面走，面下 S 核照不深。
			    if (psx_seg==0) continue;
			}

			int jUL, jLL, jLR, jUR;
			double r, s, rr, ss;
			int icell=smesh.locateInCell(midp,guess_index,jUL,jLL,jLR,jUR,r,s,rr,ss);
			int p_dom = 0;
			if (invert_joint_vpvs && convp && psx_seg==0 && !strategy_full_vp){
			    const double zc = convp->z(midp.x());
			    if (midp.y() > zc + 1e-6) p_dom = 2;
			    else if (midp.y() < zc - 1e-6) p_dom = 1;
			}
			if (icell>0){
			    auto accum = [&](int jn, double w){
				const bool is_s = (psx_seg!=0);
				if (is_s){
				    if (kernelRestricted() && !kernelAllowNode(jn)) return;
				    if (freeze_above_conv && convp){
					Point2d pn = smesh.nodePos(jn);
					if (pn.y() < convp->z(pn.x()) - 1e-6) return;
				    }
				    if (convp){
					Point2d pn = smesh.nodePos(jn);
					const double zcn = convp->z(pn.x());
					if (psx_seg==1 && pn.y() >= zcn - 1e-6) return;
					if (psx_seg==2 && pn.y() <= zcn + 1e-6) return;
					if (psx_seg==1 && seafloorp
					    && pn.y() <= seafloorp->z(pn.x()) + 1e-6)
					    return;
					// 12/13 面下 S 不到地幔；双线性不要抹到 -F 以下结点。
					if (psx_seg==2 && reflp
					    && (psx_code==12 || psx_code==13)
					    && pn.y() > reflp->z(pn.x()) + 1e-6)
					    return;
				    }
				}else if (invert_joint_vpvs){
				    if (kernelRestrictedVp() && !kernelAllowNodeVp(jn)) return;
				    if (convp){
					Point2d pn = smesh.nodePos(jn);
					const double zcn = convp->z(pn.x());
					if (p_dom==1 && pn.y() >= zcn - 1e-6) return;
					if (p_dom==2 && pn.y() <= zcn + 1e-6) return;
					if (p_dom==0 && abs(pn.y()-zcn) <= 1e-6) return;
				    }
				}else{
				    if (kernelRestricted() && !kernelAllowNode(jn)) return;
				    if (freeze_above_conv && convp){
					Point2d pn = smesh.nodePos(jn);
					if (pn.y() < convp->z(pn.x()) - 1e-6) return;
				    }
				}
				const int col = (is_s && invert_joint_vpvs) ? jn+nnodev : jn;
				if (legacy_baseline){
				    std::vector< std::pair<int,double> >& A_i = A_local_entries_legacy[ircv_local-1];
				    A_i.push_back(std::make_pair(col, w));
				}else{
				    std::unordered_map<int,double>& A_i = A_local_entries_hash[ircv_local-1];
				    A_i[col] += w;
				}
			    };
			    accum(jUL, rr*ss*dist);
			    accum(jLL, rr*s*dist);
			    accum(jLR, r*s*dist);
			    accum(jUR, r*ss*dist);
			}
		    }
		}
		path_length_local[ircv_local-1] = path_len;
	    };

	    auto add_kernel_refl_local = [&](int ircv_local, const Array1d<Point2d>& cur_path,
					    int ir0, int ir1, bool s_bounce=false,
					    int psx_code=0, int iu1=0, int id0=0, int id1=0){
		int jL, jR;
		reflp->locateInSegment(cur_path(ir0).x(),jL,jR);
		if (jL==jR){
		    cerr << "TomographicInversion2d::add_kernel_refl - bottoming point out of bounds\n";
		    return;
		}
		double x1 = reflp->x(jL);
		double x2 = reflp->x(jR);
		double z1 = reflp->z(x1);
		double z2 = reflp->z(x2);
		double x = cur_path(ir0).x();
		double dx = x2-x1;
		double dz = z2-z1;
		double cos_alpha = dx/sqrt(dx*dx+dz*dz);
		double p1 = smesh.at(cur_path(ir0));
		if (s_bounce){
		    Index2d g = smesh.nodeIndex(smesh.nearest(cur_path(ir0)));
		    if (convp)
			p1 = smesh.atPsx(cur_path(ir0), *convp, true, true, g);
		    else if (vpvs_kappa>0.0 && !smesh.hasDualVs())
			p1 *= vpvs_kappa;
		}
		Point2d a = cur_path(ir0-1)-cur_path(ir0);
		Point2d b = cur_path(ir1+1)-cur_path(ir1);
		const double an = a.norm();
		const double bn = b.norm();
		double com_factor = 0.0;
		bool depth_ok = (an>1e-12 && bn>1e-12 && std::isfinite(dx) && abs(dx)>1e-12);
		if (depth_ok){
		    a /= an; b /= bn;
		    Point2d c = a+b;
		    const double cn = c.norm();
		    depth_ok = (cn>1e-12 && std::isfinite(p1));
		    if (depth_ok){
			c /= cn;
			double cos_theta = a.inner_product(c);
			com_factor = 2.0*cos_theta*cos_alpha*p1/dx;
			depth_ok = std::isfinite(com_factor);
		    }
		}
		const int nvel = nVelUnknowns();
		if (!freeze_refl && depth_ok){
		    if (legacy_baseline){
			std::vector< std::pair<int,double> >& A_i = A_local_entries_legacy[ircv_local-1];
			A_i.push_back(std::make_pair(nvel+jL, com_factor*(x2-x)));
			A_i.push_back(std::make_pair(nvel+jR, com_factor*(x-x1)));
		    }else{
			std::unordered_map<int,double>& A_i = A_local_entries_hash[ircv_local-1];
			A_i[nvel+jL] = com_factor*(x2-x);
			A_i[nvel+jR] = com_factor*(x-x1);
		    }
		}
		add_kernel_local(ircv_local, cur_path, false, psx_code, iu1, id0, id1);
	    };

	    if (!parallel_src && verbose_level>=0){
		cerr << "isrc=" << isrc << " nrec=" << rcv(isrc).size()
		     << " ";
	    }
	    ofstream *tres_os_p = 0, *ray_os_p = 0;
	    if (printTransient || (printFinal && isFinal)){
		if (out_level >= 1){
		    char transfn[MaxStr];
		    sprintf(transfn, "%s.tres.%d.%d", out_root, iter, isrc);
		    tres_os_p = new ofstream(transfn);
		    if (tres_os_p){
			*tres_os_p << "# rcv_x residual raytype\n";
			// 第三列 raytype 与 -G 的 r 行一致：0=折射 Pg，1=反射 PmP
		    }
		}
		if (out_level >= 2){
		    char transfn[MaxStr];
		    sprintf(transfn, "%s.ray.%d.%d", out_root, iter, isrc);
		    ray_os_p = new ofstream(transfn);
		}
	    }
	    // limit range first
	    double xmin=smesh.xmax();
	    double xmax=smesh.xmin();
	    for (int ircv=1; ircv<=rcv(isrc).size(); ircv++){
		if (!strategyRayOn(isrc, ircv)) continue;
		if (joint_staged){
		    const int c = raytype(isrc)(ircv);
		    if (joint_hold_vs && c!=0 && c!=1) continue;
		    if (!joint_hold_vs && (c==0 || c==1)) continue;
		}
		double x = rcv(isrc)(ircv).x();
		if (x < xmin) xmin = x;
		if (x > xmax) xmax = x;
	    }
	    double srcx=src(isrc).x();
	    if (srcx < xmin) xmin = srcx;
	    if (srcx > xmax) xmax = srcx;
	    graph_local.limitRange(xmin,xmax);
	    if (!parallel_src && verbose_level>=1){
		cerr << "xrange=(" << xmin << "," << xmax << ") ";
	    }

	    //
	    Array1d<double> tmp_modelv_src, vp_keep, tmp_modelu_src, vs_keep;
	    if (do_full_refl){
		double pwater = 1.0/1.5;
		vp_keep.resize(smesh.numNodes());
		smesh.getVp(vp_keep);
		tmp_modelv_src.resize(vp_keep.size());
		tmp_modelv_src = vp_keep;
		Array1d<int> irefl(nx);
		for (int i=1; i<=nx; i++){
		    Point2d p = smesh.nodePos(smesh.nodeIndex(i,1));
		    double reflz = reflp->z(p.x());
		    irefl(i) = nz;
		    for (int k=1; k<=nz; k++){
			 if (smesh.nodePos(smesh.nodeIndex(i,k)).y()>reflz){
			    irefl(i) = k;
			    break;
			 }
		    }
		}
		for (int i=1; i<=nx; i++){
		    for (int k=2; k<=nz; k++){
			if (k==irefl(i)){
			    // this is to make the velocity at the node just below the reflector to
			    // be the same as the velocity just above. If I use pwater for this node,
			    // there would be a unwanted low velocity gradient surrounding the reflector.
			    tmp_modelv_src(smesh.nodeIndex(i,k)) = tmp_modelv_src(smesh.nodeIndex(i,k-1));
			}else if (k>irefl(i)){
			    tmp_modelv_src(smesh.nodeIndex(i,k)) = pwater;
			}
		    }
		}
		if (smesh.hasDualVs()){
		    vs_keep.resize(smesh.numNodes());
		    smesh.get(vs_keep);
		    tmp_modelu_src.resize(vs_keep.size());
		    tmp_modelu_src = vs_keep;
		    for (int i=1; i<=nx; i++){
			for (int k=2; k<=nz; k++){
			    if (k==irefl(i))
				tmp_modelu_src(smesh.nodeIndex(i,k)) = tmp_modelu_src(smesh.nodeIndex(i,k-1));
			    else if (k>irefl(i))
				tmp_modelu_src(smesh.nodeIndex(i,k)) = pwater;
			}
		    }
		}
	    }
	    
	    double start_t=inv_wall_seconds();
	    bool has_crust_phase=false, has_water_phase=false, has_water_mult=false;
	    bool has_recv_peg_refr=false, has_recv_peg_refl=false, has_psp_phase=false;
	    for (int jrt=1; jrt<=rcv(isrc).size(); jrt++){
		if (!strategyRayOn(isrc, jrt)) continue;
		int c = raytype(isrc)(jrt);
		if (joint_staged && joint_hold_vs && c!=0 && c!=1) continue;
		if (joint_staged && !joint_hold_vs && (c==0 || c==1)) continue;
		if (c==0 || c==1) has_crust_phase=true;
		if (c==2 || c==3 || c==4 || c==5 || c==9 || c==14 || c==15) has_water_phase=true;
		if (c==3 || c==4 || c==5 || c==9 || c==14 || c==15) has_water_mult=true;
		if (c==4) has_recv_peg_refr=true;
		if (c==5) has_recv_peg_refl=true;
		if (c==6) has_psp_phase=true;
	    }
	    if (has_crust_phase){
		// 0/1：原始 tomo2d。有 -B 时走 solve_conv，否则普通初至。不进 solve_psp。
		if (do_convert && convp)
		    graph_local.solve_conv(src(isrc), *convp);
		else
		    graph_local.solve(src(isrc));
	    }
	    if (has_psp_phase && !invert_vs_psx){
		if (convp==0)
		    error("TomographicInversion2d:: raytype 6 requires conversion interface (-B)");
		graph_local.solve_psp(src(isrc), *convp,
				      vpvs_kappa>0.0 ? vpvs_kappa : 1.0);
	    }
	    bool is_refl_solved = false;
	    double end_t=inv_wall_seconds();
	    graph_time_local += end_t-start_t;

	    start_t = inv_wall_seconds();
	    if (has_water_phase){
		const Interface2d* sf = waterBottom();
		if (!sf)
		error("TomographicInversion2d:: raytype 2/3/4/5/9/14/15 requires -Y (seafloor) or -F");
		double tw0=inv_wall_seconds();
		graph_local.solve_water(src(isrc), *sf);
		if (has_water_mult)
		    graph_local.solve_water_mult(*sf, *bathyp);
		if (has_recv_peg_refr)
		    graph_local.solve_recv_peg_refr(*sf, *bathyp);
		if (has_recv_peg_refl){
		    if (seafloorp==0 || nrefl==0)
			error("TomographicInversion2d:: raytype 5 requires -Y (seafloor) and -F (Moho)");
		    graph_local.solve_recv_peg_refl(*seafloorp, *bathyp, *reflp);
		}
		graph_time_local += inv_wall_seconds()-tw0;
	    }
	    double graph_refl_time=0;
	    int conv_mode_solved = -1;
	    for (int ircv=1; ircv<=rcv(isrc).size(); ircv++){
		if (!strategyRayOn(isrc, ircv)) continue;
		int icode_skip = raytype(isrc)(ircv);
		if (joint_staged && joint_hold_vs && icode_skip!=0 && icode_skip!=1) continue;
		if (joint_staged && !joint_hold_vs && (icode_skip==0 || icode_skip==1)) continue;
		Point2d r = rcv(isrc)(ircv);
		double orig_t, final_t;
		int iterbend;
		int icode = raytype(isrc)(ircv);
		int ray_iu1=0, ray_id0=0, ray_id1=0;
		if (icode == 0){ // refraction
		    if (conv_mode_solved >= 0){
			if (do_convert && convp)
			    graph_local.solve_conv(src(isrc), *convp);
			else
			    graph_local.solve(src(isrc));
			conv_mode_solved = -1;
		    }
		    if (smesh.inWater(r)){
			if (!parallel_src && verbose_level>=0) cerr << "*";
			int i0, i1;
			graph_local.pickPathThruWater(r,path_local,i0,i1);
			start_i_local.resize(1); end_i_local.resize(1); interp_local.resize(1);
			start_i_local(1) = i0; end_i_local(1) = i1; interp_local(1) = bathyp;
			iterbend=bend_local.refine(path_local,orig_t,final_t,
					     start_i_local, end_i_local, interp_local);
			if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    }else{
			if (!parallel_src && verbose_level>=0) cerr << ".";
			graph_local.pickPath(rcv(isrc)(ircv),path_local);
			iterbend=bend_local.refine(path_local,orig_t,final_t,bend_nfac);
			if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    }
		    if (iterbend<0){
			cerr << "TomographicInversion2d::iteration = " << iter << '\n';
			cerr << "TomographicInversion2d::too many iterations required\n";
			cerr << "TomographicInversion2d::for bending refinement at (s,r)="
			     << isrc << "," << ircv << '\n';
			exit(1);
		    }
		    // -td: type 0 only supplies T_PPP for diffs; do not write Vp→Vs columns
		    if (!do_ttdiff)
			add_kernel_local(ircv,path_local);
		}else if (icode == 1){ // reflection
		    if (conv_mode_solved >= 0){
			is_refl_solved = false;
			conv_mode_solved = -1;
		    }
		    if (nrefl==0){
			error("TomographicInversion2d:: reflector not specified.");
		    }
		    if (do_full_refl){
			// temporarily replace sub-reflector velocity field
			// with water velocity
			smesh.setVp(tmp_modelv_src);
		    }
		    if (!is_refl_solved){
			double start_t=inv_wall_seconds();
			if (do_convert && convp)
			    graph_local.solve_refl_conv(src(isrc), *reflp, *convp);
			else
			    graph_local.solve_refl(src(isrc), *reflp);
			double end_t=inv_wall_seconds();
			graph_refl_time = end_t-start_t;
			graph_time_local += graph_refl_time;
			is_refl_solved = true;
		    }
		    int ir0, ir1;
		    if (smesh.inWater(r)){
			if (!parallel_src && verbose_level>=0) cerr << "#";
			int i0, i1;
			graph_local.pickReflPathThruWater(r,path_local,i0,i1,ir0,ir1);
			start_i_local.resize(2); end_i_local.resize(2); interp_local.resize(2);
			start_i_local(1) = i0; end_i_local(1) = i1; interp_local(1) = bathyp;
			start_i_local(2) = ir0; end_i_local(2) = ir1; interp_local(2) = reflp;
			int iterbend=bend_local.refine(path_local,orig_t,final_t,
						 start_i_local, end_i_local, interp_local);
			if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    }else{
			if (!parallel_src && verbose_level>=0) cerr << "+";
			graph_local.pickReflPath(r,path_local,ir0,ir1);
			start_i_local.resize(1); end_i_local.resize(1); interp_local.resize(1);
			start_i_local(1) = ir0; end_i_local(1) = ir1; interp_local(1) = reflp;
			int iterbend=bend_local.refine(path_local,orig_t,final_t,
						 start_i_local, end_i_local, interp_local);
			if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    }
		    if (do_full_refl){
			// revert to the original smesh
			smesh.setVp(vp_keep);
		    }
		    if (!do_ttdiff)
			add_kernel_refl_local(ircv,path_local,ir0,ir1);
		}else if (icode == 2 || icode == 3){
		    const Interface2d* sf = waterBottom();
		    if (!sf)
			error("TomographicInversion2d:: raytype 2/3 requires -Y (seafloor) or -F");
		    int i0, i1, ir0, ir1;
		    bend_local.setWaterColClip(bathyp, sf);
		    if (icode==2){
			if (!parallel_src && verbose_level>=0) cerr << "w";
			graph_local.pickWaterDirectPath(r,path_local,ir0,ir1);
			clampPathToWaterCol(path_local, *bathyp, *sf);
			iterbend=bend_local.refine(path_local,orig_t,final_t,1);
		    }else{
			if (!parallel_src && verbose_level>=0) cerr << "m";
			graph_local.pickWaterMultPath(r,path_local,i0,i1,ir0,ir1);
			start_i_local.resize(2); end_i_local.resize(2); interp_local.resize(2);
			start_i_local(1) = ir0; end_i_local(1) = ir1; interp_local(1) = sf;
			start_i_local(2) = i0; end_i_local(2) = i1; interp_local(2) = bathyp;
			clampPathToWaterCol(path_local, *bathyp, *sf);
			iterbend=bend_local.refine(path_local,orig_t,final_t,
					 start_i_local, end_i_local, interp_local);
		    }
		    if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    clampPathToWaterCol(path_local, *bathyp, *sf);
		    {
			Array1d<const Point2d*> wpp;
			Array1d<Point2d> wQ(betasp.numIntp());
			makeBSpoints(path_local, wpp);
			final_t = calcTravelTime(smesh, path_local, betasp, wpp, wQ, bathyp, sf);
		    }
		    bend_local.setWaterColClip(0, 0);
		    add_kernel_local(ircv,path_local);
		}else if (icode == 4){
		    const Interface2d* sf = waterBottom();
		    if (!sf)
			error("TomographicInversion2d:: raytype 4 requires -Y (seafloor) or -F");
		    if (!parallel_src && verbose_level>=0) cerr << "P";
		    int is0, is1, ib0, ib1, ie0, ie1;
		    graph_local.pickRecvPegRefrPath(r,path_local,is0,is1,ib0,ib1,ie0,ie1);
		    start_i_local.resize(3); end_i_local.resize(3); interp_local.resize(3);
		    start_i_local(1) = is0; end_i_local(1) = is1; interp_local(1) = bathyp;
		    start_i_local(2) = ib0; end_i_local(2) = ib1; interp_local(2) = sf;
		    start_i_local(3) = ie0; end_i_local(3) = ie1; interp_local(3) = sf;
		    iterbend=bend_local.refine(path_local,orig_t,final_t,
					 start_i_local, end_i_local, interp_local);
		    if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    if (iterbend<0){
			cerr << "TomographicInversion2d::iteration = " << iter << '\n';
			cerr << "TomographicInversion2d::too many iterations required\n";
			cerr << "TomographicInversion2d::for bending refinement at (s,r)="
			     << isrc << "," << ircv << '\n';
			exit(1);
		    }
		    add_kernel_local(ircv,path_local);
		}else if (icode == 5){
		    if (seafloorp==0 || nrefl==0)
			error("TomographicInversion2d:: raytype 5 requires -Y (seafloor) and -F (Moho)");
		    if (do_full_refl){
			smesh.setVp(tmp_modelv_src);
		    }
		    if (!parallel_src && verbose_level>=0) cerr << "Q";
		    int is0, is1, ib0, ib1, ir0, ir1;
		    graph_local.pickRecvPegReflPath(r,path_local,is0,is1,ib0,ib1,ir0,ir1);
		    start_i_local.resize(3); end_i_local.resize(3); interp_local.resize(3);
		    start_i_local(1) = is0; end_i_local(1) = is1; interp_local(1) = bathyp;
		    start_i_local(2) = ib0; end_i_local(2) = ib1; interp_local(2) = seafloorp;
		    start_i_local(3) = ir0; end_i_local(3) = ir1; interp_local(3) = reflp;
		    iterbend=bend_local.refine(path_local,orig_t,final_t,
					 start_i_local, end_i_local, interp_local);
		    if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    if (do_full_refl){
			smesh.setVp(vp_keep);
		    }
		    if (iterbend<0){
			cerr << "TomographicInversion2d::iteration = " << iter << '\n';
			cerr << "TomographicInversion2d::too many iterations required\n";
			cerr << "TomographicInversion2d::for bending refinement at (s,r)="
			     << isrc << "," << ircv << '\n';
			exit(1);
		    }
		    add_kernel_refl_local(ircv,path_local,ir0,ir1);
		}else if (icode == 6 && !invert_vs_psx){
		    if (convp==0)
			error("TomographicInversion2d:: raytype 6 requires conversion interface (-B)");
		    if (!parallel_src && verbose_level>=0) cerr << "C";
		    int id0, id1, iu0, iu1;
		    if (smesh.inWater(r)){
			int i0, i1;
			graph_local.pickPspPathThruWater(r,path_local,i0,i1,id0,id1,iu0,iu1);
			start_i_local.resize(3); end_i_local.resize(3); interp_local.resize(3);
			start_i_local(1) = i0; end_i_local(1) = i1; interp_local(1) = bathyp;
			start_i_local(2) = id0; end_i_local(2) = id1; interp_local(2) = convp;
			start_i_local(3) = iu0; end_i_local(3) = iu1; interp_local(3) = convp;
		    }else{
			graph_local.pickPspPath(r,path_local,id0,id1,iu0,iu1);
			start_i_local.resize(2); end_i_local.resize(2); interp_local.resize(2);
			start_i_local(1) = id0; end_i_local(1) = id1; interp_local(1) = convp;
			start_i_local(2) = iu0; end_i_local(2) = iu1; interp_local(2) = convp;
		    }
		    ray_iu1 = iu1; ray_id0 = id0; ray_id1 = id1;
		    // Mixed type 6: mesh already holds Vp/Vs; kappa=1. Enable
		    // S-below-conv clamp. raytype 0/1 never setPsxBend.
		    bend_local.setPsxBend(iu1, id0, id1, 1.0, true, convp, false);
		    iterbend=bend_local.refine(path_local,orig_t,final_t,
					 start_i_local, end_i_local, interp_local);
		    if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    if (iterbend<0){
			// Dual conv pins can exhaust CG; keep last iterate.
			final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
						  ray_iu1, ray_id0, ray_id1, 1.0, true, false, convp);
		    }
		    bend_local.clearPsxBend();
		    // Mixed type 6: freeze lid, but kernel is S-below only (same
		    // as -k). Do not write P-leg samples onto the interface.
		    // raytype 0/1 never enter this branch.
		    add_kernel_local(ircv,path_local,true,6,ray_iu1,ray_id0,ray_id1);
		}else if (icode == 6 || icode == 7 || icode == 8){
		    if (convp==0)
			error("TomographicInversion2d:: raytype 6/7/8 requires conversion interface (-B)");
		    if (seafloorp==0)
			error("TomographicInversion2d:: raytype 6/7/8 requires seafloor (-Y)");
		    const double pk = vpvs_kappa>0.0 ? vpvs_kappa : 1.0;
		    if (conv_mode_solved != icode){
			if (icode==6)
			    graph_local.solve_psp(src(isrc), *convp, pk);
			else if (icode==7)
			    graph_local.solve_pps(src(isrc), *convp, *seafloorp, pk);
			else
			    graph_local.solve_pss(src(isrc), *convp, *seafloorp, pk);
			conv_mode_solved = icode;
		    }
		    if (!parallel_src && verbose_level>=0){
			if (icode==6) cerr << "C";
			else if (icode==7) cerr << "D";
			else cerr << "E";
		    }
		    int id0, id1, iu0, iu1;
		    int i0=0, i1=0;
		    if (smesh.inWater(r))
			graph_local.pickPspPathThruWater(r,path_local,i0,i1,id0,id1,iu0,iu1);
		    else
			graph_local.pickPspPath(r,path_local,id0,id1,iu0,iu1);
		    ray_iu1 = iu1; ray_id0 = id0; ray_id1 = id1;
		    const bool below_is_s = (icode==6 || icode==8);
		    const bool lid_is_s = (icode==7 || icode==8);
		    const bool two_conv_pins = (icode!=7);
		    if (smesh.inWater(r)){
			if (!two_conv_pins){
			    start_i_local.resize(2); end_i_local.resize(2); interp_local.resize(2);
			    start_i_local(1) = i0; end_i_local(1) = i1; interp_local(1) = bathyp;
			    start_i_local(2) = iu0; end_i_local(2) = iu1; interp_local(2) = convp;
			}else{
			    start_i_local.resize(3); end_i_local.resize(3); interp_local.resize(3);
			    start_i_local(1) = i0; end_i_local(1) = i1; interp_local(1) = bathyp;
			    start_i_local(2) = id0; end_i_local(2) = id1; interp_local(2) = convp;
			    start_i_local(3) = iu0; end_i_local(3) = iu1; interp_local(3) = convp;
			}
		    }else if (!two_conv_pins){
			start_i_local.resize(1); end_i_local.resize(1); interp_local.resize(1);
			start_i_local(1) = iu0; end_i_local(1) = iu1; interp_local(1) = convp;
		    }else{
			start_i_local.resize(2); end_i_local.resize(2); interp_local.resize(2);
			start_i_local(1) = id0; end_i_local(1) = id1; interp_local(1) = convp;
			start_i_local(2) = iu0; end_i_local(2) = iu1; interp_local(2) = convp;
		    }
		    bend_local.setPsxBend(iu1, id0, id1, pk, below_is_s, convp, lid_is_s);
		    iterbend=bend_local.refine(path_local,orig_t,final_t,
					 start_i_local, end_i_local, interp_local);
		    if (iterbend<0)
			final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
						  iu1, id0, id1, pk, below_is_s, lid_is_s, convp);
		    bend_local.clearPsxBend();
		    if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    add_kernel_local(ircv,path_local,false,icode,iu1,id0,id1);
		}else if (icode == 9 || icode == 14 || icode == 15){
		    if (convp==0)
			error("TomographicInversion2d:: raytype 9/14/15 requires conversion interface (-B)");
		    if (seafloorp==0)
			error("TomographicInversion2d:: raytype 9/14/15 requires seafloor (-Y)");
		    if (icode!=9 && vpvs_kappa<=0.0 && !smesh.hasDualVs())
			error("TomographicInversion2d:: raytype 14/15 single-field requires -k");
		    const double pk = vpvs_kappa>0.0 ? vpvs_kappa : 1.0;
		    if (conv_mode_solved != icode){
			if (icode==9)
			    graph_local.solve_psp_peg(src(isrc), *convp, *seafloorp, pk, false);
			else if (icode==14)
			    graph_local.solve_pps_peg(src(isrc), *convp, *seafloorp, pk);
			else
			    graph_local.solve_pss_peg(src(isrc), *convp, *seafloorp, pk);
			conv_mode_solved = icode;
		    }
		    if (!parallel_src && verbose_level>=0)
			cerr << (icode==9 ? "J" : (icode==14 ? "N" : "O"));
		    int id0, id1, ib0, ib1, is0, is1, iu0, iu1;
		    int i0=0, i1=0;
		    const bool shot_water = smesh.inWater(r);
		    if (shot_water)
			graph_local.pickPspPegPathThruWater(r,path_local,i0,i1,id0,id1,ib0,ib1,is0,is1,iu0,iu1);
		    else
			graph_local.pickPspPegPath(r,path_local,id0,id1,ib0,ib1,is0,is1,iu0,iu1);
		    ray_iu1 = iu1; ray_id0 = id0; ray_id1 = id1;
		    const bool below_is_s = (icode==9 || icode==15);
		    const bool lid_is_s = (icode==14 || icode==15);
		    const bool two_conv = (icode!=14);
		    int ip = 1;
		    int npin = (shot_water ? 1 : 0) + (two_conv ? 2 : 1) + 2;
		    start_i_local.resize(npin); end_i_local.resize(npin); interp_local.resize(npin);
		    if (shot_water){
			start_i_local(ip) = i0; end_i_local(ip) = i1; interp_local(ip) = bathyp; ip++;
		    }
		    if (two_conv){
			start_i_local(ip) = id0; end_i_local(ip) = id1; interp_local(ip) = convp; ip++;
		    }
		    start_i_local(ip) = iu0; end_i_local(ip) = iu1; interp_local(ip) = convp; ip++;
		    start_i_local(ip) = ib0; end_i_local(ip) = ib1; interp_local(ip) = seafloorp; ip++;
		    start_i_local(ip) = is0; end_i_local(ip) = is1; interp_local(ip) = bathyp;
		    bend_local.setPsxBend(iu1, id0, id1, pk, below_is_s, convp, lid_is_s);
		    iterbend=bend_local.refine(path_local,orig_t,final_t,
					 start_i_local, end_i_local, interp_local);
		    if (iterbend<0)
			final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
						  iu1, id0, id1, pk, below_is_s, lid_is_s, convp);
		    bend_local.clearPsxBend();
		    if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    add_kernel_local(ircv,path_local,false,icode,iu1,id0,id1);
		}else if (icode == 10 || icode == 11){
		    if (convp==0)
			error("TomographicInversion2d:: raytype 10/11 requires conversion interface (-B)");
		    if (seafloorp==0)
			error("TomographicInversion2d:: raytype 10/11 requires seafloor (-Y)");
		    if (vpvs_kappa<=0.0 && !smesh.hasDualVs())
			error("TomographicInversion2d:: raytype 10/11 single-field requires -k");
		    const double pk = vpvs_kappa>0.0 ? vpvs_kappa : 1.0;
		    if (conv_mode_solved != icode){
			if (icode==10)
			    graph_local.solve_pps_ss(src(isrc), *convp, *seafloorp, pk);
			else
			    graph_local.solve_pss_ss(src(isrc), *convp, *seafloorp, pk);
			conv_mode_solved = icode;
		    }
		    if (!parallel_src && verbose_level>=0)
			cerr << (icode==10 ? "K" : "L");
		    int id0, id1, ib0, ib1, ic0, ic1, iu0, iu1;
		    int i0=0, i1=0;
		    const bool shot_water = smesh.inWater(r);
		    if (shot_water)
			graph_local.pickPspSsPathThruWater(r,path_local,i0,i1,id0,id1,ib0,ib1,ic0,ic1,iu0,iu1);
		    else
			graph_local.pickPspSsPath(r,path_local,id0,id1,ib0,ib1,ic0,ic1,iu0,iu1);
		    ray_iu1 = iu1; ray_id0 = id0; ray_id1 = id1;
		    const bool below_is_s = (icode==11);
		    const bool lid_is_s = true;
		    int ip = 1;
		    int npin = (shot_water ? 1 : 0) + (icode==11 ? 2 : 1) + 2;
		    start_i_local.resize(npin); end_i_local.resize(npin); interp_local.resize(npin);
		    if (shot_water){
			start_i_local(ip) = i0; end_i_local(ip) = i1; interp_local(ip) = bathyp; ip++;
		    }
		    if (icode==11){
			start_i_local(ip) = id0; end_i_local(ip) = id1; interp_local(ip) = convp; ip++;
		    }
		    start_i_local(ip) = iu0; end_i_local(ip) = iu1; interp_local(ip) = convp; ip++;
		    start_i_local(ip) = ib0; end_i_local(ip) = ib1; interp_local(ip) = seafloorp; ip++;
		    start_i_local(ip) = ic0; end_i_local(ip) = ic1; interp_local(ip) = convp;
		    bend_local.setPsxBend(iu1, id0, id1, pk, below_is_s, convp, lid_is_s);
		    iterbend=bend_local.refine(path_local,orig_t,final_t,
					 start_i_local, end_i_local, interp_local);
		    if (iterbend<0)
			final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
						  iu1, id0, id1, pk, below_is_s, lid_is_s, convp);
		    bend_local.clearPsxBend();
		    if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    add_kernel_local(ircv,path_local,false,icode,iu1,id0,id1);
		}else if (icode == 12 || icode == 13){
		    if (convp==0)
			error("TomographicInversion2d:: raytype 12/13 requires conversion interface (-B)");
		    if (nrefl==0 || reflp==0)
			error("TomographicInversion2d:: raytype 12/13 requires Moho (-F)");
		    if (icode==13){
			if (seafloorp==0)
			    error("TomographicInversion2d:: raytype 13 requires seafloor (-Y)");
			if (vpvs_kappa<=0.0 && !smesh.hasDualVs())
			    error("TomographicInversion2d:: raytype 13 single-field requires -k (Vp→Vs)");
		    }
		    if (do_full_refl){
			smesh.setVp(tmp_modelv_src);
			if (smesh.hasDualVs()) smesh.set(tmp_modelu_src);
		    }
		    const double pk = vpvs_kappa>0.0 ? vpvs_kappa : 1.0;
		    if (conv_mode_solved != icode){
			if (icode==12)
			    graph_local.solve_psp_moho(src(isrc), *convp, *reflp, pk);
			else
			    graph_local.solve_pss_moho(src(isrc), *convp, *reflp, *seafloorp, pk);
			conv_mode_solved = icode;
		    }
		    if (!parallel_src && verbose_level>=0)
			cerr << (icode==12 ? "H" : "I");
		    int id0, id1, ir0, ir1, iu0, iu1;
		    int i0=0, i1=0;
		    if (smesh.inWater(r))
			graph_local.pickPspMohoPathThruWater(r,path_local,i0,i1,id0,id1,ir0,ir1,iu0,iu1);
		    else
			graph_local.pickPspMohoPath(r,path_local,id0,id1,ir0,ir1,iu0,iu1);
		    ray_iu1 = iu1; ray_id0 = id0; ray_id1 = id1;
		    const bool lid_is_s = (icode==13);
		    if (smesh.inWater(r)){
			start_i_local.resize(4); end_i_local.resize(4); interp_local.resize(4);
			start_i_local(1) = i0; end_i_local(1) = i1; interp_local(1) = bathyp;
			start_i_local(2) = id0; end_i_local(2) = id1; interp_local(2) = convp;
			start_i_local(3) = ir0; end_i_local(3) = ir1; interp_local(3) = reflp;
			start_i_local(4) = iu0; end_i_local(4) = iu1; interp_local(4) = convp;
		    }else{
			start_i_local.resize(3); end_i_local.resize(3); interp_local.resize(3);
			start_i_local(1) = id0; end_i_local(1) = id1; interp_local(1) = convp;
			start_i_local(2) = ir0; end_i_local(2) = ir1; interp_local(2) = reflp;
			start_i_local(3) = iu0; end_i_local(3) = iu1; interp_local(3) = convp;
		    }
		    bend_local.setPsxBend(iu1, id0, id1, pk, true, convp, lid_is_s);
		    bend_local.setPsxMoho(reflp);
		    iterbend=bend_local.refine(path_local,orig_t,final_t,
					 start_i_local, end_i_local, interp_local);
		    if (iterbend<0)
			final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
						  iu1, id0, id1, pk, true, lid_is_s, convp);
		    bend_local.clearPsxBend();
		    if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    if (do_full_refl){
			smesh.setVp(vp_keep);
			if (smesh.hasDualVs()) smesh.set(vs_keep);
		    }
		    add_kernel_refl_local(ircv,path_local,ir0,ir1,true,icode,iu1,id0,id1);
		}else{
		    error("TomographicInversion2d:: illegal raycode detected.");
		}
		res_ttime(isrc)(ircv)=obs_ttime(isrc)(ircv)-final_t;

		if (printTransient || (printFinal && isFinal)){
		if (out_level>=1 && tres_os_p){
			*tres_os_p << rcv(isrc)(ircv).x() << " "
				   << res_ttime(isrc)(ircv) << " "
				   << raytype(isrc)(ircv) << '\n';
		    }
		if (out_level>=2 && ray_os_p){
			*ray_os_p << ">\n";
			if (icode==2 || icode==3){
			    const Interface2d* sf = waterBottom();
			    printCurve(*ray_os_p, path_local, betasp, *bathyp, *sf);
			}else if (ray_iu1>0)
			    printCurve(*ray_os_p, path_local, betasp, ray_iu1, ray_id0, ray_id1);
			else
			    printCurve(*ray_os_p, path_local, betasp);
		    }
		}
	    }
	    if (printTransient || (printFinal && isFinal)){
	    if (out_level>=1 && tres_os_p){ tres_os_p->flush(); delete tres_os_p;}
	    if (out_level>=2 && ray_os_p){ ray_os_p->flush(); delete ray_os_p;}
	    }
	    for (int ircv=1; ircv<=nrcv_src; ircv++){
		int gidata = idata_start(isrc) + ircv - 1;
		std::map<int,double>& A_dst = A(gidata);
		A_dst.clear();
		if (legacy_baseline){
		    std::vector< std::pair<int,double> >& entries = A_local_entries_legacy[ircv-1];
		    if (!entries.empty()){
			std::sort(entries.begin(), entries.end());
			int cur_idx = entries[0].first;
			double cur_val = entries[0].second;
			for (size_t ie=1; ie<entries.size(); ie++){
			    const int idx = entries[ie].first;
			    const double val = entries[ie].second;
			    if (idx==cur_idx){
				cur_val += val;
			    }else{
				A_dst[cur_idx] = cur_val;
				cur_idx = idx;
				cur_val = val;
			    }
			}
			A_dst[cur_idx] = cur_val;
		    }
		}else{
		    std::unordered_map<int,double>& entries = A_local_entries_hash[ircv-1];
		    for (std::unordered_map<int,double>::const_iterator p=entries.begin();
			 p!=entries.end(); ++p){
			if (!std::isfinite(p->second)) continue;
			A_dst[p->first] = p->second;
		    }
		}
		path_length(gidata) = path_length_local[ircv-1];
	    }
		    
	    if (!parallel_src && verbose_level>=0) cerr << '\n';
	    end_t = inv_wall_seconds();
	    bend_time_local += end_t-start_t-graph_refl_time;
	    if (parallel_src){
		int done;
#ifdef _OPENMP
#pragma omp atomic capture
		done = ++nsrc_done;
#else
		done = ++nsrc_done;
#endif
		const int nsrc_all = src.size();
		const int step = (nsrc_all >= 10) ? (nsrc_all / 10) : 1;
		if (done==1 || done==nsrc_all || (done % step)==0){
		    inv_omp_progress_line(iter, done, nsrc_all);
		}
	    }
	    }
#pragma omp atomic
	    graph_time += graph_time_local;
#pragma omp atomic
	    bend_time += bend_time_local;
	}
	    if (enable_forward_reuse){
		A_prev = A;
		path_length_prev = path_length;
		for (int isrc=1; isrc<=src.size(); isrc++){
		    res_ttime_prev(isrc) = res_ttime(isrc);
		}
		prev_forward_modelv = modelv;
		if (invert_joint_vpvs) prev_forward_modelvp = modelvp;
		if (nrefl==1){
		    prev_forward_modeld = modeld;
		}
		has_prev_forward = true;
	    }
	}

	if (do_strategy && strategy_skip_update){
	    applyStrategyCorr();
	    configureStrategyStage(3);
	    strategy_skip_update = false;
	    has_prev_forward = false;
	    iter++;
	    continue;
	}

	// construct data vector
	int idata=1;
	rms_tres[0]=rms_tres[1]=0;
	init_chi[0]=init_chi[1]=0;
	ndata_in[0]=ndata_in[1]=0;
	ndata_valid=0;
	tmp_data=0;
	double sum_res2=0, sum_chi=0;
	for (int isrc=1; isrc<=src.size(); isrc++){
	    for (int ircv=1; ircv<=rcv(isrc).size(); ircv++){
		double res=res_ttime(isrc)(ircv);
		data_vec(idata) = res;
		double res2=res*r_dt_vec(idata);
		double res22=res2*res2;
		int icode = raytype(isrc)(ircv);
		if (!strategyRayOn(isrc, ircv)){
		    idata++;
		    continue;
		}
		if (do_ttdiff && (icode==0 || icode==1
				 || (icode==8 && ttdiff_skip_pss_abs))){
		    idata++;
		    continue;
		}
		if (joint_staged){
		    if (joint_hold_vs && (icode==6 || icode==7 || icode==8
					  || icode==12 || icode==13)){
			idata++;
			continue;
		    }
		    if (!joint_hold_vs && (icode==0 || icode==1)){
			idata++;
			continue;
		    }
		}
		if (icode==0 || icode==1){
		    rms_tres[icode] += res*res;
		    init_chi[icode] += res22;
		    ++ndata_in[icode];
		}
		sum_res2 += res*res;
		sum_chi += res22;
		tmp_data(idata) = ++ndata_valid;
		idata++;
	    }
	}
	if (do_ttdiff){
	    for (int k=1; k<=ttdiff_pairs.size(); k++){
		const TtDiffPair& d = ttdiff_pairs(k);
		const int irow = ndata_abs + k;
		const double ra = res_ttime(d.isrc)(d.ia);
		const double rb = res_ttime(d.isrc)(d.ib);
		const double res = ra - rb;
		data_vec(irow) = res;
		const double dta = obs_dt(d.isrc)(d.ia);
		const double dtb = obs_dt(d.isrc)(d.ib);
		r_dt_vec(irow) = 1.0/sqrt(dta*dta+dtb*dtb);
		A(irow).clear();
		for (mapIterator p=A(d.gid_a).begin(); p!=A(d.gid_a).end(); ++p)
		    A(irow)[p->first] += p->second;
		for (mapIterator p=A(d.gid_b).begin(); p!=A(d.gid_b).end(); ++p)
		    A(irow)[p->first] -= p->second;
		const double res2 = res*r_dt_vec(irow);
		sum_res2 += res*res;
		sum_chi += res2*res2;
		tmp_data(irow) = ++ndata_valid;
	    }
	}
	if (ndata_valid<=0){
	    if (verbose_level>=0)
		cerr << "TomographicInversion2d:: no valid data at iter="
		     << iter << " (strategy_stage=" << strategy_stage << ")\n";
	    if (isFinal) break;
	    iter++;
	    continue;
	}
	rms_tres_total = sqrt(sum_res2/ndata_valid);
	init_chi_total = sum_chi/ndata_valid;
	for (int i=0; i<=1; i++){
	    rms_tres[i] = ndata_in[i]>0 ? sqrt(rms_tres[i]/ndata_in[i]) : 0.0;
	    init_chi[i] = ndata_in[i]>0 ? init_chi[i]/ndata_in[i] : 0.0;
	}
	if (init_chi_total<target_chisq && !(do_strategy && strategy_stage<3))
	    isFinal=true;

	if (ls_eval){
	    const double phi = init_chi_total;
	    bool ok = false;
	    double rho = 0.0;
	    ls_ntry++;
	    if (use_lm){
		const double num = ls_phi0 - phi;
		const double den = ls_phi0 - lm_pred;
		if (std::isfinite(phi) && den>1e-12)
		    rho = num/den;
		else if (std::isfinite(phi) && phi<=ls_phi0)
		    rho = 1.0;
		else
		    rho = -1.0;
		ok = (rho >= lm_rho_accept);
		if (verbose_level>=0)
		    cerr << "[tt_inverse] lm try=" << ls_ntry
			 << " lambda=" << lm_lambda
			 << " rho=" << rho
			 << " chi2=" << phi << " chi2_0=" << ls_phi0
			 << " pred=" << lm_pred
			 << (ok ? " ACCEPT" : " reject") << "\n";
	    }else{
		if (std::isfinite(phi)){
		    if (ls_slope < 0.0)
			ok = (phi <= ls_phi0 + ls_c*ls_alpha*ls_slope);
		    else
			ok = (phi <= ls_phi0);
		}
		if (verbose_level>=0)
		    cerr << "[tt_inverse] linesearch try=" << ls_ntry
			 << " alpha=" << ls_alpha
			 << " chi2=" << phi << " chi2_0=" << ls_phi0
			 << " pred=" << (ls_phi0+ls_slope)
			     << (ok ? " ACCEPT" : " backtrack") << "\n";
		if (!ok && ls_alpha*ls_rho >= ls_amin){
		    ls_alpha *= ls_rho;
		    restoreModel(ls_modelv, ls_modelvp, ls_modeld);
		    applyDmodel(ls_alpha, ls_modelv, ls_modelvp, ls_modeld, false);
		    has_prev_forward = false;
		    continue;
		}
	    }
	    if (!ok && use_lm){
		restoreModel(ls_modelv, ls_modelvp, ls_modeld);
		if (lm_A.size()==A.size()){
		    A = lm_A;
		    data_vec = lm_data;
		    tmp_data = lm_tmp_data;
		}
		init_chi_total = ls_phi0;
		const double next_l = lm_lambda*lm_up;
		if (printLog)
		    *log_os_p << "# lm: reject rho=" << rho
			      << " lambda=" << lm_lambda
			      << " ntry=" << ls_ntry
			      << " chi2_0=" << ls_phi0
			      << " chi2=" << phi << '\n';
		if (next_l <= lm_lmax && ls_ntry < 8){
		    lm_lambda = next_l;
		    ls_eval = false;
		    lm_resolve = true;
		    has_prev_forward = false;
		    if (verbose_level>=0)
			cerr << "[tt_inverse] lm re-LSQR lambda=" << lm_lambda << "\n";
		    continue;
		}
		ls_alpha = 0.0;
		if (verbose_level>=0)
		    cerr << "[tt_inverse] lm stalled (keep previous model)\n";
		if (printTransient || printFinal)
		    writeCurrentModels();
		ls_eval = false;
		if (verbose_level>=0)
		    cerr << "[tt_inverse] lm stalled; stop outer iter\n";
		break;
	    }
	    if (!ok){
		restoreModel(ls_modelv, ls_modelvp, ls_modeld);
		ls_alpha = 0.0;
		if (verbose_level>=0)
		    cerr << "[tt_inverse] linesearch reject (keep previous model)\n";
		if (printTransient || printFinal)
		    writeCurrentModels();
	    }else{
		restoreModel(ls_modelv, ls_modelvp, ls_modeld);
		applyDmodel(ls_alpha, ls_modelv, ls_modelvp, ls_modeld,
			    printTransient || printFinal);
		if (jumping){
		    for (int i=1; i<=dmodel_total.size(); i++)
			dmodel_total_sum(i) += ls_alpha*dmodel_total(i);
		}
		if (use_lm && rho >= lm_rho_good)
		    lm_lambda = (lm_lambda*lm_down < lm_lmin) ? lm_lmin : lm_lambda*lm_down;
	    }
	    if (printLog){
		if (use_lm)
		    *log_os_p << "# lm: accept rho=" << rho
			      << " lambda=" << lm_lambda
			      << " ntry=" << ls_ntry
			      << " chi2_0=" << ls_phi0
			      << " chi2=" << phi << '\n';
		else
		    *log_os_p << "# linesearch: alpha=" << ls_alpha
			      << " ntry=" << ls_ntry
			      << " chi2_0=" << ls_phi0
			      << " chi2=" << phi << '\n';
	    }
	    ls_eval = false;
	    lm_resolve = false;
	    if (!ok){
		if (verbose_level>=0)
		    cerr << "[tt_inverse] linesearch stalled; stop outer iter\n";
		break;
	    }
	    if (isFinal) break;
	    iter++;
	    continue;
	}

	// rescale kernel A and data vector
	// note: averaging matrices are scaled upon their construction.
	for (int i=1; i<=ndata; i++){
	    double data_wt=r_dt_vec(i);
	    if (!std::isfinite(data_vec(i))){
		tmp_data(i) = 0;
		continue;
	    }
	    data_vec(i) *= data_wt;
	    for (mapIterator p=A(i).begin(); p!=A(i).end(); ){
		if (!std::isfinite(p->second)){
		    A(i).erase(p++);
		    continue;
		}
		int j = p->first;
		double m;
		const int nvel = nVelUnknowns();
		if (j<=nnodev){
		    m = invert_joint_vpvs ? mvscale_vp(j) : mvscale(j);
		}else if (invert_joint_vpvs && j<=nvel){
		    m = mvscale(j-nnodev);
		}else{
		    m = mdscale(j-nvel)*refl_weight;
		}
		p->second *= m*data_wt;
		if (!std::isfinite(p->second)){
		    A(i).erase(p++);
		    continue;
		}
		++p;
	    }
	}
	if (use_lm){
	    lm_A = A;
	    lm_data = data_vec;
	    lm_tmp_data = tmp_data;
	}
	} // !lm_resolve

	// construct total kernel matrix and data vecter, and solve Ax=b
	calc_sens_weights();
	if (smooth_velocity) calc_averaging_matrix();
	if (nrefl>0 && smooth_depth && !freeze_refl) calc_refl_averaging_matrix();
	if (damp_velocity) calc_damping_matrix();
	if (nrefl>0 && damp_depth && !freeze_refl) calc_refl_damping_matrix();

	// (optional) joint inversion with gravity anomalies
	if (gravity){
	    if (verbose_level>=0) cerr << "calculating residual gravity anomalies...";
	    ginv->calcGravity(grav_z0, grav_x, res_grav);
//	    for (int i=1; i<=res_grav.size(); i++){
//		cerr << grav_x(i) << " " << res_grav(i) << '\n';
//	    }
	    if (verbose_level>=0) cerr << "done.\n";

	    if (verbose_level>=0) cerr << "calculating gravity kernel...";
	    rms_grav = 0.0;
	    for (int i=1; i<=ngravdata; i++){
		if (verbose_level>=1) cerr << i << " ";
		// residual gravity anomalies
		double rg = obs_grav(i)-res_grav(i);
		res_grav(i) = rg*weight_grav;
		rms_grav += rg*rg;

		// gravity kernels
		ginv->calcGravityKernel(grav_z0, grav_x(i), B(i));
		// model normalization
		for (mapIterator p=B(i).begin(); p!=B(i).end(); p++){
		    int j = p->first;
		    double m;
		    const int nvel = nVelUnknowns();
		    if (j<=nnodev){
			m = invert_joint_vpvs ? mvscale_vp(j) : mvscale(j);
		    }else if (invert_joint_vpvs && j<=nvel){
			m = mvscale(j-nnodev);
		    }else{
			m = mdscale(j-nvel)*refl_weight;
		    }
		    p->second *= m*weight_grav;
		}
	    }
	    rms_grav = sqrt(rms_grav/ngravdata);
	    if (verbose_level>=0) cerr << "done.\n";

	    if (printTransient || (printFinal && isFinal)){
		char transfn[MaxStr];
		sprintf(transfn, "%s.rgrav.%d", out_root, iter);
		ofstream os(transfn);
		for (int i=1; i<=ngravdata; i++){
		    os << grav_x(i) << " " << res_grav(i) << " " << res_grav(i)/weight_grav << '\n';
		}
	    }
	}
	
	int iset=0;
	for (double tmp_wsv=wsv_min; tmp_wsv<=wsv_max; tmp_wsv+=dwsv){
	    for (double tmp_wsd=wsd_min; tmp_wsd<=wsd_max; tmp_wsd+=dwsd){
		iset++;

		weight_s_v = logscale_vel ? pow(10.0,tmp_wsv) : tmp_wsv;
		weight_s_d = logscale_dep ? pow(10.0,tmp_wsd) : tmp_wsd;
		if (enable_c2f){
		    weight_s_v *= smooth_stage_factor;
		    weight_s_d *= smooth_stage_factor;
		}
		if (invert_joint_vpvs && wsv_vs>=0.0){
		    weight_s_vs = wsv_vs;
		    if (enable_c2f) weight_s_vs *= smooth_stage_factor;
		}else{
		    weight_s_vs = weight_s_v;
		}
		if (invert_joint_vpvs && verbose_level>=0){
		    cerr << "[tt_inverse] joint smooth Vp=" << weight_s_v
			 << " Vs=" << weight_s_vs << '\n';
		}
		
		double wdv,wdd;
		int nlsqr=0, lsqr_iter=0;
		time_t start_t = time(NULL);
		dump_iter = iter;
		dump_iset = iset;
		dump_is_final = isFinal;
		if (damping_is_fixed){
		    const double wdv0 = weight_d_v;
		    const double wdd0 = weight_d_d;
		    if (enable_c2f){
			weight_d_v = wdv0*damp_stage_factor;
			weight_d_d = wdd0*damp_stage_factor;
		    }
		    if (use_lm){
			weight_d_v *= lm_lambda;
			weight_d_d *= lm_lambda;
		    }
		    fixed_damping(lsqr_iter,nlsqr,wdv,wdd);
		    weight_d_v = wdv0;
		    weight_d_d = wdd0;
		}else{
		    const double tdv0 = target_dv;
		    const double tdd0 = target_dd;
		    if (enable_c2f){
			if (target_dv>0.0) target_dv = tdv0/damp_stage_factor;
			if (target_dd>0.0) target_dd = tdd0/damp_stage_factor;
		    }
		    if (use_lm){
			if (target_dv>0.0) target_dv /= lm_lambda;
			if (target_dd>0.0) target_dd /= lm_lambda;
		    }
		    auto_damping(lsqr_iter,nlsqr,wdv,wdd);
		    target_dv = tdv0;
		    target_dd = tdd0;
		}
		double lsqr_time = difftime(time(NULL),start_t);

		if (do_filter){
		    if (verbose_level>=0) cerr << "filtering velocity perturbation...";
		    for (int i=1; i<=dmodel_total.size(); i++) filtered_model(i) = dmodel_total(i);
		    const int nfield = invert_joint_vpvs ? 2 : 1;
		    for (int ifield=0; ifield<nfield; ifield++){
			const int off = ifield*nnodev;
			for (int i=1; i<=nx; i++){
			    for (int k=kstart(i); k<=nz; k++){
				int inode = smesh.nodeIndex(i,k);
				Point2d p = smesh.nodePos(inode);
				double Lh, Lv;
				corr_vel_p->at(p,Lh,Lv);

				double Lh2 = Lh*Lh;
				double Lv2 = Lv*Lv;
				double sum=0.0, beta_sum=0.0;
				for (int ii=1; ii<=nx; ii++){
				    int jnode = smesh.nodeIndex(ii,k);
				    double dx = smesh.nodePos(jnode).x()-p.x();
				    if (abs(dx)<=Lh){
					double xexp = exp(-dx*dx/Lh2);
					for (int kk=kstart(ii); kk<=nz; kk++){
					    int knode = smesh.nodeIndex(ii,kk);
					    double dz = smesh.nodePos(knode).y()-p.y();
					    if (abs(dz)<=Lv){
						double beta = xexp*exp(-dz*dz/Lv2);
						sum += beta*dmodel_total(knode+off);
						beta_sum += beta;
					    }
					}
				    }
				}
				filtered_model(inode+off) = sum/beta_sum;
			    }
			}
		    }
		    if (verbose_level>=0) cerr << "done.\n";
		    
		    // conservative filtering (Deal and Nolet, GJI, 1996)
		    if (verbose_level>=0) cerr << "calculating a nullspace shuttle...";
		    dx_vec=0.0;
		    const int nvel = nVelUnknowns();
		    for (int i=1; i<=nvel; i++) dx_vec(i) = filtered_model(i)-dmodel_total(i);
		    Array1d<double> Adm(ndata_valid);
		    SparseRectangular sparseA(A,tmp_data,tmp_node,nnode_total);
		    sparseA.Ax(dx_vec,Adm);

		    int lsqr_itermax=inv_lsqr_itermax(itermax_LSQR);
		    double chi;
		    dxr_vec=0.0;
		    iterativeSolver_LSQR(sparseA,Adm,dxr_vec,
					 inv_lsqr_atol(LSQR_ATOL),lsqr_itermax,chi);

		    // note: correction_vec for depth nodes should be zero, so I don't use them.
		    double orig_norm=0, new_norm=0, iprod=0;
		    for (int i=1; i<=nvel; i++){
			orig_norm += filtered_model(i)*filtered_model(i);
			dmodel_total(i) = filtered_model(i)-dxr_vec(i);
			new_norm += dmodel_total(i)*dmodel_total(i);
			dxn_vec(i) = dx_vec(i)-dxr_vec(i);
			iprod += dxr_vec(i)*dxn_vec(i);
		    }
		    if (verbose_level>=0) cerr << "done.\n";
		    if (printLog){
			*log_os_p << "# a posteriori filter check: "
				  << sqrt(orig_norm/nvel)
				  << " " << sqrt(new_norm/nvel) 
				  << " " << iprod << endl;
		    }
		}
		
		if (invert_joint_vpvs){
		    if (kernelRestrictedVp()){
			for (int i=1; i<=nnodev; i++){
			    if (!kernelAllowNodeVp(i)) dmodel_total(i) = 0.0;
			}
		    }
		    if (kernelRestricted()){
			for (int i=1; i<=nnodev; i++){
			    if (!kernelAllowNode(i)) dmodel_total(i+nnodev) = 0.0;
			}
		    }
		}else if (kernelRestricted()){
		    for (int i=1; i<=nnodev; i++){
			if (!kernelAllowNode(i)) dmodel_total(i) = 0.0;
		    }
		}

		// take stats
		double pred_chi = calc_chi();
		double dv_norm = calc_ave_dmv();
		double dd_norm = calc_ave_dmd();
		double Lmvh=-1, Lmvv=-1, Lmd=-1;
		calc_Lm(Lmvh,Lmvv,Lmd);

		if (jumping && !line_search && !use_lm) dmodel_total_sum += dmodel_total;

		const bool single_iset = (wsv_min+dwsv > wsv_max)
		    && (wsd_min+dwsd > wsd_max);
		const bool do_lm = use_lm && single_iset && !strategy_skip_update;
		const bool do_ls = line_search && !do_lm && single_iset && !strategy_skip_update;
		if (do_lm || do_ls){
		    ls_phi0 = init_chi_total;
		    ls_slope = pred_chi - ls_phi0;
		    lm_pred = pred_chi;
		    ls_modelv = modelv;
		    if (invert_joint_vpvs) ls_modelvp = modelvp;
		    ls_modeld = modeld;
		    ls_alpha = 1.0;
		    if (!lm_resolve) ls_ntry = 0;
		    lm_resolve = false;
		    applyDmodel(1.0, ls_modelv, ls_modelvp, ls_modeld, false);
		    ls_eval = true;
		    ls_restart = true;
		    has_prev_forward = false;
		    if (verbose_level>=0){
			if (do_lm)
			    cerr << "[tt_inverse] lm start chi2_0=" << ls_phi0
				 << " pred=" << pred_chi
				 << " lambda=" << lm_lambda << "\n";
			else
			    cerr << "[tt_inverse] linesearch start chi2_0=" << ls_phi0
				 << " pred=" << pred_chi << "\n";
		    }
		}else{
		    applyDmodel(1.0, modelv, modelvp, modeld,
				printTransient || (printFinal && dump_is_final));
		}

		if (printLog){
		    *log_os_p << iter << " " << iset << " "
			      << ndata-ndata_valid << " "
			      << rms_tres_total << " " << init_chi_total << " "
			      << ndata_in[0] << " " << rms_tres[0] << " " << init_chi[0] << " "
			      << ndata_in[1] << " " << rms_tres[1] << " " << init_chi[1] << " "
			      << graph_time << " " << bend_time << " " 
			      << weight_s_v << " " << weight_s_d << " "
			      << wdv << " " << wdd << " "
			      << nlsqr << " " << lsqr_iter << " " << lsqr_time << " "
			      << pred_chi << " " << dv_norm << " " << dd_norm << " "
			      << Lmvh << " " << Lmvv << " " << Lmd;
		    if (gravity){
			*log_os_p << " " << rms_grav;
		    }
		    *log_os_p << endl;
		}
		append_inv_status_jsonl(iter, iset,
					rms_tres_total, init_chi_total,
					pred_chi, dv_norm, dd_norm,
					Lmvh, Lmvv, Lmd,
					isFinal ? 1 : 0, out_root);
		if (ls_restart) break;
	    }
	    if (ls_restart) break;
	}

	if (ls_restart) continue;
	if (isFinal) break;
	iter++;
    }

    // output DWS for the last iteration
    if (out_mask){
	dws.resize(nnodev);
	dws=0.0;
	typedef map<int,double>::iterator mapIterator;
	for (int i=1; i<=A.size(); i++){
	    for (mapIterator p=A(i).begin(); p!=A(i).end(); p++){
		int inode=p->first;
		if (inode<=dws.size()) dws(inode) += p->second;
	    }
	}
	smesh.printMaskGrid(*vmesh_os_p, dws);
    }
    if (out_grav_dws){
	dws.resize(nnodev);
	dws=0.0;
	typedef map<int,double>::iterator mapIterator;
	for (int i=1; i<=B.size(); i++){
	    for (mapIterator p=B(i).begin(); p!=B(i).end(); p++){
		int inode=p->first;
		if (inode<=dws.size()) dws(inode) += p->second/(weight_grav*mvscale(inode));
	    }
	}
	smesh.printMaskGrid(*grav_dws_osp, dws);
    }
}

void
TomographicInversion2d::addRefl(Interface2d* intfp)
{
    reflp = intfp;
    nrefl = 1;
    nnoded = reflp->numNodes();
    Rd.resize(nnoded);
    Td.resize(nnoded);
    modeld.resize(nnoded);
    mdscale.resize(nnoded);
    dmodel_total.resize(nnodev+nnoded);
    nnode_total = nnodev+nnoded;
    tmp_node.resize(nnodev+nnoded);
    tmp_nodedc.resize(nnoded);
    tmp_nodedr.resize(nnoded);
}

void TomographicInversion2d::addSeafloor(Interface2d* intfp)
{
    seafloorp = intfp;
}

void TomographicInversion2d::invertWaterOnly()
{
    invert_water_only = true;
}

void TomographicInversion2d::invertCrustOnly()
{
    invert_crust_only = true;
}

const Interface2d* TomographicInversion2d::waterBottom() const
{
    return seafloorp ? seafloorp : reflp;
}

bool TomographicInversion2d::kernelRestricted() const
{
    if (strategy_full_vp)
	return invert_water_only || invert_crust_only;
    if (invert_vs_psx)
	return invert_water_only || invert_crust_only || freeze_psx_lid || freeze_below;
    return invert_water_only || invert_crust_only || convp != 0 || freeze_below;
}

int TomographicInversion2d::nVelUnknowns() const
{
    return invert_joint_vpvs ? 2*nnodev : nnodev;
}

int TomographicInversion2d::convFaceDomain(int inode) const
{
    if (convp==0) return 0;
    Point2d p = smesh.nodePos(inode);
    if (p.y() < convp->z(p.x()) - 1e-6) return 1;
    return 2;
}

bool TomographicInversion2d::kernelRestrictedVp() const
{
    return invert_water_only || invert_crust_only;
}

bool TomographicInversion2d::kernelAllowNodeVp(int inode) const
{
    if (!invert_water_only && !invert_crust_only)
	return true;
    Point2d p = smesh.nodePos(inode);
    const Interface2d* sf = waterBottom();
    if (!sf) return true;
    const double zsf = sf->z(p.x());
    if (invert_water_only)
	return p.y() <= zsf + 1e-6;
    return p.y() >= zsf - 1e-6;
}

int TomographicInversion2d::psxVelDomain(int inode) const
{
    if (!invert_vs_psx || freeze_psx_lid || convp==0) return 0;
    Point2d p = smesh.nodePos(inode);
    if (p.y() < convp->z(p.x()) - 1e-6) return 1;
    return 2;
}

bool TomographicInversion2d::kernelAllowNode(int inode) const
{
    Point2d p = smesh.nodePos(inode);
    if (!strategy_full_vp){
    if (freeze_below && convp && p.y() >= convp->z(p.x()) - 1e-6)
	return false;
    if (convp && !freeze_below && (!invert_vs_psx || freeze_psx_lid)
	&& p.y() < convp->z(p.x()) - 1e-6)
	return false;
    }
    if (!invert_water_only && !invert_crust_only)
	return true;
    const Interface2d* sf = waterBottom();
    if (!sf) return true;
    const double zsf = sf->z(p.x());
    if (invert_water_only)
	return p.y() <= zsf + 1e-6;
    // -w: freeze water. For PPS/PSS lid-S, also freeze the seafloor
    // row (initialized as water). PSP already freezes the whole lid.
    if (invert_vs_psx && !freeze_psx_lid)
	return p.y() > zsf + 1e-6;
    if (freeze_below)
	return p.y() > zsf + 1e-6;
    return p.y() >= zsf - 1e-6;
}

void TomographicInversion2d::freezeRefl()
{
    freeze_refl = true;
    nnode_total = nVelUnknowns();
    dmodel_total.resize(nnode_total);
    tmp_node.resize(nnode_total);
}

void TomographicInversion2d::doFullRefl()
{
    do_full_refl = true;
    graph.do_refl_downward();
}

void TomographicInversion2d::doConvert(Interface2d* convtp)
{
    do_convert = true;
    convp = convtp;
}

void TomographicInversion2d::setKappa(double k)
{
    setKappa(k, k);
}

void TomographicInversion2d::setKappa(double k_lid, double k_below)
{
    if (k_lid<=0.0 || k_below<=0.0)
	error("TomographicInversion2d::setKappa - kappa must be positive.");
    vpvs_kappa = k_lid;
    vpvs_kappa_below = k_below;
    if (!smesh.hasDualVs()) smesh.setMixKappa(k_lid, k_below);
}

void TomographicInversion2d::setReflWeight(double x)
{
    if (x>0){
	refl_weight = x;
    }else{
	cerr << "TomographicInversion2d::setReflWeight - non-positive value ignored.\n";
    }
}

void TomographicInversion2d::SmoothVelocity(const char* fn,
					    double start, double end, double d,
					    bool scale)
{
    smooth_velocity=true;
    wsv_min=start; wsv_max=end; dwsv=d;
    logscale_vel=scale;
    corr_vel_p = new CorrelationLength2d(fn);
}

void TomographicInversion2d::SmoothVelocityVs(double w)
{
    wsv_vs = w;
}

void TomographicInversion2d::applyFilter(const char *fn)
{
    do_filter=true;
    if (fn[0] != '\0'){
	uboundp = new Interface2d(fn);
    }else{
	uboundp = new Interface2d(smesh); // use bathymetry as upperbound
    }
}

void TomographicInversion2d::SmoothDepth(const char* fn,
					 double start, double end, double d,
					 bool scale)
{
    SmoothDepth(start,end,d,scale);
    corr_dep_p = new CorrelationLength1d(fn);
}

void TomographicInversion2d::SmoothDepth(double start, double end, double d,
					 bool scale)
{
    smooth_depth=true;
    wsd_min=start; wsd_max=end; dwsd=d;
    logscale_dep=scale;
}

void TomographicInversion2d::DampVelocity(double a)
{
    damp_velocity = true;
    target_dv = a/100.0; // % -> fraction
}

void TomographicInversion2d::DampDepth(double a)
{
    damp_depth = true;
    target_dd = a/100.0; // % -> fraction 
}

void TomographicInversion2d::FixDamping(double v, double d)
{
    damping_is_fixed = true;
    damp_velocity = true;
    damp_depth = true;
    weight_d_v = v;
    weight_d_d = d;
}

void TomographicInversion2d::Squeezing(const char* fn)
{
    damping_wt_p = new DampingWeight2d(fn);
    do_squeezing = true;
}

void TomographicInversion2d::doJumping()
{
    jumping = true;
}

void TomographicInversion2d::reset_kernel()
{
    typedef map<int,double>::iterator mapIterator;
    mapIterator p;
    mapIterator e;
    for (int i=1; i<=A.size(); i++){
    p=A(i).begin();e=A(i).end();
    A(i).erase(p,e);
    }
    for (int i=1; i<=Rv_h.size(); i++){
    p=Rv_h(i).begin();e=Rv_h(i).end();
    Rv_h(i).erase(p,e);
    }
    for (int i=1; i<=Rv_v.size(); i++){
    p=Rv_v(i).begin();e=Rv_v(i).end();    	
    Rv_v(i).erase(p,e);
    }
    for (int i=1; i<=Tv.size(); i++){
    p=Tv(i).begin();e=Tv(i).end();    	
    Tv(i).erase(p,e);
    }
    if (invert_joint_vpvs){
	for (int i=1; i<=Rv_h_vs.size(); i++){
	    p=Rv_h_vs(i).begin();e=Rv_h_vs(i).end();
	    Rv_h_vs(i).erase(p,e);
	}
	for (int i=1; i<=Rv_v_vs.size(); i++){
	    p=Rv_v_vs(i).begin();e=Rv_v_vs(i).end();
	    Rv_v_vs(i).erase(p,e);
	}
	for (int i=1; i<=Tv_vs.size(); i++){
	    p=Tv_vs(i).begin();e=Tv_vs(i).end();
	    Tv_vs(i).erase(p,e);
	}
    }
    if (nrefl>0){
	for (int i=1; i<=Rd.size(); i++){
    p=Rd(i).begin();e=Rd(i).end();		
    Rd(i).erase(p,e);
	}
	for (int i=1; i<=Td.size(); i++){
		p=Td(i).begin();e=Td(i).end();		
    Td(i).erase(p,e);
	}
    }
    if (gravity){
	for (int i=1; i<=B.size(); i++){
	  	p=B(i).begin();e=B(i).end();				
      B(i).erase(p,e);
	}
    }
}

void TomographicInversion2d::add_kernel(int idata, const Array1d<Point2d>& cur_path)
{
    map<int,double>& A_i = A(idata);

    int np = cur_path.size();
    int nintp = betasp.numIntp();
    makeBSpoints(cur_path,pp);

    double path_len=0.0;
    Index2d guess_index = smesh.nodeIndex(smesh.nearest(*pp(1)));
    for (int i=1; i<=np+1; i++){
	int j1=i;
	int j2=i+1;
	int j3=i+2;
	int j4=i+3;
	betasp.interpolate(*pp(j1),*pp(j2),*pp(j3),*pp(j4),Q);
	
	for (int j=2; j<=nintp; j++){
	    Point2d midp = 0.5*(Q(j-1)+Q(j));
	    double dist= Q(j).distance(Q(j-1));
	    path_len+=dist;

	    int jUL, jLL, jLR, jUR;
	    double r, s, rr, ss;
	    int icell=smesh.locateInCell(midp,guess_index,
					 jUL,jLL,jLR,jUR,r,s,rr,ss);
	    if (icell>0){
		auto accum = [&](int jn, double w){
		    if (invert_joint_vpvs){
			if (kernelRestrictedVp() && !kernelAllowNodeVp(jn)) return;
		    }else{
			if (kernelRestricted() && !kernelAllowNode(jn)) return;
		    }
		    A_i[jn] += w;
		};
		accum(jUL, rr*ss*dist);
		accum(jLL, rr*s*dist);
		accum(jLR, r*s*dist);
		accum(jUR, r*ss*dist);

		// nodev_hit is not consumed elsewhere; keep kernel assembly lock-free in OMP path.
	    }
	}
    }
    path_length(idata)=path_len;
}

void TomographicInversion2d::add_kernel_refl(int idata, const Array1d<Point2d>& cur_path,
					     int ir0, int ir1)
{
    map<int,double>& A_i = A(idata);

    int jL, jR;
    reflp->locateInSegment(cur_path(ir0).x(),jL,jR);
    if (jL==jR){
	cerr << "TomographicInversion2d::add_kernel_refl - bottoming point out of bounds\n";
	return; // ignore this ray info
    }
    double x1 = reflp->x(jL);
    double x2 = reflp->x(jR);
    double z1 = reflp->z(x1);
    double z2 = reflp->z(x2);
    double x = cur_path(ir0).x();

    double dx = x2-x1;
    double dz = z2-z1;
    double cos_alpha = dx/sqrt(dx*dx+dz*dz);

    double p1 = smesh.at(cur_path(ir0));

    Point2d a = cur_path(ir0-1)-cur_path(ir0); a /= a.norm();
    Point2d b = cur_path(ir1+1)-cur_path(ir1); b /= b.norm();
    Point2d c = a+b; c /= c.norm();
    double cos_theta = a.inner_product(c);
    
    double com_factor = 2.0*cos_theta*cos_alpha*p1/dx;
    if (!freeze_refl){
	A_i[nVelUnknowns()+jL] = com_factor*(x2-x);
	A_i[nVelUnknowns()+jR] = com_factor*(x-x1);
    }

    add_kernel(idata,cur_path);
}    

static double inv_median_pos(std::vector<double>& a)
{
    if (a.empty()) return 0.0;
    const size_t n = a.size();
    const size_t mid = n / 2;
    std::nth_element(a.begin(), a.begin() + mid, a.end());
    if (n % 2) return a[mid];
    const double hi = a[mid];
    std::nth_element(a.begin(), a.begin() + (mid - 1), a.end());
    return 0.5 * (hi + a[mid - 1]);
}

double TomographicInversion2d::sensVelW(int inode, bool for_vs) const
{
    if (!sens_weight || inode < 1 || inode > nnodev) return 1.0;
    if (for_vs){
	if (sens_w_vs.size() >= static_cast<size_t>(nnodev)) return sens_w_vs(inode);
	return 1.0;
    }
    if (sens_w_vp.size() >= static_cast<size_t>(nnodev)) return sens_w_vp(inode);
    return 1.0;
}

double TomographicInversion2d::sensDepW(int inode) const
{
    if (!sens_weight || inode < 1 || inode > nnoded) return 1.0;
    if (sens_w_d.size() >= static_cast<size_t>(nnoded)) return sens_w_d(inode);
    return 1.0;
}

double TomographicInversion2d::sensCouple(double wi, double wj) const
{
    const double s = wi + wj;
    if (s <= 1e-12) return 1.0;
    return 2.0 / s;
}

void TomographicInversion2d::assignSensWeights(const Array1d<double>& s, Array1d<double>& w,
					      bool for_vs, bool joint_vp, double med_out[3])
{
    if (med_out){
	med_out[0] = med_out[1] = med_out[2] = 0.0;
    }
    if (nnodev <= 0) return;
    w.resize(nnodev);
    w = 1.0;
    if (!sens_weight) return;

    const bool rest = for_vs ? kernelRestricted() : kernelRestrictedVp();
    std::vector<double> acc[3];
    for (int i = 1; i <= nnodev; i++){
	const bool allow = for_vs ? kernelAllowNode(i) : kernelAllowNodeVp(i);
	if (rest && !allow) continue;
	int d = 0;
	if (for_vs) d = psxVelDomain(i);
	else if (joint_vp) d = convFaceDomain(i);
	if (d < 0 || d > 2) d = 0;
	if (i <= static_cast<int>(s.size()) && s(i) > 0.0)
	    acc[d].push_back(s(i));
    }
    double med[3];
    std::vector<double> all;
    for (int d = 0; d < 3; d++){
	med[d] = inv_median_pos(acc[d]);
	all.insert(all.end(), acc[d].begin(), acc[d].end());
    }
    const double med_all = inv_median_pos(all);
    if (med_out){
	for (int d = 0; d < 3; d++) med_out[d] = med[d] > 0.0 ? med[d] : med_all;
    }
    if (med_all <= 0.0) return;

    const double k = (sens_kappa > 0.0) ? sens_kappa : 10.0;
    const double lo = 1.0 / k;
    const double hi = k;
    const double eps = (sens_eps > 0.0) ? sens_eps : 0.05;
    for (int i = 1; i <= nnodev; i++){
	const bool allow = for_vs ? kernelAllowNode(i) : kernelAllowNodeVp(i);
	if (rest && !allow){
	    w(i) = hi;
	    continue;
	}
	int d = 0;
	if (for_vs) d = psxVelDomain(i);
	else if (joint_vp) d = convFaceDomain(i);
	if (d < 0 || d > 2) d = 0;
	double m = med[d];
	if (m <= 0.0) m = med_all;
	const double sj = (i <= static_cast<int>(s.size())) ? s(i) : 0.0;
	if (sj <= 0.0){
	    w(i) = hi;
	    continue;
	}
	double ww = m / (sj + eps * m);
	if (ww < lo) ww = lo;
	if (ww > hi) ww = hi;
	w(i) = ww;
    }
}

void TomographicInversion2d::calc_sens_weights()
{
    if (nnodev > 0){
	sens_w_vp.resize(nnodev); sens_w_vp = 1.0;
	sens_w_vs.resize(nnodev); sens_w_vs = 1.0;
    }
    if (nnoded > 0){
	sens_w_d.resize(nnoded); sens_w_d = 1.0;
    }
    if (!sens_weight) return;

    Array1d<double> s_vp(nnodev), s_vs(nnodev);
    Array1d<double> s_d(nnoded > 0 ? nnoded : 1);
    if (nnodev > 0){ s_vp = 0.0; s_vs = 0.0; }
    s_d = 0.0;
    const int nvel = nVelUnknowns();
    typedef map<int,double>::const_iterator cit;
    for (int i = 1; i <= ndata; i++){
	if (tmp_data(i) <= 0) continue;
	for (cit p = A(i).begin(); p != A(i).end(); ++p){
	    const int j = p->first;
	    const double a = std::fabs(p->second);
	    if (a == 0.0) continue;
	    if (invert_joint_vpvs){
		if (j >= 1 && j <= nnodev) s_vp(j) += a;
		else if (j <= nvel) s_vs(j - nnodev) += a;
		else{
		    const int id = j - nvel;
		    if (id >= 1 && id <= nnoded) s_d(id) += a;
		}
	    }else{
		if (j >= 1 && j <= nnodev) s_vs(j) += a;
		else{
		    const int id = j - nnodev;
		    if (id >= 1 && id <= nnoded) s_d(id) += a;
		}
	    }
	}
    }

    double med_vp[3] = {0.0, 0.0, 0.0};
    double med_vs[3] = {0.0, 0.0, 0.0};
    if (invert_joint_vpvs)
	assignSensWeights(s_vp, sens_w_vp, false, true, med_vp);
    assignSensWeights(s_vs, sens_w_vs, true, false, med_vs);

    double med_moho = 0.0;
    if (nnoded > 0 && nrefl > 0 && !freeze_refl){
	std::vector<double> acc;
	acc.reserve(static_cast<size_t>(nnoded));
	for (int i = 1; i <= nnoded; i++){
	    if (s_d(i) > 0.0) acc.push_back(s_d(i));
	}
	med_moho = inv_median_pos(acc);
	const double k = (sens_kappa > 0.0) ? sens_kappa : 10.0;
	const double lo = 1.0 / k;
	const double hi = k;
	const double eps = (sens_eps > 0.0) ? sens_eps : 0.05;
	if (med_moho > 0.0){
	    for (int i = 1; i <= nnoded; i++){
		if (s_d(i) <= 0.0){
		    sens_w_d(i) = hi;
		    continue;
		}
		double ww = med_moho / (s_d(i) + eps * med_moho);
		if (ww < lo) ww = lo;
		if (ww > hi) ww = hi;
		sens_w_d(i) = ww;
	    }
	}
    }

    if (verbose_level >= 0){
	cerr << "[tt_inverse] sens_weight DWS median"
	     << " vel=" << med_vs[0]
	     << " lid=" << (med_vs[1] > 0.0 ? med_vs[1] : med_vp[1])
	     << " below=" << (med_vs[2] > 0.0 ? med_vs[2] : med_vp[2]);
	if (invert_joint_vpvs)
	    cerr << " vp_lid=" << med_vp[1] << " vp_below=" << med_vp[2];
	if (nnoded > 0 && nrefl > 0 && !freeze_refl)
	    cerr << " moho=" << med_moho;
	cerr << "  (w=clip(med/(s+eps*med)," << (1.0 / ((sens_kappa > 0.0) ? sens_kappa : 10.0))
	     << "," << ((sens_kappa > 0.0) ? sens_kappa : 10.0) << "))\n";
    }
}

void TomographicInversion2d::fill_vel_averaging(sparseMat& Rh, sparseMat& Rv,
					       const Array1d<double>& scale, bool for_vs)
{
    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
	    int inode = smesh.nodeIndex(i,k);
	    const bool rest = for_vs ? kernelRestricted() : kernelRestrictedVp();
	    const bool allow_i = for_vs ? kernelAllowNode(inode) : kernelAllowNodeVp(inode);
	    if (rest && !allow_i) continue;
	    const int idom = for_vs ? psxVelDomain(inode)
		: (invert_joint_vpvs ? convFaceDomain(inode) : 0);
	    Point2d p = smesh.nodePos(inode);
	    double Lh, Lv;
	    corr_vel_p->at(p,Lh,Lv);

	    double Lh2 = Lh*Lh;
	    double Lv2 = Lv*Lv;
	    const double wi = sensVelW(inode, for_vs);
	    double beta_sum = 0.0;
	    for (int ii=1; ii<=nx; ii++){
		int jnode = smesh.nodeIndex(ii,k);
		if (jnode!=inode){
		    const bool allow_j = for_vs ? kernelAllowNode(jnode) : kernelAllowNodeVp(jnode);
		    if (rest && !allow_j) continue;
		    if (idom){
			int jdom = for_vs ? psxVelDomain(jnode) : convFaceDomain(jnode);
			if (jdom!=idom) continue;
		    }
		    double dx = smesh.nodePos(jnode).x()-p.x();
		    if (abs(dx)<=Lh){
			double dxL2 = dx*dx/Lh2;
			double beta = exp(-dxL2);
			if (sens_weight)
			    beta *= sensCouple(wi, sensVelW(jnode, for_vs));
			Rh(inode)[jnode] = beta*scale(jnode);
			beta_sum += beta;
		    }
		}
	    }
	    Rh(inode)[inode] = -beta_sum*scale(inode);

	    beta_sum = 0.0;
	    for (int kk=1; kk<=nz; kk++){
		int jnode = smesh.nodeIndex(i,kk);
		if (jnode!=inode){
		    const bool allow_j = for_vs ? kernelAllowNode(jnode) : kernelAllowNodeVp(jnode);
		    if (rest && !allow_j) continue;
		    if (idom){
			int jdom = for_vs ? psxVelDomain(jnode) : convFaceDomain(jnode);
			if (jdom!=idom) continue;
		    }
		    double dz = smesh.nodePos(jnode).y()-p.y();
		    if (abs(dz)<=Lv){
			double dzL2 = dz*dz/Lv2;
			double beta = exp(-dzL2);
			if (sens_weight)
			    beta *= sensCouple(wi, sensVelW(jnode, for_vs));
			Rv(inode)[jnode] = beta*scale(jnode);
			beta_sum += beta;
		    }
		}
	    }
	    Rv(inode)[inode] = -beta_sum*scale(inode);
	}
    }
}

void TomographicInversion2d::fill_vel_damping(sparseMat& Tmat, bool for_vs)
{
    Array2d<double> local_T(4,4);
    Array1d<int> j(4);

    for (int i=1; i<=smesh.numCells(); i++){
	smesh.cellNodes(i, j(1), j(2), j(3), j(4));
	const bool rest = for_vs ? kernelRestricted() : kernelRestrictedVp();
	if (rest){
	    bool all_ok = true;
	    for (int m=1; m<=4; m++){
		const bool allow = for_vs ? kernelAllowNode(j(m)) : kernelAllowNodeVp(j(m));
		if (!allow){ all_ok = false; break; }
	    }
	    if (!all_ok) continue;
	}
	if ((for_vs && invert_vs_psx && !freeze_psx_lid && convp)
	    || (!for_vs && invert_joint_vpvs && convp)){
	    int d0 = for_vs ? psxVelDomain(j(1)) : convFaceDomain(j(1));
	    bool split = false;
	    for (int m=2; m<=4; m++){
		int dm = for_vs ? psxVelDomain(j(m)) : convFaceDomain(j(m));
		if (d0 && dm && dm!=d0){ split = true; break; }
	    }
	    if (split) continue;
	}
	smesh.cellNormKernel(i, local_T);

	if (do_squeezing){
	     Point2d p = 0.25*(smesh.nodePos(j(1))+smesh.nodePos(j(2))
			       +smesh.nodePos(j(3))+smesh.nodePos(j(4)));
	     double w;
	     damping_wt_p->at(p,w);
	     local_T *= w;
	}
	if (sens_weight){
	    const double wcell = 0.25*(sensVelW(j(1), for_vs)+sensVelW(j(2), for_vs)
				       +sensVelW(j(3), for_vs)+sensVelW(j(4), for_vs));
	    local_T *= wcell;
	}
	add_global(local_T, j, Tmat);
    }
}

void TomographicInversion2d::applyDmodel(double alpha,
					 const Array1d<double>& base_v,
					 const Array1d<double>& base_vp,
					 const Array1d<double>& base_d,
					 bool write_out)
{
    Array1d<double> tmp_v = base_v;
    Array1d<double> tmp_vp;
    if (invert_joint_vpvs) tmp_vp = base_vp;
    Array1d<double> tmp_d = base_d;
    const bool hold_vs = joint_staged && dump_iter<=joint_vp_only_iters;
    const double pmin = 1.0/30.0;
    int nclip_p = 0;
    if (invert_joint_vpvs){
	bool vs_update_ok = true;
	if (!hold_vs){
	    for (int i=1; i<=nnodev; i++){
		if (!std::isfinite(dmodel_total(i+nnodev))){
		    vs_update_ok = false;
		    break;
		}
	    }
	    if (!vs_update_ok){
		if (verbose_level>=0)
		    cerr << "TomographicInversion2d:: discard non-finite Vs update\n";
		for (int i=1; i<=nnodev; i++) dmodel_total(i+nnodev) = 0.0;
	    }
	}
	for (int i=1; i<=nnodev; i++){
	    if (hold_vs){
		tmp_vp(i) += alpha*mvscale_vp(i)*dmodel_total(i);
		if (!(tmp_vp(i) > pmin)){
		    tmp_vp(i) = pmin;
		    nclip_p++;
		}
	    }else if (vs_update_ok){
		tmp_v(i) += alpha*mvscale(i)*dmodel_total(i+nnodev);
		if (!(tmp_v(i) > pmin)){
		    tmp_v(i) = pmin;
		    nclip_p++;
		}
	    }
	}
    }else{
	bool vel_update_ok = true;
	for (int i=1; i<=dmodel_total.size(); i++){
	    if (!std::isfinite(dmodel_total(i))){
		vel_update_ok = false;
		break;
	    }
	}
	if (!vel_update_ok){
	    if (verbose_level>=0)
		cerr << "TomographicInversion2d:: discard non-finite update\n";
	    for (int i=1; i<=dmodel_total.size(); i++) dmodel_total(i) = 0.0;
	}
	for (int i=1; i<=nnodev; i++){
	    tmp_v(i) += alpha*mvscale(i)*dmodel_total(i);
	    if (!(tmp_v(i) > pmin)){
		tmp_v(i) = pmin;
		nclip_p++;
	    }
	}
    }
    if (nclip_p>0 && verbose_level>=0){
	cerr << "TomographicInversion2d:: clamped " << nclip_p
	     << " slowness nodes to pmin=" << pmin
	     << " (vmax=30 km/s)\n";
    }
    if (invert_joint_vpvs){
	if (!smesh.hasDualVs())
	    smesh.set(tmp_vp);
	else{
	    smesh.setVp(tmp_vp);
	    smesh.set(tmp_v);
	}
    }else if (strategy_full_vp && smesh.hasDualVs())
	smesh.setVp(tmp_v);
    else
	smesh.set(tmp_v);
    if (nrefl>0 && !freeze_refl){
	for (int i=1; i<=nnoded; i++){
	    int tmp_i = tmp_nodedc(i);
	    tmp_d(i) += alpha*mdscale(i)*dmodel_total(tmp_i)*refl_weight;
	}
	reflp->set(tmp_d);
    }
    if (!write_out)
	return;
    writeCurrentModels();
}

void TomographicInversion2d::restoreModel(const Array1d<double>& base_v,
					  const Array1d<double>& base_vp,
					  const Array1d<double>& base_d)
{
    if (invert_joint_vpvs){
	if (!smesh.hasDualVs())
	    smesh.set(base_vp);
	else{
	    smesh.setVp(base_vp);
	    smesh.set(base_v);
	}
    }else if (strategy_full_vp && smesh.hasDualVs())
	smesh.setVp(base_v);
    else
	smesh.set(base_v);
    if (nrefl>0 && !freeze_refl)
	reflp->set(base_d);
}

void TomographicInversion2d::writeCurrentModels()
{
    if (!out_root || out_root[0]=='\0') return;
    char transfn[MaxStr];
    sprintf(transfn, "%s.smesh.%d.%d", out_root, dump_iter, dump_iset);
    ofstream os(transfn);
    smesh.outMesh(os);
    if (invert_joint_vpvs || smesh.hasDualVs()){
	char transfn_vp[MaxStr];
	sprintf(transfn_vp, "%s.vp.smesh.%d.%d", out_root, dump_iter, dump_iset);
	ofstream os_vp(transfn_vp);
	if (os_vp) smesh.outMeshVp(os_vp);
	sprintf(transfn_vp, "%s.vp.smesh", out_root);
	ofstream os_vp0(transfn_vp);
	if (os_vp0) smesh.outMeshVp(os_vp0);
    }
    if (nrefl>0){
	char transfn2[MaxStr];
	sprintf(transfn2, "%s.refl.%d.%d", out_root, dump_iter, dump_iset);
	ofstream os2(transfn2);
	os2 << *reflp;
    }
}

void TomographicInversion2d::calc_averaging_matrix()
{
    if (invert_joint_vpvs){
	fill_vel_averaging(Rv_h, Rv_v, mvscale_vp, false);
	fill_vel_averaging(Rv_h_vs, Rv_v_vs, mvscale, true);
	return;
    }
    fill_vel_averaging(Rv_h, Rv_v, mvscale, true);
}

void TomographicInversion2d::calc_refl_averaging_matrix()
{
    for (int i=1; i<=nnoded; i++){
	double x = reflp->x(i);
	double Lh;
	if (corr_dep_p != 0){
	    Lh = corr_dep_p->at(x);
	}else if (corr_vel_p != 0){
	    double Lv;
	    corr_vel_p->at(Point2d(x,reflp->z(x)),Lh,Lv);
	}else{
	    error("TomographicInversion2d::calc_refl_averaging_matrix - no correlation length available");
	}
	double Lh2 = Lh*Lh;
	double beta_sum = 0.0;
	const double wi = sensDepW(i);
	for (int ii=1; ii<=nnoded; ii++){
	    if (ii!=i){
		double dx = reflp->x(ii)-x;
		if (abs(dx)<=Lh){
		    double beta = exp(-dx*dx/Lh2);
		    if (sens_weight)
			beta *= sensCouple(wi, sensDepW(ii));
		    Rd(i)[ii] = beta*mdscale(ii)*refl_weight;
		    beta_sum += beta;
		}
	    }
	}
	Rd(i)[i] = -beta_sum*mdscale(i)*refl_weight;
    }
}

void TomographicInversion2d::calc_damping_matrix()
{
    if (invert_joint_vpvs){
	fill_vel_damping(Tv, false);
	fill_vel_damping(Tv_vs, true);
	return;
    }
    fill_vel_damping(Tv, true);
}

void TomographicInversion2d::calc_refl_damping_matrix()
{
    double dx = reflp->x(2)-reflp->x(1); // assumes uniform interval
    double fac = sqrt(dx);

    for (int i=1; i<=nnoded; i++){
	Td(i)[i] = fac * (sens_weight ? sensDepW(i) : 1.0);
    }
}

void TomographicInversion2d::add_global(const Array2d<double>& a,
					const Array1d<int>& j, sparseMat& global)
{
    for (int m=1; m<=4; m++){
	for (int n=1; n<=4; n++){
	    global(j(m))[j(n)] += a(m,n);
	}
    }
}

double TomographicInversion2d::calc_ave_dmv()
{
    double dm_norm=0.0;
    if (invert_joint_vpvs){
	const bool vs_only = dump_iter > joint_vp_only_iters;
	int n = 0;
	for (int i=1; i<=nnodev; i++){
	    if (vs_only){
		if (kernelRestricted() && !kernelAllowNode(i)) continue;
		dm_norm += dmodel_total(i+nnodev)*dmodel_total(i+nnodev);
	    }else{
		if (kernelRestrictedVp() && !kernelAllowNodeVp(i)) continue;
		dm_norm += dmodel_total(i)*dmodel_total(i);
	    }
	    n++;
	}
	return n>0 ? sqrt(dm_norm/n) : 0.0;
    }
    const int nvel = nVelUnknowns();
    int n = 0;
    for (int i=1; i<=nvel; i++){
	const double v = dmodel_total(i);
	if (!std::isfinite(v)) continue;
	if (invert_vs_psx && kernelRestricted() && !kernelAllowNode(i)) continue;
	dm_norm += v*v;
	n++;
    }
    return n>0 ? sqrt(dm_norm/n) : 0.0;
}

double TomographicInversion2d::calc_ave_dmd()
{
    if (nrefl==0) return 0.0;
	
    double dm_norm=0.0;
    const int nvel = nVelUnknowns();
    int n = 0;
    for (int i=1+nvel; i<=nnoded+nvel; i++){
	const double v = dmodel_total(i);
	if (!std::isfinite(v)) continue;
	dm_norm += v*v;
	n++;
    }
    return n>0 ? sqrt(dm_norm/n)*refl_weight : 0.0;
}

void TomographicInversion2d::calc_Lm(double& lmh, double& lmv, double& lmd)
{
    if (smooth_velocity){
	Array1d<double> Rvm(nnodev), dmv_vec(nnodev);
	SparseRectangular sparseRv_h(Rv_h,tmp_nodev,tmp_nodev,nnodev);
	for (int i=1; i<=nnodev; i++) dmv_vec(i) = dmodel_total(i);

	sparseRv_h.Ax(dmv_vec,Rvm);
	double val=0.0;
	for (int i=1; i<=nnodev; i++) val += Rvm(i)*Rvm(i);
	if (invert_joint_vpvs){
	    SparseRectangular sparseRv_h_vs(Rv_h_vs,tmp_nodev,tmp_nodev,nnodev);
	    Array1d<double> dmv_vs(nnodev), Rvm_vs(nnodev);
	    for (int i=1; i<=nnodev; i++) dmv_vs(i) = dmodel_total(i+nnodev);
	    sparseRv_h_vs.Ax(dmv_vs,Rvm_vs);
	    for (int i=1; i<=nnodev; i++) val += Rvm_vs(i)*Rvm_vs(i);
	    lmh = sqrt(val/(2*nnodev));
	}else{
	    lmh = sqrt(val/nnodev);
	}

	SparseRectangular sparseRv_v(Rv_v,tmp_nodev,tmp_nodev,nnodev);
	sparseRv_v.Ax(dmv_vec,Rvm);
	val = 0.0;
	for (int i=1; i<=nnodev; i++) val += Rvm(i)*Rvm(i);
	if (invert_joint_vpvs){
	    SparseRectangular sparseRv_v_vs(Rv_v_vs,tmp_nodev,tmp_nodev,nnodev);
	    Array1d<double> dmv_vs(nnodev), Rvm_vs(nnodev);
	    for (int i=1; i<=nnodev; i++) dmv_vs(i) = dmodel_total(i+nnodev);
	    sparseRv_v_vs.Ax(dmv_vs,Rvm_vs);
	    for (int i=1; i<=nnodev; i++) val += Rvm_vs(i)*Rvm_vs(i);
	    lmv = sqrt(val/(2*nnodev));
	}else{
	    lmv = sqrt(val/nnodev);
	}
    }

    if (nrefl>0 && smooth_depth && !freeze_refl){
	Array1d<double> Rdm(nnoded), dmd_vec(nnoded);
	SparseRectangular sparseRd(Rd,tmp_nodedr,tmp_nodedr);
	const int nvel = nVelUnknowns();
	for (int i=1+nvel, j=1; i<=nnoded+nvel; i++) dmd_vec(j++) = dmodel_total(i);

	sparseRd.Ax(dmd_vec,Rdm);
	double val = 0.0;
	for (int i=1; i<=nnoded; i++) val += Rdm(i)*Rdm(i);
	lmd = sqrt(val/nnoded);
    }
}
 
double TomographicInversion2d::calc_chi()
{
    Array1d<double> Adm(ndata_valid);
    SparseRectangular sparseA(A,tmp_data,tmp_node,nnode_total);
    sparseA.Ax(dmodel_total,Adm);
    double val=0.0;
    for (int i=1; i<=ndata; i++){
	int j=tmp_data(i);
	if (j>0){
	    double res=Adm(j)-data_vec(i);
	    val += res*res;
	}
    }
    return val/ndata_valid;
}

int TomographicInversion2d::_solve(bool sv, double wsv, bool sd, double wsd,
				   bool dv, double wdv, bool dd, double wdd)
{
    const double wsvs = invert_joint_vpvs ? weight_s_vs : wsv;
    // construct total kernel
    Array1d<const sparseMat*> As;
    Array1d<SparseMatAux> matspec;
    As.push_back(&A);
    matspec.push_back(SparseMatAux(1.0,&tmp_data,&tmp_node));
    int ndata_plus=ndata_valid;
    if (gravity){
	As.push_back(&B);
	matspec.push_back(SparseMatAux(1.0,&tmp_gravdata,&tmp_node));
	ndata_plus += ngravdata;
    }
    if (sv){
	if (wsv<0) error("TomographicInversion2d::_solve - negative wsv");
	if (invert_joint_vpvs && wsvs<0)
	    error("TomographicInversion2d::_solve - negative wsvs");
	As.push_back(&Rv_h);
	matspec.push_back(SparseMatAux(wsv,&tmp_nodev,&tmp_nodev));
	ndata_plus += nnodev;
	As.push_back(&Rv_v);
	matspec.push_back(SparseMatAux(wsv,&tmp_nodev,&tmp_nodev));
	ndata_plus += nnodev;
	if (invert_joint_vpvs){
	    As.push_back(&Rv_h_vs);
	    matspec.push_back(SparseMatAux(wsvs,&tmp_nodev,&tmp_nodev_vs));
	    ndata_plus += nnodev;
	    As.push_back(&Rv_v_vs);
	    matspec.push_back(SparseMatAux(wsvs,&tmp_nodev,&tmp_nodev_vs));
	    ndata_plus += nnodev;
	}
    }
    if (nrefl>0 && sd && !freeze_refl){
	if (wsd<0) error("TomographicInversion2d::_solve - negative wsd");
	As.push_back(&Rd);
	matspec.push_back(SparseMatAux(wsd,&tmp_nodedr,&tmp_nodedc));
	ndata_plus += nnoded;
    }
    if (dv){
	if (wdv<0) error("TomographicInversion2d::_solve - negative wdv");
	As.push_back(&Tv);
	matspec.push_back(SparseMatAux(wdv,&tmp_nodev,&tmp_nodev));
	ndata_plus += nnodev;
	if (invert_joint_vpvs){
	    As.push_back(&Tv_vs);
	    matspec.push_back(SparseMatAux(wdv,&tmp_nodev,&tmp_nodev_vs));
	    ndata_plus += nnodev;
	}
    }
    if (nrefl>0 && dd && !freeze_refl){
	if (wdd<0) error("TomographicInversion2d::_solve - negative wdd");
	As.push_back(&Td);
	matspec.push_back(SparseMatAux(wdd,&tmp_nodedr,&tmp_nodedc));
	ndata_plus += nnoded;
    }
    SparseRectangular B(As,matspec,nnode_total);

    // construct total data vector
    total_data_vec.resize(ndata_plus);
    int idata=1;
    for (int i=1; i<=ndata; i++){
	if (tmp_data(i)>0){
	    total_data_vec(idata++) = data_vec(i);
	}
    }
    if (gravity){
	for (int i=1; i<=ngravdata; i++){
	    total_data_vec(idata++) = res_grav(i);
	}
    }
    if (sv){
	if (jumping){
	    Array1d<double> Rvm(nnodev), dmv_vec(nnodev);
	    SparseRectangular sparseRv_h(Rv_h,tmp_nodev,tmp_nodev,nnodev);
	    for (int i=1; i<=nnodev; i++) dmv_vec(i) = dmodel_total_sum(i);
	    sparseRv_h.Ax(dmv_vec,Rvm);
	    for (int i=1; i<=nnodev; i++) total_data_vec(idata++) = -wsv*Rvm(i); // Lh

	    SparseRectangular sparseRv_v(Rv_v,tmp_nodev,tmp_nodev,nnodev);
	    sparseRv_v.Ax(dmv_vec,Rvm);
	    for (int i=1; i<=nnodev; i++) total_data_vec(idata++) = -wsv*Rvm(i); // Lv
	    if (invert_joint_vpvs){
		Array1d<double> dmv_vs(nnodev);
		for (int i=1; i<=nnodev; i++) dmv_vs(i) = dmodel_total_sum(i+nnodev);
		SparseRectangular sparseRv_h_vs(Rv_h_vs,tmp_nodev,tmp_nodev,nnodev);
		sparseRv_h_vs.Ax(dmv_vs,Rvm);
		for (int i=1; i<=nnodev; i++) total_data_vec(idata++) = -wsvs*Rvm(i);
		SparseRectangular sparseRv_v_vs(Rv_v_vs,tmp_nodev,tmp_nodev,nnodev);
		sparseRv_v_vs.Ax(dmv_vs,Rvm);
		for (int i=1; i<=nnodev; i++) total_data_vec(idata++) = -wsvs*Rvm(i);
	    }
	}else{
	    for (int i=1; i<=nnodev; i++) total_data_vec(idata++) = 0.0; // Lh
	    for (int i=1; i<=nnodev; i++) total_data_vec(idata++) = 0.0; // Lv
	    if (invert_joint_vpvs){
		for (int i=1; i<=nnodev; i++) total_data_vec(idata++) = 0.0;
		for (int i=1; i<=nnodev; i++) total_data_vec(idata++) = 0.0;
	    }
	}
    }
    if (nrefl>0 && sd){
	if (jumping){
	    Array1d<double> Rdm(nnoded), dmd_vec(nnoded);
	    SparseRectangular sparseRd(Rd,tmp_nodedr,tmp_nodedr);
	    const int nvel = nVelUnknowns();
	    for (int i=1+nvel, j=1; i<=nnoded+nvel; i++) dmd_vec(j++) = dmodel_total_sum(i);
	    sparseRd.Ax(dmd_vec,Rdm);
	    for (int i=1; i<=nnoded; i++) total_data_vec(idata++) = -wsd*Rdm(i); // Ld
	}else{
	    for (int i=1; i<=nnoded; i++) total_data_vec(idata++) = 0.0;
	}
    }
    if (dv){
	if (jumping){
	    Array1d<double> Tvm(nnodev), dmv_vec(nnodev);
	    SparseRectangular sparseTv(Tv,tmp_nodev,tmp_nodev,nnodev);
	    for (int i=1; i<=nnodev; i++) dmv_vec(i) = dmodel_total_sum(i);
	    sparseTv.Ax(dmv_vec,Tvm);
	    for (int i=1; i<=nnodev; i++) total_data_vec(idata++) = -wdv*Tvm(i);
	    if (invert_joint_vpvs){
		Array1d<double> dmv_vs(nnodev);
		for (int i=1; i<=nnodev; i++) dmv_vs(i) = dmodel_total_sum(i+nnodev);
		SparseRectangular sparseTv_vs(Tv_vs,tmp_nodev,tmp_nodev,nnodev);
		sparseTv_vs.Ax(dmv_vs,Tvm);
		for (int i=1; i<=nnodev; i++) total_data_vec(idata++) = -wdv*Tvm(i);
	    }
	}else{
	    for (int i=1; i<=nnodev; i++) total_data_vec(idata++) = 0.0;
	    if (invert_joint_vpvs){
		for (int i=1; i<=nnodev; i++) total_data_vec(idata++) = 0.0;
	    }
	}
    }
    if (nrefl>0 && dd){
	if (jumping){
	    Array1d<double> Tdm(nnoded), dmd_vec(nnoded);
	    SparseRectangular sparseTd(Td,tmp_nodedr,tmp_nodedr);
	    const int nvel = nVelUnknowns();
	    for (int i=1+nvel, j=1; i<=nnoded+nvel; i++) dmd_vec(j++) = dmodel_total_sum(i);
	    sparseTd.Ax(dmd_vec,Tdm);
	    for (int i=1; i<=nnoded; i++) total_data_vec(idata++) = -wdd*Tdm(i); 
	}else{
	    for (int i=1; i<=nnoded; i++) total_data_vec(idata++) = 0.0;
	}
    }

    int lsqr_itermax=inv_lsqr_itermax(itermax_LSQR);
    double chi;
    const double atol = inv_lsqr_atol(LSQR_ATOL);

    // 无阻尼探步 (dv=dd=false) 不列缩放：暗列 D 会被顶满。
    iterativeSolver_LSQR(B,total_data_vec,dmodel_total,atol,lsqr_itermax,chi,
			 dv || dd);
    for (int i=1; i<=dmodel_total.size(); i++){
	if (!std::isfinite(dmodel_total(i))) dmodel_total(i) = 0.0;
    }

    return lsqr_itermax;
}

void TomographicInversion2d::fixed_damping(int& iter, int& n, double& wdv, double& wdd)
{
    wdv = weight_d_v;
    wdd = weight_d_d;
    
    iter += _solve(smooth_velocity,weight_s_v,smooth_depth,weight_s_d,
		   true,weight_d_v,true,weight_d_d);
    n++;

    if (robust){
	if (verbose_level>=0){
	    cerr << "TomographicInversion2d:: check outliers... ";
	}
	int ndata_valid_orig=ndata_valid;
	removeOutliers();
	if (verbose_level>=0){
	    cerr << ndata_valid_orig-ndata_valid << " found\n";
	}
	if (ndata_valid < ndata_valid_orig){
	    if (verbose_level>=0){
		cerr << "TomographicInversion2d:: re-inverting without outliers...\n";
	    }
	    iter += _solve(smooth_velocity,weight_s_v,smooth_depth,weight_s_d,
			   true,weight_d_v,true,weight_d_d);
	    n++;
	}
    }
}

void TomographicInversion2d::auto_damping(int& iter, int& n, double& wdv, double& wdd)
{
    // 联合 Vs-only：Vp 列冻结 + 面下 S 照明截断后大量空列。
    // -TV 的无阻尼探步 (wdv=0) 达不到 ATOL，会顶到 nnode*100。
    const bool skip_undamped = invert_joint_vpvs && dump_iter > joint_vp_only_iters;
    const bool skip_undamped_psx_depth = invert_vs_psx && damp_depth && !freeze_refl;
    if (skip_undamped_psx_depth){
	if (verbose_level>=0)
	    cerr << "\t\tskip undamped probe (vs_psx + depth); one damped solve\n";
	if (printLog)
	    *log_os_p << "# auto_damp: skip undamped probe (vs_psx + depth iter="
		      << dump_iter << ")\n" << flush;
	wdv = damp_velocity ? 50.0 : 0.0;
	wdd = damp_depth ? 20.0 : 0.0;
	if (verbose_level>=0)
	    cerr << "\t\tvs_psx+depth fixed damp wdv=" << wdv
		 << " wdd=" << wdd << "\n";
	if (printLog)
	    *log_os_p << "# auto_damp: vs_psx+depth wdv=" << wdv
		      << " wdd=" << wdd << "\n" << flush;
	iter += _solve(smooth_velocity,weight_s_v,smooth_depth,weight_s_d,
		       damp_velocity,wdv,damp_depth,wdd);
	n++;
	return;
    }
    if (skip_undamped){
	if (verbose_level>=0)
	    cerr << "\t\tskip undamped probe (joint Vs-only); one damped solve\n";
	if (printLog)
	    *log_os_p << "# auto_damp: skip undamped probe (joint Vs-only iter="
		      << dump_iter << ")\n" << flush;
	// 弱照明下 -TV 搜索的每次探步都顶满 LSQR。一次中等阻尼即可。
	wdv = damp_velocity ? 50.0 : 0.0;
	wdd = 0.0;
	if (verbose_level>=0)
	    cerr << "\t\tjoint Vs-only fixed damp wdv=" << wdv << "\n";
	if (printLog)
	    *log_os_p << "# auto_damp: joint Vs-only wdv=" << wdv << "\n" << flush;
	iter += _solve(smooth_velocity,weight_s_v,smooth_depth,weight_s_d,
		       damp_velocity,wdv,false,0.0);
	n++;
	return;
    }

    // check if damping is necessary
    iter += _solve(smooth_velocity,weight_s_v,smooth_depth,weight_s_d,
		   false,0.0,false,0.0);
    n++;

    if (robust){
	if (verbose_level>=0){
	    cerr << "TomographicInversion2d:: check outliers... ";
	}
	int ndata_valid_orig=ndata_valid;
	removeOutliers();
	if (verbose_level>=0){
	    cerr << ndata_valid_orig-ndata_valid << " found\n";
	}
	if (ndata_valid < ndata_valid_orig){
	    if (verbose_level>=0){
		cerr << "TomographicInversion2d:: re-inverting without outliers...\n";
	    }
	    iter += _solve(smooth_velocity,weight_s_v,smooth_depth,weight_s_d,
			   false,0.0,false,0.0);
	    n++;
	}
    }

    double ave_dmv0 = calc_ave_dmv();
    double ave_dmd0 = calc_ave_dmd();
    if (verbose_level>=0){
	cerr << "\t\tave_dm = " << ave_dmv0*100
	     << "%, " << ave_dmd0*100 << "% with no damping\n";
    }
    wdv = wdd = 1.0;
    if (ave_dmv0<=target_dv || !damp_velocity) wdv = 0.0;
    if (ave_dmd0<=target_dd || !damp_depth) wdd = 0.0; 
    if (wdv==0.0 && wdd==0.0) return;

    if (wdd>0.0) wdd = auto_damping_depth(wdv,iter,n);
    if (wdv>0.0) wdv = auto_damping_vel(wdd,iter,n);
}

double TomographicInversion2d::auto_damping_depth(double wdv, int& iter, int& n)
{
    // secant and bisection search
    if (verbose_level>=0) cerr << "\t\tsearching weight_d_depth...\n";
    const double wd_max = 1e7; // absolute bound
    double wd1 = 1.0; // initial guess
    double wd2 = 1e2;
    const double ddm = target_dd*0.3; // target accuracy = 30 %
    const int secant_itermax=10;
    double rts, xl, swap, dx, fl, f;

    iter += _solve(smooth_velocity,weight_s_v,smooth_depth,weight_s_d,
		   damp_velocity,wdv,true,wd1); n++;
    fl = calc_ave_dmd()-target_dd;
    if (verbose_level>=0){
	cerr << "\t\tave_dm = " << (fl+target_dd)*100 << "% at wd1("
	     << wd1 << ")\n";
    }
    iter += _solve(smooth_velocity,weight_s_v,smooth_depth,weight_s_d,
		   damp_velocity,wdv,true,wd2); n++;
    f = calc_ave_dmd()-target_dd;
    if (verbose_level>=0){
	cerr << "\t\tave_dm = " << (f+target_dd)*100 << "% at wd2("
	     << wd2 << ")\n";
    }

    if (abs(fl) < abs(f)){
	rts = wd1;
	xl = wd2;
	swap=fl; fl=f; f=swap;
    }else{
	xl = wd1; rts = wd2;
    }
    for (int j=1; j<=secant_itermax; j++){
	dx = (xl-rts)*f/(f-fl);
	if (rts+dx<=0){ // switch to bisection
	    xl=rts; fl=f;
	    rts *= 0.5;
	}else if(rts+dx>wd_max){
	    xl=rts; fl=f;
	    rts += (wd_max-rts)*0.5;
	}else{
	    xl=rts; fl=f;
	    rts+=dx;
	}
	iter += _solve(smooth_velocity,weight_s_v,smooth_depth,weight_s_d,
		       damp_velocity,wdv,true,rts); n++;
	f = calc_ave_dmd()-target_dd;
	if (verbose_level>=0){
	    cerr << "\t\tave_dm = " << (f+target_dd)*100
		 << "% at w_d of " << rts << '\n';
	}
	if (rts > wd_max) break;
	if (abs(f) < ddm) break;
    }
    return rts;
}

double TomographicInversion2d::auto_damping_vel(double wdd, int& iter, int& n)
{
    // secant and bisection search
    if (verbose_level>=0) cerr << "\t\tsearching weight_d_vel...\n";
    const double wd_max = 1e5; // absolute bound
    double wd1 = 1.0; // initial guess
    double wd2 = 1e2;
    const double ddm = target_dv*0.3; // target accuracy = 30 %
    const int secant_itermax=10;
    double rts, xl, swap, dx, fl, f;

    iter += _solve(smooth_velocity,weight_s_v,smooth_depth,weight_s_d,
		   true,wd1,damp_depth,wdd); n++;
    fl = calc_ave_dmv()-target_dv;
    if (verbose_level>=0){
	cerr << "\t\tave_dm = " << (fl+target_dv)*100 << "% at wd1("
	     << wd1 << ")\n";
    }
    iter += _solve(smooth_velocity,weight_s_v,smooth_depth,weight_s_d,
		   true,wd2,damp_depth,wdd); n++;
    f = calc_ave_dmv()-target_dv;
    if (verbose_level>=0){
	cerr << "\t\tave_dm = " << (f+target_dv)*100 << "% at wd2("
	     << wd2 << ")\n";
    }

    if (abs(fl) < abs(f)){
	rts = wd1;
	xl = wd2;
	swap=fl; fl=f; f=swap;
    }else{
	xl = wd1; rts = wd2;
    }
    for (int j=1; j<=secant_itermax; j++){
	dx = (xl-rts)*f/(f-fl);
	if (rts+dx<=0){ // switch to bisection
	    xl=rts; fl=f;
	    rts *= 0.5;
	}else{
	    xl=rts; fl=f;
	    rts+=dx;
	}
	iter += _solve(smooth_velocity,weight_s_v,smooth_depth,weight_s_d,
		       true,rts,damp_depth,wdd); n++;
	f = calc_ave_dmv()-target_dv;
	if (verbose_level>=0){
	    cerr << "\t\tave_dm = " << (f+target_dv)*100
		 << "% at w_d of " << rts << '\n';
	}
	if (rts > wd_max) break;
	if (abs(f) < ddm) break;
    }
    return rts;
}

