/*
 * syngen.cc - forward traveltime calculation 
 * 
 * Jun Korenaga, MIT/WHOI
 * January 1999
 */

#include <iostream>
#include <fstream>
#include <string>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <vector>
#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif
#include "syngen.h"
#include "betaspline.h"
#include "traveltime.h"
#ifdef _OPENMP
#include <omp.h>
#endif

static inline double fwd_wall_seconds()
{
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return double(clock())/CLOCKS_PER_SEC;
#endif
}

static void fwd_omp_progress_line(int done, int nsrc)
{
    char buf[192];
    int n = std::snprintf(
	buf, sizeof(buf),
	"SyntheticTraveltimeGenerator2d:: ray tracing %d/%d sources (OMP)\n",
	done, nsrc);
    if (n <= 0) return;
    if (n >= (int)sizeof(buf)) n = (int)sizeof(buf) - 1;
#ifdef _WIN32
    _write(2, buf, (unsigned)n);
#else
    ssize_t w = write(STDERR_FILENO, buf, (size_t)n);
    (void)w;
#endif
}

SyntheticTraveltimeGenerator2d::SyntheticTraveltimeGenerator2d(
    SlownessMesh2d& m, const char* ifn,
    int xorder, int zorder, double clen, int nintp, double cg_tol, double br_tol)
    : smesh(m), graph(smesh,xorder,zorder),
      betasp(1,0,nintp), bend(smesh,betasp,cg_tol,br_tol),
      nrefl(0), do_full_refl(false),
      outray(false), use_clock(false), vpvs_kappa(0.0), vpvs_kappa_below(0.0), verbose_level(-1)
{
    if (clen>0.0) graph.refineIfLong(clen);

    start_i.reserve(5); end_i.reserve(5);
    interp.reserve(5);
    bathyp = new Interface2d(smesh);
    reflp = 0;
    convp = 0;
    seafloorp = 0;

    graph_only = false;
    read_file(ifn);
}

void SyntheticTraveltimeGenerator2d::graphOnly()
{
    graph_only = true;
}

void SyntheticTraveltimeGenerator2d::setKappa(double k)
{
    setKappa(k, k);
}

void SyntheticTraveltimeGenerator2d::setKappa(double k_lid, double k_below)
{
    if (k_lid<=0.0 || k_below<=0.0)
	error("SyntheticTraveltimeGenerator2d::setKappa - kappa must be positive.");
    vpvs_kappa = k_lid;
    vpvs_kappa_below = k_below;
    if (!smesh.hasDualVs()) smesh.setMixKappa(k_lid, k_below);
}

void SyntheticTraveltimeGenerator2d::conduct()
{
    Array1d<double> modelv, tmp_modelv, modelu, tmp_modelu;
    bool has_code5=false, has_psx=false;
    for (int is=1; is<=src.size(); is++){
	for (int ir=1; ir<=raytype(is).size(); ir++){
	    int ic = raytype(is)(ir);
	    if (ic==5) has_code5=true;
	    if (ic==6 || ic==7 || ic==8 || ic==9 || ic==10 || ic==11
		|| ic==12 || ic==13 || ic==14 || ic==15) has_psx=true;
	}
    }
    // Single-field 6/7/8 with -k: -M is Vp. Lid Vs=Vp/k_lid, below Vs=Vp/k_below.
    if (has_psx && convp && !smesh.hasDualVs() && vpvs_kappa>0.0)
	smesh.enableDualVs(vpvs_kappa,
			   vpvs_kappa_below>0.0 ? vpvs_kappa_below : vpvs_kappa,
			   convp);

    ofstream *rout_p = 0;
    if (outray) rout_p = new ofstream(rayfn);

    const bool mask_below_refl = (do_full_refl || has_code5) && nrefl>0 && reflp!=0;
    if (mask_below_refl){
	modelv.resize(smesh.numNodes());
	tmp_modelv.resize(smesh.numNodes());
	// Original -A masks the P field. Dual: that is pgrid_vp, not Vs.
	smesh.getVp(modelv);
	int nx(smesh.Nx()), nz(smesh.Nz());
	double pwater = 1.0/1.5;
	tmp_modelv = modelv;
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
		    // be the same sa the velocity just above. If I use pwater for this node,
		    // there would be a unwanted low velocity gradient surrounding the reflector.
		    tmp_modelv(smesh.nodeIndex(i,k)) = tmp_modelv(smesh.nodeIndex(i,k-1));
		}else if (k>irefl(i)){
		    tmp_modelv(smesh.nodeIndex(i,k)) = pwater;
		}
	    }
	}
	if (smesh.hasDualVs()){
	    modelu.resize(smesh.numNodes());
	    tmp_modelu.resize(smesh.numNodes());
	    smesh.get(modelu);
	    tmp_modelu = modelu;
	    for (int i=1; i<=nx; i++){
		for (int k=2; k<=nz; k++){
		    if (k==irefl(i))
			tmp_modelu(smesh.nodeIndex(i,k)) = tmp_modelu(smesh.nodeIndex(i,k-1));
		    else if (k>irefl(i))
			tmp_modelu(smesh.nodeIndex(i,k)) = pwater;
		}
	    }
	}
    }
    
    graph_time=bend_time=0.0;

    bool request_src_parallel = false;
    bool compiled_with_openmp = false;
#ifdef _OPENMP
    compiled_with_openmp = true;
#endif
    {
	const char* omp_env = getenv("TOMO2D_FWD_OMP");
	request_src_parallel = (omp_env && atoi(omp_env)!=0);
    }
    const bool parallel_src = request_src_parallel && compiled_with_openmp && !mask_below_refl;
    if (request_src_parallel && !parallel_src){
	cerr << "SyntheticTraveltimeGenerator2d:: OMP requested but source-wise parallel disabled. "
	     << "compiled_with_openmp=" << compiled_with_openmp
	     << " do_full_refl=" << do_full_refl
	     << "\n";
	cerr.flush();
    }
    if (parallel_src){
	int nthr = 1;
#ifdef _OPENMP
	nthr = omp_get_max_threads();
#endif
	cerr << "SyntheticTraveltimeGenerator2d:: parallel ray tracing enabled (source-wise, OMP)"
	     << " threads=" << nthr
	     << " nsrc=" << src.size()
	     << "\n";
	cerr.flush();
    }

    std::vector<std::string> ray_blocks;
    if (outray) ray_blocks.resize(src.size());
    int nsrc_done = 0;

#pragma omp parallel if(parallel_src)
    {
	GraphSolver2d graph_local(smesh,graph.xOrder(),graph.zOrder());
	if (graph.critLength()>0) graph_local.refineIfLong(graph.critLength());
	if (graph.reflDownward()) graph_local.do_refl_downward();
	BendingSolver2d bend_local(smesh,betasp,bend.tolerance(),bend.brentTolerance());
	Array1d<Point2d> path_local;
	const int max_np = int(2*sqrt(float(smesh.numNodes())));
	path_local.reserve(max_np);
	Array1d<int> start_i_local, end_i_local;
	Array1d<const Interface2d*> interp_local;
	    start_i_local.reserve(3); end_i_local.reserve(3); interp_local.reserve(3);
	    const Interface2d* sf_local = seafloorp ? seafloorp : reflp;
	    double graph_time_local = 0.0;
	double bend_time_local = 0.0;

#pragma omp for schedule(dynamic,1)
	for (int isrc=1; isrc<=src.size(); isrc++){
	    std::ostringstream ray_os;
	    if (!parallel_src && verbose_level>=0){
		cerr << "isrc=" << isrc << " nrec=" << rcv(isrc).size()
		     << " ";
	    }

	    // limit range first
	    double xmin=smesh.xmax();
	    double xmax=smesh.xmin();
	    for (int ircv=1; ircv<=rcv(isrc).size(); ircv++){
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

	    double start_t = fwd_wall_seconds();
	    bool has_crust_phase=false, has_water_phase=false, has_water_mult=false;
	    bool has_recv_peg_refr=false, has_recv_peg_refl=false, has_psp_phase=false;
	    int conv_mode_solved = 0;
	    for (int jrt=1; jrt<=rcv(isrc).size(); jrt++){
		int c = raytype(isrc)(jrt);
		if (c==0 || c==1) has_crust_phase=true;
		if (c==2 || c==3 || c==4 || c==5 || c==9 || c==14 || c==15) has_water_phase=true;
		if (c==3 || c==4 || c==5 || c==9 || c==14 || c==15) has_water_mult=true;
		if (c==4) has_recv_peg_refr=true;
		if (c==5) has_recv_peg_refl=true;
		if (c==6) has_psp_phase=true;
	    }
	    if (has_crust_phase)
		graph_local.solve(src(isrc));
	    if (has_psp_phase){
		if (convp==0)
		    error("SyntheticTraveltimeGenerator2d:: raytype 6 requires conversion interface (-X)");
		graph_local.solve_psp(src(isrc), *convp,
				      vpvs_kappa>0.0 ? vpvs_kappa : 1.0);
	    }
	    bool is_refl_solved=false;
	    double end_t = fwd_wall_seconds();
	    graph_time_local += end_t-start_t;

	    start_t = fwd_wall_seconds();
	    if (has_water_phase){
		if (sf_local==0)
		    error("SyntheticTraveltimeGenerator2d:: raytype 2/3/4/5/9/14/15 requires seafloor (-B) or -F");
		double tw0=fwd_wall_seconds();
		graph_local.solve_water(src(isrc), *sf_local);
		if (has_water_mult)
		    graph_local.solve_water_mult(*sf_local, *bathyp);
		if (has_recv_peg_refr)
		    graph_local.solve_recv_peg_refr(*sf_local, *bathyp);
		if (has_recv_peg_refl){
		    if (seafloorp==0 || nrefl==0)
			error("SyntheticTraveltimeGenerator2d:: raytype 5 requires -B (seafloor) and -F (Moho)");
		    graph_local.solve_recv_peg_refl(*seafloorp, *bathyp, *reflp);
		}
		graph_time_local += fwd_wall_seconds()-tw0;
	    }
	    double graph_refl_time=0.0;
	    for (int ircv=1; ircv<=rcv(isrc).size(); ircv++){
		Point2d r = rcv(isrc)(ircv);
		double orig_t, final_t;
		int icode = raytype(isrc)(ircv);
		int ray_iu1=0, ray_id0=0, ray_id1=0;
		if (icode == 0){ // refraction
		    if (smesh.inWater(r)){
			if (!parallel_src && verbose_level>=0) cerr << "*";
			int i0, i1;
			graph_local.pickPathThruWater(r,path_local,i0,i1);
			start_i_local.resize(1); end_i_local.resize(1); interp_local.resize(1);
			start_i_local(1) = i0; end_i_local(1) = i1; interp_local(1) = bathyp;
			if (graph_only){
			    final_t = calcTravelTime(smesh,path_local,betasp.numIntp());
			}else{
			    int iterbend=bend_local.refine(path_local,orig_t,final_t,
							   start_i_local, end_i_local, interp_local);
			    if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
			}
		    }else{
			if (!parallel_src && verbose_level>=0) cerr << ".";
			graph_local.pickPath(r,path_local);
			if (graph_only){
			    final_t = calcTravelTime(smesh,path_local,betasp.numIntp());
			}else{
			    const int nfac=1;
			    int iterbend=bend_local.refine(path_local,orig_t,final_t,nfac);
			    if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
			}
		    }
		}else if (icode == 1){ // reflection
		    if (nrefl==0){
			error("SyntheticTraveltimeGenerator2d:: reflector not specified.");
		    }
		    if (do_full_refl){
			// temporarily replace sub-reflector velocity field
			// with water velocity
			smesh.setVp(tmp_modelv);
		    }
		    if (!is_refl_solved){
			double start_t_refl=fwd_wall_seconds();
			graph_local.solve_refl(src(isrc), *reflp);
			double end_t_refl=fwd_wall_seconds();
			graph_refl_time = end_t_refl-start_t_refl;
			graph_time_local += graph_refl_time;
			
			is_refl_solved = true;
		    }
		    if (smesh.inWater(r)){
			if (!parallel_src && verbose_level>=0) cerr << "#";
			int i0, i1, ir0, ir1;
			graph_local.pickReflPathThruWater(r,path_local,i0,i1,ir0,ir1);
			start_i_local.resize(2); end_i_local.resize(2); interp_local.resize(2);
			start_i_local(1) = i0; end_i_local(1) = i1; interp_local(1) = bathyp;
			start_i_local(2) = ir0; end_i_local(2) = ir1; interp_local(2) = reflp;
			if (graph_only){
			    final_t = calcTravelTime(smesh,path_local,betasp.numIntp());
			}else{
			    int iterbend=bend_local.refine(path_local,orig_t,final_t,
							   start_i_local, end_i_local, interp_local);
			    if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
			}
		    }else{
			if (!parallel_src && verbose_level>=0) cerr << "+";
			int ir0, ir1;
			graph_local.pickReflPath(r,path_local,ir0,ir1);
			start_i_local.resize(1); end_i_local.resize(1); interp_local.resize(1);
			start_i_local(1) = ir0; end_i_local(1) = ir1; interp_local(1) = reflp;
			if (graph_only){
			    final_t = calcTravelTime(smesh,path_local,betasp.numIntp());
			}else{
			    int iterbend=bend_local.refine(path_local,orig_t,final_t,
							   start_i_local, end_i_local, interp_local);
			    if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
			}
		    }
		    if (do_full_refl){
			// revert to the original smesh
			smesh.setVp(modelv);
		    }
		}else if (icode == 2 || icode == 3){
		    if (sf_local==0)
			error("SyntheticTraveltimeGenerator2d:: raytype 2/3 requires seafloor (-B) or -F");
		    int i0, i1, ir0, ir1;
		    if (icode==2){
			if (!parallel_src && verbose_level>=0) cerr << "w";
			graph_local.pickWaterDirectPath(r,path_local,ir0,ir1);
		    }else{
			if (!parallel_src && verbose_level>=0) cerr << "m";
			graph_local.pickWaterMultPath(r,path_local,i0,i1,ir0,ir1);
			start_i_local.resize(2); end_i_local.resize(2); interp_local.resize(2);
			start_i_local(1) = ir0; end_i_local(1) = ir1; interp_local(1) = sf_local;
			start_i_local(2) = i0; end_i_local(2) = i1; interp_local(2) = bathyp;
		    }
		    clampPathToWaterCol(path_local, *bathyp, *sf_local);
		    bend_local.setWaterColClip(bathyp, sf_local);
		    if (graph_only){
			final_t = calcTravelTime(smesh,path_local,betasp.numIntp());
		    }else if (icode==2){
			const int nfac=1;
			int iterbend=bend_local.refine(path_local,orig_t,final_t,nfac);
			if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
			clampPathToWaterCol(path_local, *bathyp, *sf_local);
			Array1d<const Point2d*> wpp;
			Array1d<Point2d> wQ(betasp.numIntp());
			makeBSpoints(path_local, wpp);
			final_t = calcTravelTime(smesh, path_local, betasp, wpp, wQ, bathyp, sf_local);
		    }else{
			int iterbend=bend_local.refine(path_local,orig_t,final_t,
						       start_i_local, end_i_local, interp_local);
			if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
			clampPathToWaterCol(path_local, *bathyp, *sf_local);
			Array1d<const Point2d*> wpp;
			Array1d<Point2d> wQ(betasp.numIntp());
			makeBSpoints(path_local, wpp);
			final_t = calcTravelTime(smesh, path_local, betasp, wpp, wQ, bathyp, sf_local);
		    }
		    bend_local.setWaterColClip(0, 0);
		}else if (icode == 4){
		    if (sf_local==0)
			error("SyntheticTraveltimeGenerator2d:: raytype 4 requires seafloor (-B) or -F");
		    if (!parallel_src && verbose_level>=0) cerr << "P";
		    int is0, is1, ib0, ib1, ie0, ie1;
		    graph_local.pickRecvPegRefrPath(r,path_local,is0,is1,ib0,ib1,ie0,ie1);
		    start_i_local.resize(3); end_i_local.resize(3); interp_local.resize(3);
		    start_i_local(1) = is0; end_i_local(1) = is1; interp_local(1) = bathyp;
		    start_i_local(2) = ib0; end_i_local(2) = ib1; interp_local(2) = sf_local;
		    start_i_local(3) = ie0; end_i_local(3) = ie1; interp_local(3) = sf_local;
		    if (graph_only){
			final_t = calcTravelTime(smesh,path_local,betasp.numIntp());
		    }else{
			int iterbend=bend_local.refine(path_local,orig_t,final_t,
						       start_i_local, end_i_local, interp_local);
			if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    }
		}else if (icode == 5){
		    if (seafloorp==0 || nrefl==0)
			error("SyntheticTraveltimeGenerator2d:: raytype 5 requires -B (seafloor) and -F (Moho)");
		    if (mask_below_refl) smesh.setVp(tmp_modelv);
		    if (!parallel_src && verbose_level>=0) cerr << "Q";
		    int is0, is1, ib0, ib1, ir0, ir1;
		    graph_local.pickRecvPegReflPath(r,path_local,is0,is1,ib0,ib1,ir0,ir1);
		    start_i_local.resize(3); end_i_local.resize(3); interp_local.resize(3);
		    start_i_local(1) = is0; end_i_local(1) = is1; interp_local(1) = bathyp;
		    start_i_local(2) = ib0; end_i_local(2) = ib1; interp_local(2) = seafloorp;
		    start_i_local(3) = ir0; end_i_local(3) = ir1; interp_local(3) = reflp;
		    if (graph_only){
			final_t = calcTravelTime(smesh,path_local,betasp.numIntp());
		    }else{
			int iterbend=bend_local.refine(path_local,orig_t,final_t,
						       start_i_local, end_i_local, interp_local);
			if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    }
		    if (mask_below_refl) smesh.setVp(modelv);
		}else if (icode == 6 || icode == 7 || icode == 8){
		    if (convp==0)
			error("SyntheticTraveltimeGenerator2d:: raytype 6/7/8 requires conversion interface (-X)");
		    if (icode!=6){
			if (sf_local==0)
			    error("SyntheticTraveltimeGenerator2d:: raytype 7/8 requires seafloor (-B)");
			if (vpvs_kappa<=0.0 && !smesh.hasDualVs())
			    error("SyntheticTraveltimeGenerator2d:: raytype 7/8 single-field requires -k<k> or -k<k_lid>/<k_below> (Vp→Vs)");
		    }
		    if (conv_mode_solved != icode){
			if (icode==6)
			    graph_local.solve_psp(src(isrc), *convp,
						  vpvs_kappa>0.0 ? vpvs_kappa : 1.0);
			else if (icode==7)
			    graph_local.solve_pps(src(isrc), *convp, *sf_local, vpvs_kappa);
			else
			    graph_local.solve_pss(src(isrc), *convp, *sf_local, vpvs_kappa);
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
		    const bool dual_psp = smesh.hasDualVs() || vpvs_kappa>0.0;
		    const bool one_hinge = (id0 <= iu1);
		    // Type 6 mixed or dual: always pin B and C so S-leg clamp has
		    // a defined B–C interval (S stays below conv; lid is P).
		    const bool whole_psx_pins = (icode==6) || (icode==8);
		    if (smesh.inWater(r)){
			if (one_hinge && !whole_psx_pins){
			    start_i_local.resize(2); end_i_local.resize(2); interp_local.resize(2);
			    start_i_local(1) = i0; end_i_local(1) = i1; interp_local(1) = bathyp;
			    start_i_local(2) = iu0; end_i_local(2) = iu1; interp_local(2) = convp;
			}else{
			    start_i_local.resize(3); end_i_local.resize(3); interp_local.resize(3);
			    start_i_local(1) = i0; end_i_local(1) = i1; interp_local(1) = bathyp;
			    start_i_local(2) = id0; end_i_local(2) = id1; interp_local(2) = convp;
			    start_i_local(3) = iu0; end_i_local(3) = iu1; interp_local(3) = convp;
			}
		    }else if (one_hinge && !whole_psx_pins){
			start_i_local.resize(1); end_i_local.resize(1); interp_local.resize(1);
			start_i_local(1) = iu0; end_i_local(1) = iu1; interp_local(1) = convp;
		    }else{
			start_i_local.resize(2); end_i_local.resize(2); interp_local.resize(2);
			start_i_local(1) = id0; end_i_local(1) = id1; interp_local(1) = convp;
			start_i_local(2) = iu0; end_i_local(2) = iu1; interp_local(2) = convp;
		    }
		    if (icode==6){
			const double psp_k = vpvs_kappa>0.0 ? vpvs_kappa : 1.0;
			if (graph_only){
			    final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
						       iu1, id0, id1, psp_k, true, false, convp);
			}else if (dual_psp){
			    // Same refine + pins as converse; mixed slowness via psx_at.
			    bend_local.setPsxBend(iu1, id0, id1, psp_k, true, convp, false);
			    int iterbend=bend_local.refine(path_local,orig_t,final_t,
							   start_i_local, end_i_local, interp_local);
			    if (iterbend<0)
				final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
							   iu1, id0, id1, psp_k, true, false, convp);
			    bend_local.clearPsxBend();
			    if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
			}else{
			    // Mixed type 6: kappa=1, S-leg stays below conv.
			    bend_local.setPsxBend(iu1, id0, id1, psp_k, true, convp, false);
			    int iterbend=bend_local.refine(path_local,orig_t,final_t,
							   start_i_local, end_i_local, interp_local);
			    if (iterbend<0)
				final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
							   iu1, id0, id1, psp_k, true, false, convp);
			    bend_local.clearPsxBend();
			    if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
			}
		    }else if (icode==7 || icode==8){
			const bool below_s = (icode==8);
			if (graph_only){
			    final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
						       iu1, id0, id1, vpvs_kappa, below_s, true, convp);
			}else{
			    // Same whole-path refine as dual PSP; 7 below P, 8 below S.
			    bend_local.setPsxBend(iu1, id0, id1, vpvs_kappa, below_s, convp, true);
			    int iterbend=bend_local.refine(path_local,orig_t,final_t,
							   start_i_local, end_i_local, interp_local);
			    if (iterbend<0)
				final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
							   iu1, id0, id1, vpvs_kappa, below_s, true, convp);
			    bend_local.clearPsxBend();
			    if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
			}
		    }else{
			error("SyntheticTraveltimeGenerator2d:: illegal raycode detected.");
		    }
		    }else if (icode == 9 || icode == 14 || icode == 15){
		    if (convp==0)
			error("SyntheticTraveltimeGenerator2d:: raytype 9/14/15 requires conversion interface (-X)");
		    if (sf_local==0)
			error("SyntheticTraveltimeGenerator2d:: raytype 9/14/15 requires seafloor (-B)");
		    if (icode!=9 && vpvs_kappa<=0.0 && !smesh.hasDualVs())
			error("SyntheticTraveltimeGenerator2d:: raytype 14/15 single-field requires -k (Vp→Vs)");
		    if (conv_mode_solved != icode){
			const double pk = vpvs_kappa>0.0 ? vpvs_kappa : 1.0;
			if (icode==9)
			    graph_local.solve_psp_peg(src(isrc), *convp, *sf_local, pk, false);
			else if (icode==14)
			    graph_local.solve_pps_peg(src(isrc), *convp, *sf_local, pk);
			else
			    graph_local.solve_pss_peg(src(isrc), *convp, *sf_local, pk);
			conv_mode_solved = icode;
		    }
		    if (!parallel_src && verbose_level>=0){
			if (icode==9) cerr << "J";
			else if (icode==14) cerr << "N";
			else cerr << "O";
		    }
		    int id0, id1, ib0, ib1, is0, is1, iu0, iu1;
		    int i0=0, i1=0;
		    const bool shot_water = smesh.inWater(r);
		    if (shot_water)
			graph_local.pickPspPegPathThruWater(r,path_local,i0,i1,id0,id1,ib0,ib1,is0,is1,iu0,iu1);
		    else
			graph_local.pickPspPegPath(r,path_local,id0,id1,ib0,ib1,is0,is1,iu0,iu1);
		    ray_iu1 = iu1; ray_id0 = id0; ray_id1 = id1;
		    const double pk = vpvs_kappa>0.0 ? vpvs_kappa : 1.0;
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
		    start_i_local(ip) = ib0; end_i_local(ip) = ib1; interp_local(ip) = sf_local; ip++;
		    start_i_local(ip) = is0; end_i_local(ip) = is1; interp_local(ip) = bathyp;
		    if (graph_only){
			final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
						   iu1, id0, id1, pk, below_is_s, lid_is_s, convp);
		    }else{
			bend_local.setPsxBend(iu1, id0, id1, pk, below_is_s, convp, lid_is_s);
			int iterbend=bend_local.refine(path_local,orig_t,final_t,
						       start_i_local, end_i_local, interp_local);
			if (iterbend<0)
			    final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
						       iu1, id0, id1, pk, below_is_s, lid_is_s, convp);
			bend_local.clearPsxBend();
			if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    }
		}else if (icode == 10 || icode == 11){
		    if (convp==0)
			error("SyntheticTraveltimeGenerator2d:: raytype 10/11 requires conversion interface (-X)");
		    if (sf_local==0)
			error("SyntheticTraveltimeGenerator2d:: raytype 10/11 requires seafloor (-B)");
		    if (vpvs_kappa<=0.0 && !smesh.hasDualVs())
			error("SyntheticTraveltimeGenerator2d:: raytype 10/11 single-field requires -k (Vp→Vs)");
		    if (conv_mode_solved != icode){
			const double pk = vpvs_kappa>0.0 ? vpvs_kappa : 1.0;
			if (icode==10)
			    graph_local.solve_pps_ss(src(isrc), *convp, *sf_local, pk);
			else
			    graph_local.solve_pss_ss(src(isrc), *convp, *sf_local, pk);
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
		    const double pk = vpvs_kappa>0.0 ? vpvs_kappa : 1.0;
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
		    start_i_local(ip) = ib0; end_i_local(ip) = ib1; interp_local(ip) = sf_local; ip++;
		    start_i_local(ip) = ic0; end_i_local(ip) = ic1; interp_local(ip) = convp;
		    if (graph_only){
			final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
						   iu1, id0, id1, pk, below_is_s, lid_is_s, convp);
		    }else{
			bend_local.setPsxBend(iu1, id0, id1, pk, below_is_s, convp, lid_is_s);
			int iterbend=bend_local.refine(path_local,orig_t,final_t,
						       start_i_local, end_i_local, interp_local);
			if (iterbend<0)
			    final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
						       iu1, id0, id1, pk, below_is_s, lid_is_s, convp);
			bend_local.clearPsxBend();
			if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    }
		}else if (icode == 12 || icode == 13){
		    if (convp==0)
			error("SyntheticTraveltimeGenerator2d:: raytype 12/13 requires conversion interface (-X)");
		    if (nrefl==0 || reflp==0)
			error("SyntheticTraveltimeGenerator2d:: raytype 12/13 requires Moho (-F)");
		    if (icode==13){
			if (sf_local==0)
			    error("SyntheticTraveltimeGenerator2d:: raytype 13 requires seafloor (-B)");
			if (vpvs_kappa<=0.0 && !smesh.hasDualVs())
			    error("SyntheticTraveltimeGenerator2d:: raytype 13 single-field requires -k (Vp→Vs)");
		    }
		    if (do_full_refl){
			smesh.setVp(tmp_modelv);
			if (smesh.hasDualVs()) smesh.set(tmp_modelu);
		    }
		    if (conv_mode_solved != icode){
			const double pk = vpvs_kappa>0.0 ? vpvs_kappa : 1.0;
			if (icode==12)
			    graph_local.solve_psp_moho(src(isrc), *convp, *reflp, pk);
			else
			    graph_local.solve_pss_moho(src(isrc), *convp, *reflp, *sf_local, pk);
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
		    const double pk = vpvs_kappa>0.0 ? vpvs_kappa : 1.0;
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
		    if (graph_only){
			final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
						   iu1, id0, id1, pk, true, lid_is_s, convp);
		    }else{
			bend_local.setPsxBend(iu1, id0, id1, pk, true, convp, lid_is_s);
			bend_local.setPsxMoho(reflp);
			int iterbend=bend_local.refine(path_local,orig_t,final_t,
						       start_i_local, end_i_local, interp_local);
			if (iterbend<0)
			    final_t = calcPsxGraphTime(smesh, path_local, betasp.numIntp(),
						       iu1, id0, id1, pk, true, lid_is_s, convp);
			bend_local.clearPsxBend();
			if (!parallel_src && verbose_level>=1) cerr << "(" << iterbend << ")";
		    }
		    if (do_full_refl){
			smesh.setVp(modelv);
			if (smesh.hasDualVs()) smesh.set(modelu);
		    }
		}else{
		    error("SyntheticTraveltimeGenerator2d:: illegal raycode detected.");
		}
		if (outray){
		    ray_os << ">\n";
		    if (graph_only){
			for (int i=1; i<=path_local.size(); i++){
			    ray_os << path_local(i).x() << " " << path_local(i).y() << '\n';
			}
		    }else if (icode==2 || icode==3){
			printCurve(ray_os,path_local,betasp,*bathyp,*sf_local);
		    }else if (ray_iu1>0){
			printCurve(ray_os,path_local,betasp,ray_iu1,ray_id0,ray_id1);
		    }else{
			printCurve(ray_os,path_local,betasp);
		    }
		}
		syn_ttime(isrc)(ircv) = final_t;
	    }
	    if (outray){
		ray_blocks[size_t(isrc-1)] = ray_os.str();
	    }
	    if (!parallel_src && verbose_level>=0) cerr << '\n';
	    end_t = fwd_wall_seconds();
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
		    fwd_omp_progress_line(done, nsrc_all);
		}
	    }
	}
#pragma omp atomic
	graph_time += graph_time_local;
#pragma omp atomic
	bend_time += bend_time_local;
    }

    if (outray){
	for (size_t isrc=0; isrc<ray_blocks.size(); isrc++){
	    *rout_p << ray_blocks[isrc];
	}
    }
    if (use_clock){
	ofstream os(timefn);
	os << "graph_time " << graph_time << " sec " 
	   << "bend_time " << bend_time << " sec \n";
    }
    if (outray){
	rout_p->flush(); // make sure all rays are printed out.
	delete rout_p;
    }
}

void SyntheticTraveltimeGenerator2d::outputRay(const char* fn)
{ outray=true; rayfn=fn; }

void SyntheticTraveltimeGenerator2d::useClock(const char* fn)
{ use_clock=true; timefn=fn; }

void SyntheticTraveltimeGenerator2d::setVerbose(int i){ verbose_level = i; }

void SyntheticTraveltimeGenerator2d::read_file(const char* ifn)
{
    ifstream in(ifn);
    if (!in){
	cerr << "SyntheticTraveltimeGenerator2d::cannot open " << ifn << "\n";
	exit(1);
    }

    int iline=0;

    string first;
    in >> first; iline++;
    if (!isdigit(*first.c_str()))
	error("SyntheticTraveltimeGenerator2d::first line should be nsrc");
    int nsrc = atoi(first.c_str());
    if (nsrc<=0) error("SyntheticTraveltimeGenerator2d::invalid nsrc");
    src.resize(nsrc); rcv.resize(nsrc);
    raytype.resize(nsrc);
    syn_ttime.resize(nsrc); obs_ttime.resize(nsrc); obs_dt.resize(nsrc);

    int isrc=0;
    while(in){
	char flag;
	double x, y;
	int nrcv;

	in >> flag >> x >> y >> nrcv; iline++;
	if (flag!='s'){
	    cerr << "SyntheticTraveltimeGenerator2d::bad input (s) at l."
		 << iline << '\n';
	    exit(1);
	}
	isrc++;
	src(isrc).set(x,y);

	rcv(isrc).resize(nrcv); raytype(isrc).resize(nrcv);
	syn_ttime(isrc).resize(nrcv);
	obs_ttime(isrc).resize(nrcv); obs_dt(isrc).resize(nrcv);
	for (int ircv=1; ircv<=nrcv; ircv++){
	    int n;
	    double ttime_val, dt_val;
	    in >> flag >> x >> y >> n >> ttime_val >> dt_val; iline++;
	    if (flag!='r'){
		cerr << "SyntheticTraveltimeGenerator2d::bad input (r) at l."
		     << iline << '\n';
		exit(1);
	    }
	    rcv(isrc)(ircv).set(x,y);
	    raytype(isrc)(ircv) = n;
	    obs_ttime(isrc)(ircv) = ttime_val;
	    obs_dt(isrc)(ircv) = dt_val;
	}
	if (isrc==nsrc) break;
    }
    if (isrc != nsrc) error("SyntheticTraveltimeGenerator2d::mismatch in nsrc");
}

void
SyntheticTraveltimeGenerator2d::readRefl(const char* fn)
{
    reflp = new Interface2d(fn);
    nrefl = 1;
}

void
SyntheticTraveltimeGenerator2d::readSeafloor(const char* fn)
{
    seafloorp = new Interface2d(fn);
}

void
SyntheticTraveltimeGenerator2d::readConv(const char* fn)
{
    convp = new Interface2d(fn);
}

void
SyntheticTraveltimeGenerator2d::doFullRefl()
{
    do_full_refl = true;
    graph.do_refl_downward();
}
    
void
SyntheticTraveltimeGenerator2d::printSource(ostream& os) const
{
    for (int isrc=1; isrc<=src.size(); isrc++){
	os << src(isrc).x() << " "
	   << src(isrc).y() << '\n';
    }
}

void
SyntheticTraveltimeGenerator2d::printSynTime(ostream& os, double vred) const
{
    for (int isrc=1; isrc<=src.size(); isrc++){
	os << ">\n";
	for (int ircv=1; ircv<=rcv(isrc).size(); ircv++){
	    double redtime = syn_ttime(isrc)(ircv);
	    if (vred!=0){
		redtime -= abs(rcv(isrc)(ircv).x()-src(isrc).x())/vred;
	    }
	    if (ircv>1 &&
		raytype(isrc)(ircv)!=raytype(isrc)(ircv-1)) os << ">\n";
	    os << rcv(isrc)(ircv).x() << " " << redtime << '\n';
	}
    }
}

void
SyntheticTraveltimeGenerator2d::printObsTime(ostream& os, double vred) const
{
    for (int isrc=1; isrc<=src.size(); isrc++){
	os << ">\n";
	for (int ircv=1; ircv<=rcv(isrc).size(); ircv++){
	    double redtime = obs_ttime(isrc)(ircv);
	    if (vred!=0){
		redtime -= abs(rcv(isrc)(ircv).x()-src(isrc).x())/vred;
	    }
	    if (ircv>1 &&
		raytype(isrc)(ircv)!=raytype(isrc)(ircv-1)) os << ">\n";
	    os << rcv(isrc)(ircv).x() << " " << redtime << " "
	       << obs_dt(isrc)(ircv) << '\n';
	}
    }
}

void SyntheticTraveltimeGenerator2d::printDiff(ostream& os, double& misfit, double& chisq) const
{
    misfit = 0.0;
    chisq = 0.0;
    int ndata=0;
    for (int isrc=1; isrc<=src.size(); isrc++){
	for (int ircv=1; ircv<=rcv(isrc).size(); ircv++){
	    double tdiff, tdiff2;
	    tdiff = obs_ttime(isrc)(ircv)-syn_ttime(isrc)(ircv);
	    tdiff2 = tdiff*tdiff;
	    misfit += tdiff2;
	    if (obs_dt(isrc)(ircv)==0.0){
		cerr << "SyntheticTraveltimeGenerator2d::printDiff - zero error found at "
		     << "isrc=" << isrc << ", ircv=" << ircv << '\n';
		exit(1);
	    }
	    double obsdt2 = obs_dt(isrc)(ircv)*obs_dt(isrc)(ircv);
	    chisq += tdiff2/obsdt2;
	    os << tdiff << " " << tdiff/obs_dt(isrc)(ircv) << '\n';
	    ndata++;
	}
    }
    misfit = sqrt(misfit/ndata);
    chisq = chisq/ndata;
    os << "# t_misfit " << misfit << '\n';
    os << "# chisq " << chisq << '\n';
}


ostream&
operator<<(ostream& out, const SyntheticTraveltimeGenerator2d& syn)
{
    const double syn_dt = 0.01; // assume 10 ms error
    
    out << syn.src.size() << '\n'; 
    for (int isrc=1; isrc<=syn.src.size(); isrc++){
	out << 's' << " "
	    << syn.src(isrc).x() << " "
	    << syn.src(isrc).y() << " "
	    << syn.rcv(isrc).size() << '\n';
	for (int ircv=1; ircv<=syn.rcv(isrc).size(); ircv++){
	    out << 'r' << " "
		<< syn.rcv(isrc)(ircv).x() << " "
		<< syn.rcv(isrc)(ircv).y() << " "
		<< syn.raytype(isrc)(ircv) << " "
		<< syn.syn_ttime(isrc)(ircv) << " "
		<< syn_dt << '\n';
	}
    }
    return out;
}
