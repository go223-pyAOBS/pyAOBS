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
#include <sstream>
#include <vector>
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

SyntheticTraveltimeGenerator2d::SyntheticTraveltimeGenerator2d(
    SlownessMesh2d& m, const char* ifn,
    int xorder, int zorder, double clen, int nintp, double cg_tol, double br_tol)
    : smesh(m), graph(smesh,xorder,zorder),
      betasp(1,0,nintp), bend(smesh,betasp,cg_tol,br_tol),
      nrefl(0), do_full_refl(false),
      outray(false), use_clock(false), verbose_level(-1)
{
    if (clen>0.0) graph.refineIfLong(clen);

    start_i.reserve(2); end_i.reserve(2);
    interp.reserve(2);
    bathyp = new Interface2d(smesh);

    graph_only = false;
    read_file(ifn);
}

void SyntheticTraveltimeGenerator2d::graphOnly()
{
    graph_only = true;
}

void SyntheticTraveltimeGenerator2d::conduct()
{
    ofstream *rout_p = 0;
    if (outray) rout_p = new ofstream(rayfn);
	
    Array1d<double> modelv, tmp_modelv;
    if (do_full_refl){
	modelv.resize(smesh.numNodes());
	tmp_modelv.resize(smesh.numNodes());
	smesh.get(modelv);
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
    const bool parallel_src = request_src_parallel && compiled_with_openmp && !do_full_refl;
    if (request_src_parallel && !parallel_src){
	cerr << "SyntheticTraveltimeGenerator2d:: OMP requested but source-wise parallel disabled. "
	     << "compiled_with_openmp=" << compiled_with_openmp
	     << " do_full_refl=" << do_full_refl
	     << "\n";
    }
    if (parallel_src){
	cerr << "SyntheticTraveltimeGenerator2d:: parallel ray tracing enabled (source-wise, OMP)\n";
    }

    std::vector<std::string> ray_blocks;
    if (outray) ray_blocks.resize(src.size());

#pragma omp parallel if(parallel_src)
    {
	GraphSolver2d graph_local(smesh,graph.xOrder(),graph.zOrder());
	if (graph.critLength()>0) graph_local.refineIfLong(graph.critLength());
	BendingSolver2d bend_local(smesh,betasp,bend.tolerance(),bend.brentTolerance());
	Array1d<Point2d> path_local;
	const int max_np = int(2*sqrt(float(smesh.numNodes())));
	path_local.reserve(max_np);
	Array1d<int> start_i_local, end_i_local;
	Array1d<const Interface2d*> interp_local;
	start_i_local.reserve(2); end_i_local.reserve(2); interp_local.reserve(2);
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
	    graph_local.solve(src(isrc));
	    bool is_refl_solved=false;
	    double end_t = fwd_wall_seconds();
	    graph_time_local += end_t-start_t;

	    start_t = fwd_wall_seconds();
	    double graph_refl_time=0.0;
	    for (int ircv=1; ircv<=rcv(isrc).size(); ircv++){
		Point2d r = rcv(isrc)(ircv);
		double orig_t, final_t;
		int icode = raytype(isrc)(ircv);
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
			smesh.set(tmp_modelv);
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
			smesh.set(modelv);
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
