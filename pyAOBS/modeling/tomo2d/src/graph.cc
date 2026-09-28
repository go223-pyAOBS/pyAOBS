/*
 * graph.cc - graph solver implementation
 *
 * Jun Korenaga, MIT/WHOI
 * January 1999
 */

#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <ctime>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "smesh.h"
#include "graph.h"
#include <util.h>

// TOMO2D_GRAPH_FS_ENUM!=0：按 forward-star 下标枚举邻居（O((N+E) log N)）
// 未设或 =0：原策略，扫剩余 C/B 再用 isIn 过滤。GUI「并行/策略」勾选写入 1。
static bool graph_fs_enum_enabled()
{
    const char* e = getenv("TOMO2D_GRAPH_FS_ENUM");
    return e && atoi(e) != 0;
}

static void graph_fs_enum_log_on()
{
#ifdef _OPENMP
#pragma omp critical(tomo2d_graph_fs_enum_log)
#endif
    {
	static int once = 0;
	if (!once){
	    once = 1;
	    cerr << "GraphSolver2d:: forward-star neighbor enum ON (TOMO2D_GRAPH_FS_ENUM=1); "
		    "unset/0 = original C/B scan\n";
	    cerr.flush();
	}
    }
}

typedef bool (*FsAllowBUpdate)(const SlownessMesh2d&, Array1d<int>&, int, const Interface2d*);

// Type 0/1 original field is Vp. Dual stores that in pgrid_vp.
static double calc_ttime_orig_vp(const SlownessMesh2d& smesh, int a, int b)
{
    if (smesh.hasDualVs()) return smesh.calc_ttime_vp(a, b);
    return smesh.calc_ttime(a, b);
}

static bool fs_allow_b_not_below_itf(const SlownessMesh2d& smesh, Array1d<int>& prv,
				     int v, const Interface2d* itc)
{
    int pre = prv(v);
    double pBx = smesh.nodePos(pre).x();
    double pBz = smesh.nodePos(pre).y();
    return !(itc->z(pBx) < pBz);
}

struct FsNodeMinHeap {
    Array1d<double>* tt;
    std::vector<int> h;
    std::vector<int> pos;

    FsNodeMinHeap(Array1d<double>* t, int nnodes) : tt(t), pos(nnodes + 1, -1) {}

    bool empty() const { return h.empty(); }

    void swap_at(int i, int j)
    {
	std::swap(h[i], h[j]);
	pos[h[i]] = i;
	pos[h[j]] = j;
    }

    void up(int i)
    {
	while (i > 0){
	    int p = (i - 1) / 2;
	    if ((*tt)(h[p]) <= (*tt)(h[i])) break;
	    swap_at(i, p);
	    i = p;
	}
    }

    void down(int i)
    {
	int n = (int)h.size();
	while (true){
	    int l = 2 * i + 1, r = l + 1, best = i;
	    if (l < n && (*tt)(h[l]) < (*tt)(h[best])) best = l;
	    if (r < n && (*tt)(h[r]) < (*tt)(h[best])) best = r;
	    if (best == i) break;
	    swap_at(i, best);
	    i = best;
	}
    }

    void push(int n)
    {
	if (n <= 0 || pos[n] >= 0) return;
	pos[n] = (int)h.size();
	h.push_back(n);
	up(pos[n]);
    }

    int pop()
    {
	int n0 = h[0];
	pos[n0] = -1;
	int last = h.back();
	h.pop_back();
	if (!h.empty()){
	    h[0] = last;
	    pos[last] = 0;
	    down(0);
	}
	return n0;
    }

    void decreased(int n)
    {
	if (n > 0 && pos[n] >= 0) up(pos[n]);
    }
};

static void for_each_fs_neighbor(const SlownessMesh2d& smesh, int N0,
				 int xorder, int zorder, void (*fn)(int, void*), void* ctx)
{
    Index2d o = smesh.nodeIndex(N0);
    ForwardStar2d fstar(o, xorder, zorder);
    int i0 = o.i(), k0 = o.k();
    int nx = smesh.Nx(), nz = smesh.Nz();
    for (int i = i0 - xorder; i <= i0 + xorder; ++i){
	if (i < 1 || i > nx) continue;
	for (int k = k0 - zorder; k <= k0 + zorder; ++k){
	    if (k < 1 || k > nz) continue;
	    if (i == i0 && k == k0) continue;
	    if (!fstar.isIn(Index2d(i, k))) continue;
	    fn(smesh.nodeIndex(i, k), ctx);
	}
    }
}

struct FsRelaxCtx {
    const SlownessMesh2d* smesh;
    Array1d<double>* tt;
    Array1d<int>* prv;
    std::vector<char>* in_C;
    std::vector<char>* in_B;
    std::vector<char>* done;
    FsNodeMinHeap* heap;
    int N0;
    FsAllowBUpdate allow_b;
    const Interface2d* itc;
};

static void fs_relax_one(int v, void* raw)
{
    FsRelaxCtx* c = static_cast<FsRelaxCtx*>(raw);
    if (v == c->N0) return;
    if ((*c->done)[v]) return;
    if ((*c->in_C)[v]){
	(*c->tt)(v) = (*c->tt)(c->N0) + c->smesh->calc_ttime(c->N0, v);
	(*c->prv)(v) = c->N0;
	(*c->in_C)[v] = 0;
	(*c->in_B)[v] = 1;
	c->heap->push(v);
	return;
    }
    if ((*c->in_B)[v]){
	if (c->allow_b && !c->allow_b(*c->smesh, *c->prv, v, c->itc)) return;
	double tmp = (*c->tt)(c->N0) + c->smesh->calc_ttime(c->N0, v);
	if ((*c->tt)(v) > tmp){
	    (*c->tt)(v) = tmp;
	    (*c->prv)(v) = c->N0;
	    c->heap->decreased(v);
	}
    }
}

static void dijkstra_fs_enum(const SlownessMesh2d& smesh, int xorder, int zorder, int nnodes,
			     Array1d<double>& tt, Array1d<int>& prv,
			     std::vector<char>& in_C, const std::vector<int>& start_B,
			     FsAllowBUpdate allow_b, const Interface2d* itc)
{
    std::vector<char> in_B(nnodes + 1, 0);
    std::vector<char> done(nnodes + 1, 0);
    FsNodeMinHeap heap(&tt, nnodes);
    for (size_t i = 0; i < start_B.size(); ++i){
	int n = start_B[i];
	in_B[n] = 1;
	in_C[n] = 0;
	heap.push(n);
    }
    FsRelaxCtx ctx;
    ctx.smesh = &smesh;
    ctx.tt = &tt;
    ctx.prv = &prv;
    ctx.in_C = &in_C;
    ctx.in_B = &in_B;
    ctx.done = &done;
    ctx.heap = &heap;
    ctx.allow_b = allow_b;
    ctx.itc = itc;
    while (!heap.empty()){
	int N0 = heap.pop();
	if (done[N0]) continue;
	done[N0] = 1;
	in_B[N0] = 0;
	ctx.N0 = N0;
	for_each_fs_neighbor(smesh, N0, xorder, zorder, fs_relax_one, &ctx);
    }
}

static bool run_fs_enum_if_enabled(const SlownessMesh2d& smesh, int xorder, int zorder, int nnodes,
				   heap_deque<int, TTCmp>& heap, list<int>& C,
				   Array1d<double>& tt, Array1d<int>& prv,
				   FsAllowBUpdate allow_b, const Interface2d* itc)
{
    if (!graph_fs_enum_enabled()) return false;
    graph_fs_enum_log_on();
    std::vector<char> in_C(nnodes + 1, 0);
    for (list<int>::const_iterator it = C.begin(); it != C.end(); ++it)
	in_C[*it] = 1;
    std::vector<int> start_B;
    start_B.reserve(heap.size());
    for (heap_deque<int, TTCmp>::const_iterator p = heap.begin(); p != heap.end(); ++p)
	start_B.push_back(*p);
    dijkstra_fs_enum(smesh, xorder, zorder, nnodes, tt, prv, in_C, start_B, allow_b, itc);
    C.clear();
    heap.resize(0);
    return true;
}

GraphSolver2d::GraphSolver2d(const SlownessMesh2d& m, int xorder, int zorder)
    : smesh(m), nnodes(m.numNodes()),
      is_solved(false), is_refl_solved(false), solve_refl_downward(false),
      is_water_solved(false), is_water_mult_solved(false),
      is_recv_peg_refr_solved(false), is_recv_peg_refl_solved(false),
      is_psp_solved(false), is_psp_moho_solved(false), is_psp_peg_solved(false),
      is_psp_ss_solved(false),
      psx_two_leg(false),
      fs_xorder(xorder), fs_zorder(zorder),
      cmp(ttime), B(cmp),
      cmp_down(ttime_down), B_down(cmp_down),
      cmp_up(ttime_up), B_up(cmp_up),
      cmp_w0(ttime_w0), B_w0(cmp_w0),
      cmp_w1(ttime_w1), B_w1(cmp_w1),
      cmp_w2(ttime_w2), B_w2(cmp_w2),
      cmp_rp_crust(ttime_rp_crust), B_rp_crust(cmp_rp_crust),
      cmp_rp2(ttime_rp2), B_rp2(cmp_rp2),
      cmp_sp(ttime_sp), B_sp(cmp_sp),
      cmp_rf_down(ttime_rf_down), B_rf_down(cmp_rf_down),
      cmp_psp_a(ttime_psp_a), B_psp_a(cmp_psp_a),
      cmp_psp_b(ttime_psp_b), B_psp_b(cmp_psp_b),
      cmp_psp_c(ttime_psp_c), B_psp_c(cmp_psp_c),
      cmp_psm_u(ttime_psm_u), B_psm_u(cmp_psm_u),
      cmp_ss_dn(ttime_ss_dn), B_ss_dn(cmp_ss_dn),
      cmp_ss_up(ttime_ss_up), B_ss_up(cmp_ss_up),
      water_seafloor(0), water_surface(0), water_moho(0),
      psp_convp(0), psp_mohop(0),
      psp_istar(0),
      local_search_radius(10.0), xmin(smesh.xmin()), xmax(smesh.xmax()),
      refine_if_long(false), ttime_scale(1.0), edge_use_vp(false),
      edge_vp_mixed_conv(false), edge_psx_leg(0)
{
    prev_node.resize(nnodes);
    ttime.resize(nnodes);
    prev_w0.resize(nnodes); ttime_w0.resize(nnodes);
    prev_w1.resize(nnodes); ttime_w1.resize(nnodes);
    prev_w2.resize(nnodes); ttime_w2.resize(nnodes);
    prev_rp_crust.resize(nnodes); ttime_rp_crust.resize(nnodes);
    prev_rp2.resize(nnodes); ttime_rp2.resize(nnodes);
    prev_sp.resize(nnodes); ttime_sp.resize(nnodes);
    prev_rf_down.resize(nnodes); ttime_rf_down.resize(nnodes);
    prev_psp_a.resize(nnodes); ttime_psp_a.resize(nnodes);
    prev_psp_b.resize(nnodes); ttime_psp_b.resize(nnodes);
    prev_psp_c.resize(nnodes); ttime_psp_c.resize(nnodes);
    prev_psm_u.resize(nnodes); ttime_psm_u.resize(nnodes);
    prev_ss_dn.resize(nnodes); ttime_ss_dn.resize(nnodes);
    prev_ss_up.resize(nnodes); ttime_ss_up.resize(nnodes);
}

void GraphSolver2d::limitRange(double x1, double x2)
{
    if (x1>x2) error("GraphSolver2d::limitRange - invalid input");
    xmin=x1-local_search_radius;
    xmax=x2+local_search_radius;
}

void GraphSolver2d::delimitRange()
{
    xmin = smesh.xmin();
    xmax = smesh.xmax();
}

void GraphSolver2d::refineIfLong(double x)
{
    refine_if_long = true;
    crit_len = x;
}

void GraphSolver2d::solve(const Point2d& s)
{
    double ttime_inf = 1e30;

    src = s;
    if (smesh.inWater(src) || smesh.inAir(src)){
	error("GraphSolver2d::solve - currently unsupported source configuration detected.\n");
    }
    int Nsrc = smesh.nearest(src);
    
    // initialization
    B.resize(0);
    prev_node(Nsrc) = Nsrc;
    ttime(Nsrc) = 0.0;
    B.push(Nsrc); 
	
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	if (i!=Nsrc){
	    double x = smesh.nodePos(i).x();
	    if (x>=xmin && x<=xmax){
		C.push_back(i);
		ttime(i) = ttime_inf;
		prev_node(i) = Nsrc;
	    }
	}
    }

    if (run_fs_enum_if_enabled(smesh, fs_xorder, fs_zorder, nnodes,
			      B, C, ttime, prev_node, 0, 0)){
	is_solved = true;
	return;
    }
    const bool fa_vp = smesh.hasDualVs();
    while(B.size() != 0){
	// find a node with minimum traveltime from B,
	// (and transfer it to A)
	int N0=B.top();
	B.pop();

	// form a forward star
	ForwardStar2d fstar(smesh.nodeIndex(N0), fs_xorder, fs_zorder);

	// loop over [FS and B] to update traveltime
	heapBrowser pB=B.begin();
	while(pB!=B.end()){
	    if (fstar.isIn(smesh.nodeIndex(*pB))){
		double dtt = fa_vp ? smesh.calc_ttime_vp(N0,*pB)
				    : smesh.calc_ttime(N0,*pB);
		double tmp_ttime=ttime(N0)+dtt;
		double orig_ttime=ttime(*pB);
 		if (orig_ttime>tmp_ttime){
		    ttime(*pB) = tmp_ttime;
		    prev_node(*pB) = N0;
		    B.promote(pB);
		}
	    }
	    pB++;  
	} 
	
	// loop over [FS and C] to calculate traveltime
	// and transfer them to B
	nodeIterator pC=C.begin();
	nodeIterator epC;
	while(pC!=C.end()){
	    if (fstar.isIn(smesh.nodeIndex(*pC))){
		double dtt = fa_vp ? smesh.calc_ttime_vp(N0,*pC)
				    : smesh.calc_ttime(N0,*pC);
		ttime(*pC) = ttime(N0)+dtt;
		prev_node(*pC) = N0;
		B.push(*pC);

		epC = pC;
		pC++;
		C.erase(epC);
	    }else{ 
		pC++;
	    }
	}
    }
    is_solved = true;
}

void GraphSolver2d::solve_conv(const Point2d& s, const Interface2d& itc)
{
    double ttime_inf = 1e30;
    Array1d<int> itcnodes;
    cerr << "test solve_conv1\n";
    smesh.nearest(itc,itcnodes);
    cerr << "test solve_conv2\n";
    src = s;
    if (smesh.inWater(src) || smesh.inAir(src)){
        error("GraphSolver2d::solve - currently unsupported source configuration detected.\n");
    }
    int Nsrc = smesh.nearest(src);

    // initialization
    B.resize(0);
    prev_node(Nsrc) = Nsrc;
    ttime(Nsrc) = 0.0;
    B.push(Nsrc);

    C.resize(0);
    for (int i=1; i<=nnodes; i++){
        if (i!=Nsrc){
            double x = smesh.nodePos(i).x();
            if (x>=xmin && x<=xmax){
                C.push_back(i);
                ttime(i) = ttime_inf;
                prev_node(i) = Nsrc;
            }
        }
    }

    if (run_fs_enum_if_enabled(smesh, fs_xorder, fs_zorder, nnodes,
			      B, C, ttime, prev_node, fs_allow_b_not_below_itf, &itc)){
	is_solved = true;
	return;
    }
    while(B.size() != 0){
        // find a node with minimum traveltime from B,
        // (and transfer it to A)
        int N0=B.top();
        B.pop();

        // form a forward star
        ForwardStar2d fstar(smesh.nodeIndex(N0), fs_xorder, fs_zorder);

        // loop over [FS and B] to update traveltime
        heapBrowser pB=B.begin();

        while(pB!=B.end()){
            int pre_pB = prev_node(*pB);
            double pBx = smesh.nodePos(pre_pB).x();
            double pBz = smesh.nodePos(pre_pB).y();
            double itcz = itc.z(pBx);
            bool belowi=false;
            if (itcz<pBz){
                belowi=true;
            }

            if (fstar.isIn(smesh.nodeIndex(*pB))){
                double tmp_ttime=ttime(N0)+smesh.calc_ttime(N0,*pB);
                double orig_ttime=ttime(*pB);
                if (orig_ttime>tmp_ttime && !belowi){
                    ttime(*pB) = tmp_ttime;
                    prev_node(*pB) = N0;
                    B.promote(pB);
                }
            }
            pB++;
        }

        // loop over [FS and C] to calculate traveltime
        // and transfer them to B
        nodeIterator pC=C.begin();
        nodeIterator epC;
        while(pC!=C.end()){
            if (fstar.isIn(smesh.nodeIndex(*pC))){
                ttime(*pC) = ttime(N0)+smesh.calc_ttime(N0,*pC);
                prev_node(*pC) = N0;
                B.push(*pC);

                epC = pC;
                pC++;
                C.erase(epC);
            }else{
                pC++;
            }
        }
    }
    is_solved = true;
}

void GraphSolver2d::do_refl_downward(){ solve_refl_downward = true; }

void GraphSolver2d::solve_refl(const Point2d& s, const Interface2d& itf)
{
    if (!is_solved && !solve_refl_downward)
	error("GraphSolver2d::not ready for solve_refl()");
    
    double ttime_inf = 1e30;

    if (prev_node_down.size() != nnodes) prev_node_down.resize(nnodes);
    if (ttime_down.size() != nnodes) ttime_down.resize(nnodes);
    if (prev_node_up.size() != nnodes) prev_node_up.resize(nnodes);
    if (ttime_up.size() != nnodes) ttime_up.resize(nnodes);
    itfp = &itf;
    Array1d<int> itfnodes;
    smesh.nearest(itf,itfnodes);

    if (solve_refl_downward){
	// initialization (downgoing)
	src = s;
	if (smesh.inWater(src) || smesh.inAir(src)){
	    error("GraphSolver2d::solve_refl - currently unsupported source configuration detected.\n");
	}
	int Nsrc = smesh.nearest(src);
    
	B_down.resize(0);
	prev_node_down(Nsrc) = Nsrc;
	ttime_down(Nsrc) = 0.0;
	B_down.push(Nsrc); 
	
	C.resize(0);
	for (int i=1; i<=nnodes; i++){
	    if (i!=Nsrc){
		double x = smesh.nodePos(i).x();
		double z = smesh.nodePos(i).y();
		double itfz = itf.z(x);
		if (x>=xmin && x<=xmax){
		    bool is_ok = false;
		    if (z<=itfz){
			is_ok = true;
		    }else{
			for (int n=1; n<=itfnodes.size(); n++){
			    if (itfnodes(n)==i){
				is_ok = true; break;
			    }
			}
		    }
		    if (is_ok){
			C.push_back(i);
			ttime_down(i) = ttime_inf;
			prev_node_down(i) = Nsrc;
		    }
		}
	    }
	}

	if (!run_fs_enum_if_enabled(smesh, fs_xorder, fs_zorder, nnodes,
				    B_down, C, ttime_down, prev_node_down, 0, 0))
	while(B_down.size() != 0){
	    // find a node with minimum traveltime from B,
	    // (and transfer it to A)
	    int N0=B_down.top();
	    B_down.pop();

	    // form a forward star
	    ForwardStar2d fstar(smesh.nodeIndex(N0), fs_xorder, fs_zorder);

	    // loop over [FS and B] to update traveltime
	    heapBrowser pB=B_down.begin();
	    while(pB!=B_down.end()){
		if (fstar.isIn(smesh.nodeIndex(*pB))){
		    double tmp_ttime=ttime_down(N0)+calc_ttime_orig_vp(smesh,N0,*pB);
		    double orig_ttime=ttime_down(*pB);
		    if (orig_ttime>tmp_ttime){
			ttime_down(*pB) = tmp_ttime;
			prev_node_down(*pB) = N0;
			B_down.promote(pB);
		    }
		}
		pB++;  
	    } 
	
	    // loop over [FS and C] to calculate traveltime
	    // and transfer them to B
	    nodeIterator pC=C.begin();
	    nodeIterator epC;
	    while(pC!=C.end()){
		if (fstar.isIn(smesh.nodeIndex(*pC))){
		    ttime_down(*pC) = ttime_down(N0)+calc_ttime_orig_vp(smesh,N0,*pC);
		    prev_node_down(*pC) = N0;
		    B_down.push(*pC);

		    epC = pC;
		    pC++;
		    C.erase(epC);
		}else{ 
		    pC++;
		}
	    }
	}
    }else{
	// use refraction's graph solution
	prev_node_down = prev_node;
	ttime_down = ttime;
    }
    
    // initialization (upgoing)
    B_up.resize(0);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	bool is_itf = false;
	double x=smesh.nodePos(i).x();
	double z=smesh.nodePos(i).y();
	double itfz=itf.z(x);
	if (x>=xmin && x<=xmax){
	    for (int n=1; n<=itfnodes.size(); n++){
		if (itfnodes(n)==i){
		    is_itf = true;
		    prev_node_up(i) = -1;
		    ttime_up(i) = ttime_down(i);
		    B_up.push(i);
		    break;
		}
	    }
	    if (!is_itf && z<=itfz){ // take nodes above the reflector
		ttime_up(i) = ttime_inf;
		prev_node_up(i) = 0;
		C.push_back(i);
	    }
	}
    }

    if (!run_fs_enum_if_enabled(smesh, fs_xorder, fs_zorder, nnodes,
			       B_up, C, ttime_up, prev_node_up, 0, 0))
    while(B_up.size() != 0){
	// find a node with minimum traveltime from B,
	// (and transfer it to A)
	int N0=B_up.top();
	B_up.pop();

	// form a forward star
	ForwardStar2d fstar(smesh.nodeIndex(N0), fs_xorder, fs_zorder);

	// loop over [FS and B] to update traveltime
	heapBrowser pB=B_up.begin();
	while(pB!=B_up.end()){
	    if (fstar.isIn(smesh.nodeIndex(*pB))){
		double tmp_ttime=ttime_up(N0)+calc_ttime_orig_vp(smesh,N0,*pB);
		double orig_ttime=ttime_up(*pB);
 		if (orig_ttime>tmp_ttime){
			ttime_up(*pB) = tmp_ttime;
			prev_node_up(*pB) = N0;
			B_up.promote(pB);
		    }
		}
		pB++;  
	    }
	
	// loop over [FS and C] to calculate traveltime
	// and transfer them to B
	nodeIterator pC=C.begin();
	nodeIterator epC;
	while(pC!=C.end()){
	    if (fstar.isIn(smesh.nodeIndex(*pC))){
		ttime_up(*pC) = ttime_up(N0)+calc_ttime_orig_vp(smesh,N0,*pC);
		prev_node_up(*pC) = N0;
		B_up.push(*pC);
		epC = pC;
		pC++;
		C.erase(epC);
	    }else{ 
		pC++;
	    }
	}
    }
    
    is_refl_solved = true;
}

void GraphSolver2d::solve_refl_conv(const Point2d& s, const Interface2d& itf, const Interface2d& itc)
{
    if (!is_solved && !solve_refl_downward)
        error("GraphSolver2d::not ready for solve_refl()");

    double ttime_inf = 1e30;

    if (prev_node_down.size() != nnodes) prev_node_down.resize(nnodes);
    if (ttime_down.size() != nnodes) ttime_down.resize(nnodes);
    if (prev_node_up.size() != nnodes) prev_node_up.resize(nnodes);
    if (ttime_up.size() != nnodes) ttime_up.resize(nnodes);
    itfp = &itf;
    Array1d<int> itfnodes,itcnodes;
    smesh.nearest(itf,itfnodes);
    smesh.nearest(itc,itcnodes);
    if (solve_refl_downward){
        // initialization (downgoing)
        src = s;
        if (smesh.inWater(src) || smesh.inAir(src)){
            error("GraphSolver2d::solve_refl - currently unsupported source configuration detected.\n");
        }
        int Nsrc = smesh.nearest(src);

        B_down.resize(0);
        prev_node_down(Nsrc) = Nsrc;
        ttime_down(Nsrc) = 0.0;
        B_down.push(Nsrc);

        C.resize(0);
        for (int i=1; i<=nnodes; i++){
            if (i!=Nsrc){
                double x = smesh.nodePos(i).x();
                double z = smesh.nodePos(i).y();
                double itfz = itf.z(x);
                if (x>=xmin && x<=xmax){
                    bool is_ok = false;
                    if (z<=itfz){
                        is_ok = true;
                    }else{
                        for (int n=1; n<=itfnodes.size(); n++){
                            if (itfnodes(n)==i){
                                is_ok = true; break;
                            }
                        }
                    }
                    if (is_ok){
                        C.push_back(i);
                        ttime_down(i) = ttime_inf;
                        prev_node_down(i) = Nsrc;
                    }
                }
            }
        }

        if (!run_fs_enum_if_enabled(smesh, fs_xorder, fs_zorder, nnodes,
				    B_down, C, ttime_down, prev_node_down,
				    fs_allow_b_not_below_itf, &itc))
        while(B_down.size() != 0){
            // find a node with minimum traveltime from B,
            // (and transfer it to A)
            int N0=B_down.top();
            B_down.pop();

            // form a forward star
            ForwardStar2d fstar(smesh.nodeIndex(N0), fs_xorder, fs_zorder);

            // loop over [FS and B] to update traveltime
            heapBrowser pB=B_down.begin();
            while(pB!=B_down.end()){
                int pre_pB = prev_node(*pB);
                double pBx = smesh.nodePos(pre_pB).x();
                double pBz = smesh.nodePos(pre_pB).y();
                double itcz = itc.z(pBx);
                bool belowi=false;
                if (itcz<pBz)belowi=true;
                if (fstar.isIn(smesh.nodeIndex(*pB))){
                    double tmp_ttime=ttime_down(N0)+calc_ttime_orig_vp(smesh,N0,*pB);
                    double orig_ttime=ttime_down(*pB);
                    if (orig_ttime>tmp_ttime && !belowi){
                        ttime_down(*pB) = tmp_ttime;
                        prev_node_down(*pB) = N0;
                        B_down.promote(pB);
                    }
                }
                pB++;
            }

            // loop over [FS and C] to calculate traveltime
            // and transfer them to B
            nodeIterator pC=C.begin();
            nodeIterator epC;
            while(pC!=C.end()){
                if (fstar.isIn(smesh.nodeIndex(*pC))){
                    ttime_down(*pC) = ttime_down(N0)+calc_ttime_orig_vp(smesh,N0,*pC);
                    prev_node_down(*pC) = N0;
                    B_down.push(*pC);

                    epC = pC;
                    pC++;
                    C.erase(epC);
                }else{
                    pC++;
                }
            }
        }
    }else{
        // use refraction's graph solution
        prev_node_down = prev_node;
        ttime_down = ttime;
    }

    // initialization (upgoing)
    B_up.resize(0);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
        bool is_itf = false;
        double x=smesh.nodePos(i).x();
        double z=smesh.nodePos(i).y();
        double itfz=itf.z(x);
        if (x>=xmin && x<=xmax){
            for (int n=1; n<=itfnodes.size(); n++){
                if (itfnodes(n)==i){
                    is_itf = true;
                    prev_node_up(i) = -1;
                    ttime_up(i) = ttime_down(i);
                    B_up.push(i);
                    break;
                }
            }
            if (!is_itf && z<=itfz){ // take nodes above the reflector
                ttime_up(i) = ttime_inf;
                prev_node_up(i) = 0;
                C.push_back(i);
            }
        }
    }

    if (!run_fs_enum_if_enabled(smesh, fs_xorder, fs_zorder, nnodes,
			       B_up, C, ttime_up, prev_node_up,
			       fs_allow_b_not_below_itf, &itc))
    while(B_up.size() != 0){
        // find a node with minimum traveltime from B,
        // (and transfer it to A)
        int N0=B_up.top();
        B_up.pop();

        // form a forward star
        ForwardStar2d fstar(smesh.nodeIndex(N0), fs_xorder, fs_zorder);

        // loop over [FS and B] to update traveltime
        heapBrowser pB=B_up.begin();
        while(pB!=B_up.end()){
            int pre_pB = prev_node(*pB);
            double pBx = smesh.nodePos(pre_pB).x();
            double pBz = smesh.nodePos(pre_pB).y();
            double itcz = itc.z(pBx);
            bool belowi=false;
            if (itcz<pBz)belowi=true;
            if (fstar.isIn(smesh.nodeIndex(*pB))){
                double tmp_ttime=ttime_up(N0)+calc_ttime_orig_vp(smesh,N0,*pB);
                double orig_ttime=ttime_up(*pB);
                if (orig_ttime>tmp_ttime && !belowi){
                    ttime_up(*pB) = tmp_ttime;
                    prev_node_up(*pB) = N0;
                    B_up.promote(pB);
                }
            }
            pB++;
        }

        // loop over [FS and C] to calculate traveltime
        // and transfer them to B
        nodeIterator pC=C.begin();
        nodeIterator epC;
        while(pC!=C.end()){
            if (fstar.isIn(smesh.nodeIndex(*pC))){
                ttime_up(*pC) = ttime_up(N0)+calc_ttime_orig_vp(smesh,N0,*pC);
                prev_node_up(*pC) = N0;
                B_up.push(*pC);
                epC = pC;
                pC++;
                C.erase(epC);
            }else{
                pC++;
            }
        }
    }

    is_refl_solved = true;
}

void GraphSolver2d::pickPath(const Point2d& rcv, Array1d<Point2d>& path) const
{
    if (!is_solved) error("GraphSolver2d::not yet solved");

    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);

    int cur = smesh.nearest(rcv);
    int prev;
    int nadd=0;
    while ((prev=prev_node(cur)) != cur){
	Point2d p = smesh.nodePos(prev);
	tmp_path.push_back(p); nadd++;
	cur = prev;
    }

    if (nadd>0){
	tmp_path.pop_back();
    }
    if (nadd==1){
	// insert mid point for later bending refinement
	tmp_path.push_back(0.5*(src+rcv));
    }
    tmp_path.push_back(src);

    if (refine_if_long){
	list<Point2d>::iterator pt=tmp_path.begin();
	Point2d prev_p = *pt;
	pt++; // skip the first point
	while(pt!=tmp_path.end()){
	    double seg_len = prev_p.distance(*pt);
	    if (seg_len>crit_len){
		int ndiv = int(seg_len/crit_len+1.0);
		double frac = 1.0/ndiv;
		for (int j=1; j<ndiv; j++){
		    double ratio = frac*j;
		    Point2d new_p = (1.0-ratio)*prev_p+ratio*(*pt);
		    tmp_path.insert(pt,new_p);
		}
	    }
	    prev_p = *pt++;
	}
    }

    path.resize(tmp_path.size());
    list<Point2d>::const_iterator pt=tmp_path.begin();
    int i=1;
    while(pt!=tmp_path.end()){
	path(i++) = *pt++;
    }
}

void GraphSolver2d::pickPathThruWater(const Point2d& rcv, Array1d<Point2d>& path,
				      int& i0, int& i1) const
{
    if (!is_solved) error("GraphSolver2d::not yet solved");

    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);

    int cur = smesh.nearest(rcv);
    findRayEntryPoint(rcv,cur,ttime);

    Point2d entryp=smesh.nodePos(cur);
    tmp_path.push_back(entryp);
    tmp_path.push_back(entryp);
    tmp_path.push_back(entryp);
    i0 = 2; i1 = 4;
    
    int prev;
    int nadd=0;
    while ((prev=prev_node(cur)) != cur){
	Point2d p = smesh.nodePos(prev);
	tmp_path.push_back(p); nadd++;
	cur = prev;
    }
    
    if (nadd>0){
	tmp_path.pop_back();
    }
    if (nadd==1){
	// insert mid point for later bending refinement
	tmp_path.push_back(0.5*(src+entryp));
    }
    tmp_path.push_back(src);

    if (refine_if_long){
	list<Point2d>::iterator pt=tmp_path.begin();
	pt++; pt++; pt++; // now at the entry point
	Point2d prev_p = *pt;
	pt++; // skip the entry point
	while(pt!=tmp_path.end()){
	    double seg_len = prev_p.distance(*pt);
	    if (seg_len>crit_len){
		int ndiv = int(seg_len/crit_len+1.0);
		double frac = 1.0/ndiv;
		for (int j=1; j<ndiv; j++){
		    double ratio = frac*j;
		    Point2d new_p = (1.0-ratio)*prev_p+ratio*(*pt);
		    tmp_path.insert(pt,new_p);
		}
	    }
	    prev_p = *pt++;
	}
    }
    
    path.resize(tmp_path.size());
    list<Point2d>::const_iterator pt=tmp_path.begin();
    int i=1;
    while(pt!=tmp_path.end()){
	path(i++) = *pt++;
    }
}

void GraphSolver2d::pickReflPath(const Point2d& rcv, Array1d<Point2d>& path,
				 int& ir0, int& ir1) const
{
    if (!is_refl_solved) error("GraphSolver2d::not yet solved");

    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);

    int cur = smesh.nearest(rcv);
    int prev;
    int nadd=0;
    while ((prev=prev_node_up(cur)) != -1){
	Point2d p = smesh.nodePos(prev);
	tmp_path.push_back(p); nadd++; 
	cur = prev;
    }
    double reflx = smesh.nodePos(cur).x();
    double reflz = itfp->z(reflx);
    Point2d onrefl(reflx,reflz);

    if (nadd>0){
	tmp_path.pop_back();
    }
    if (nadd==1){
	// insert mid point for later bending refinement
	tmp_path.push_back(0.5*(onrefl+rcv));
    }

    tmp_path.push_back(onrefl); ir0 = tmp_path.size();
    tmp_path.push_back(onrefl);
    tmp_path.push_back(onrefl); ir1 = tmp_path.size();
    
    nadd=0;
    while ((prev=prev_node_down(cur)) != cur){
	Point2d p = smesh.nodePos(prev);
	tmp_path.push_back(p); nadd++;
	cur = prev;
    }
    if (nadd>0){
	tmp_path.pop_back();
    }
    if (nadd==1){
	// insert mid point for later bending refinement
	tmp_path.push_back(0.5*(src+rcv));
    }
    tmp_path.push_back(src);

    if (refine_if_long){
	int old_i=1, nadd_pre=0;
	list<Point2d>::iterator pt=tmp_path.begin();
	Point2d prev_p = *pt;
	pt++; old_i++; // skip the first point
	while(pt!=tmp_path.end()){
	    double seg_len = prev_p.distance(*pt);
	    if (seg_len>crit_len){
		int ndiv = int(seg_len/crit_len+1.0);
		double frac = 1.0/ndiv;
		for (int j=1; j<ndiv; j++){
		    double ratio = frac*j;
		    Point2d new_p = (1.0-ratio)*prev_p+ratio*(*pt);
		    tmp_path.insert(pt,new_p);
		    if (old_i<=ir0) nadd_pre++;
		}
	    }
	    prev_p = *pt++; old_i++;
	}
	ir0 += nadd_pre;
	ir1 += nadd_pre;
    }
    
    path.resize(tmp_path.size());
    list<Point2d>::const_iterator pt=tmp_path.begin();
    int i=1;
    while(pt!=tmp_path.end()){
	path(i++) = *pt++;
    }
}

void GraphSolver2d::pickReflPathThruWater(const Point2d& rcv, Array1d<Point2d>& path,
					  int& i0, int& i1, int& ir0, int& ir1) const
{
    if (!is_refl_solved) error("GraphSolver2d::not yet solved");

    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);

    int cur = smesh.nearest(rcv);
    findRayEntryPoint(rcv,cur,ttime_up);

    Point2d entryp=smesh.nodePos(cur);
    tmp_path.push_back(entryp);
    tmp_path.push_back(entryp);
    tmp_path.push_back(entryp);
    i0 = 2; i1 = 4;
    
    int prev;
    int nadd=0;
    while ((prev=prev_node_up(cur)) != -1){
	Point2d p = smesh.nodePos(prev);
	tmp_path.push_back(p); nadd++; 
	cur = prev;
    }
    double reflx = smesh.nodePos(cur).x();
    double reflz = itfp->z(reflx);
    Point2d onrefl(reflx,reflz);

    if (nadd>0){
	tmp_path.pop_back();
    }
    if (nadd==1){
	// insert mid point for later bending refinement
	tmp_path.push_back(0.5*(onrefl+rcv));
    }

    tmp_path.push_back(onrefl); ir0 = tmp_path.size();
    tmp_path.push_back(onrefl);
    tmp_path.push_back(onrefl); ir1 = tmp_path.size();
    
    nadd=0;
    while ((prev=prev_node_down(cur)) != cur){
	Point2d p = smesh.nodePos(prev);
	tmp_path.push_back(p); nadd++;
	cur = prev;
    }
    if (nadd>0){
	tmp_path.pop_back();
    }
    if (nadd==1){
	// insert mid point for later bending refinement
	tmp_path.push_back(0.5*(src+rcv));
    }
    tmp_path.push_back(src);

    if (refine_if_long){
	int nadd_pre=0;
	list<Point2d>::iterator pt=tmp_path.begin();
	pt++; pt++; pt++; // now at the entry point
	Point2d prev_p = *pt;
	pt++; // skip the entry point
	int old_i = 5;
	while(pt!=tmp_path.end()){
	    double seg_len = prev_p.distance(*pt);
	    if (seg_len>crit_len){
		int ndiv = int(seg_len/crit_len+1.0);
		double frac = 1.0/ndiv;
		for (int j=1; j<ndiv; j++){
		    double ratio = frac*j;
		    Point2d new_p = (1.0-ratio)*prev_p+ratio*(*pt);
		    tmp_path.insert(pt,new_p);
		    if (old_i<=ir0) nadd_pre++;
		}
	    }
	    prev_p = *pt++; old_i++;
	}
	ir0 += nadd_pre;
	ir1 += nadd_pre;
    }

    path.resize(tmp_path.size());
    list<Point2d>::const_iterator pt=tmp_path.begin();
    int i=1;
    while(pt!=tmp_path.end()){
	path(i++) = *pt++;
    }
}

void GraphSolver2d::findRayEntryPoint(const Point2d& rcv, int& cur, const Array1d<double>& nodetime) const
{
    Point2d p0 = smesh.nodePos(cur);
    double pwater = smesh.atWater();
    double ttime0 = pwater*rcv.distance(p0)+nodetime(cur);

    // local grid search for best ray entry point (+/- 10km radius)
    Point2d pmin(rcv.x()-local_search_radius,rcv.y());
    Point2d pmax(rcv.x()+local_search_radius,rcv.y()); 
    Index2d indmin = smesh.nodeIndex(smesh.nearest(pmin)); 
    Index2d indmax = smesh.nodeIndex(smesh.nearest(pmax)); 
    for (int i=indmin.i()+1; i<indmax.i(); i++){
	int inode = smesh.nodeIndex(i,1);
	Point2d p = smesh.nodePos(inode);
	double tmp_ttime = pwater*rcv.distance(p)+nodetime(inode);
	if (tmp_ttime < ttime0){
	    ttime0 = tmp_ttime;
	    cur = inode; 
	}
    }
}

void GraphSolver2d::printPath(ostream& os) const
{
    for (int cur=1; cur<=nnodes; cur++){
	int prev = prev_node(cur);
	Point2d cur_p = smesh.nodePos(cur);
	Point2d prev_p = smesh.nodePos(prev);
	    os << ">\n"
	       << cur_p.x() << " "  << cur_p.y() << "\n"
	       << prev_p.x() << " " << prev_p.y() << '\n';
    }
}

bool GraphSolver2d::node_in_water_col(int inode, const Interface2d& seafloor) const
{
    Point2d p = smesh.nodePos(inode);
    if (p.y() < -1e-8) return false;
    double x = p.x();
    if (x<xmin || x>xmax) return false;
    return p.y() <= seafloor.z(x)+1e-6;
}

bool GraphSolver2d::node_in_crust(int inode, const Interface2d& seafloor) const
{
    Point2d p = smesh.nodePos(inode);
    if (p.x()<xmin || p.x()>xmax) return false;
    return p.y() > seafloor.z(p.x()) + 1e-6;
}

void GraphSolver2d::dijkstra_crust(heap_deque<int,TTCmp>& Bq, Array1d<int>& prv,
				   Array1d<double>& tt, const Interface2d& seafloor,
				   list<int>& Cq)
{
    while(Bq.size() != 0){
	int N0=Bq.top();
	Bq.pop();
	ForwardStar2d fstar(smesh.nodeIndex(N0), fs_xorder, fs_zorder);

	heapBrowser pB=Bq.begin();
	while(pB!=Bq.end()){
	    if (fstar.isIn(smesh.nodeIndex(*pB))){
		double tmp_ttime=tt(N0)+smesh.calc_ttime(N0,*pB);
		if (tt(*pB)>tmp_ttime){
		    tt(*pB) = tmp_ttime;
		    prv(*pB) = N0;
		    Bq.promote(pB);
		}
	    }
	    pB++;
	}

	nodeIterator pC=Cq.begin();
	nodeIterator epC;
	while(pC!=Cq.end()){
	    if (fstar.isIn(smesh.nodeIndex(*pC)) && node_in_crust(*pC, seafloor)){
		tt(*pC) = tt(N0)+smesh.calc_ttime(N0,*pC);
		prv(*pC) = N0;
		Bq.push(*pC);
		epC = pC;
		pC++;
		Cq.erase(epC);
	    }else{
		pC++;
	    }
	}
    }
}

void GraphSolver2d::list_to_path(const list<Point2d>& tmp_path, Array1d<Point2d>& path) const
{
    path.resize((int)tmp_path.size());
    list<Point2d>::const_iterator pt=tmp_path.begin();
    int i=1;
    while(pt!=tmp_path.end()){
	path(i++) = *pt++;
    }
}

void GraphSolver2d::dijkstra_region(heap_deque<int,TTCmp>& Bq, Array1d<int>& prv,
				   Array1d<double>& tt, list<int>& Cq)
{
    while(Bq.size() != 0){
	int N0=Bq.top();
	Bq.pop();
	ForwardStar2d fstar(smesh.nodeIndex(N0), fs_xorder, fs_zorder);

	heapBrowser pB=Bq.begin();
	while(pB!=Bq.end()){
	    if (fstar.isIn(smesh.nodeIndex(*pB))){
		double tmp_ttime=tt(N0)+edge_tt(N0,*pB);
		if (tt(*pB)>tmp_ttime){
		    tt(*pB) = tmp_ttime;
		    prv(*pB) = N0;
		    Bq.promote(pB);
		}
	    }
	    pB++;
	}

	nodeIterator pC=Cq.begin();
	nodeIterator epC;
	while(pC!=Cq.end()){
	    if (fstar.isIn(smesh.nodeIndex(*pC))){
		tt(*pC) = tt(N0)+edge_tt(N0,*pC);
		prv(*pC) = N0;
		Bq.push(*pC);
		epC = pC;
		pC++;
		Cq.erase(epC);
	    }else{
		pC++;
	    }
	}
    }
}

void GraphSolver2d::dijkstra_water(heap_deque<int,TTCmp>& Bq, Array1d<int>& prv,
				   Array1d<double>& tt, const Interface2d& seafloor,
				   list<int>& Cq)
{
    while(Bq.size() != 0){
	int N0=Bq.top();
	Bq.pop();
	ForwardStar2d fstar(smesh.nodeIndex(N0), fs_xorder, fs_zorder);

	heapBrowser pB=Bq.begin();
	while(pB!=Bq.end()){
	    if (fstar.isIn(smesh.nodeIndex(*pB))){
		double tmp_ttime=tt(N0)+smesh.calc_ttime(N0,*pB);
		if (tt(*pB)>tmp_ttime){
		    tt(*pB) = tmp_ttime;
		    prv(*pB) = N0;
		    Bq.promote(pB);
		}
	    }
	    pB++;
	}

	nodeIterator pC=Cq.begin();
	nodeIterator epC;
	while(pC!=Cq.end()){
	    if (fstar.isIn(smesh.nodeIndex(*pC)) && node_in_water_col(*pC, seafloor)){
		tt(*pC) = tt(N0)+smesh.calc_ttime(N0,*pC);
		prv(*pC) = N0;
		Bq.push(*pC);
		epC = pC;
		pC++;
		Cq.erase(epC);
	    }else{
		pC++;
	    }
	}
    }
}

void GraphSolver2d::solve_water(const Point2d& s, const Interface2d& seafloor)
{
    const double ttime_inf = 1e30;
    src = s;
    water_seafloor = &seafloor;
    is_water_solved = false;
    is_water_mult_solved = false;
    is_recv_peg_refr_solved = false;
    is_recv_peg_refl_solved = false;

    int Nsrc = smesh.nearest(src);
    if (!node_in_water_col(Nsrc, seafloor)){
	error("GraphSolver2d::solve_water - source is not in the water column (z<=seafloor).");
    }

    B_w0.resize(0);
    prev_w0(Nsrc) = Nsrc;
    ttime_w0(Nsrc) = 0.0;
    B_w0.push(Nsrc);

    C_w.resize(0);
    for (int i=1; i<=nnodes; i++){
	if (i==Nsrc) continue;
	if (node_in_water_col(i, seafloor)){
	    C_w.push_back(i);
	    ttime_w0(i) = ttime_inf;
	    prev_w0(i) = Nsrc;
	}
    }
    dijkstra_water(B_w0, prev_w0, ttime_w0, seafloor, C_w);
    is_water_solved = true;
}

void GraphSolver2d::solve_water_mult(const Interface2d& seafloor, const Interface2d& surface)
{
    if (!is_water_solved)
	error("GraphSolver2d::solve_water_mult - call solve_water first");
    const double ttime_inf = 1e30;
    water_seafloor = &seafloor;
    water_surface = &surface;

    Array1d<int> surfnodes, botnodes;
    smesh.nearest(surface, surfnodes);
    smesh.nearest(seafloor, botnodes);

    B_w1.resize(0);
    C_w.resize(0);
    for (int i=1; i<=nnodes; i++){
	if (!node_in_water_col(i, seafloor)) continue;
	bool is_surf=false;
	for (int n=1; n<=surfnodes.size(); n++){
	    if (surfnodes(n)==i){ is_surf=true; break; }
	}
	if (is_surf){
	    prev_w1(i) = -1;
	    ttime_w1(i) = ttime_w0(i);
	    B_w1.push(i);
	}else{
	    ttime_w1(i) = ttime_inf;
	    prev_w1(i) = 0;
	    C_w.push_back(i);
	}
    }
    dijkstra_water(B_w1, prev_w1, ttime_w1, seafloor, C_w);

    B_w2.resize(0);
    C_w.resize(0);
    for (int i=1; i<=nnodes; i++){
	if (!node_in_water_col(i, seafloor)) continue;
	bool is_bot=false;
	for (int n=1; n<=botnodes.size(); n++){
	    if (botnodes(n)==i){ is_bot=true; break; }
	}
	if (is_bot){
	    prev_w2(i) = -1;
	    ttime_w2(i) = ttime_w1(i);
	    B_w2.push(i);
	}else{
	    ttime_w2(i) = ttime_inf;
	    prev_w2(i) = 0;
	    C_w.push_back(i);
	}
    }
    dijkstra_water(B_w2, prev_w2, ttime_w2, seafloor, C_w);
    is_water_mult_solved = true;
}

void GraphSolver2d::pickWaterDirectPath(const Point2d& rcv, Array1d<Point2d>& path,
					int& ir0, int& ir1) const
{
    if (!is_water_solved) error("GraphSolver2d::pickWaterDirectPath - not solved");

    // Same as pickPath: backtrack prev_w0 node chain, then optional refineIfLong.
    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);
    int cur = smesh.nearest(rcv);
    int prev;
    int nadd=0;
    while ((prev=prev_w0(cur)) != cur){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
    }
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(rcv+src));
    tmp_path.push_back(src);
    ir0 = 0;
    ir1 = 0;

    if (refine_if_long){
	list<Point2d>::iterator pt=tmp_path.begin();
	Point2d prev_p = *pt;
	pt++;
	while(pt!=tmp_path.end()){
	    double seg_len = prev_p.distance(*pt);
	    if (seg_len>crit_len){
		int ndiv = int(seg_len/crit_len+1.0);
		double frac = 1.0/ndiv;
		for (int j=1; j<ndiv; j++){
		    double ratio = frac*j;
		    Point2d new_p = (1.0-ratio)*prev_p+ratio*(*pt);
		    tmp_path.insert(pt,new_p);
		}
	    }
	    prev_p = *pt++;
	}
    }
    list_to_path(tmp_path, path);
}

void GraphSolver2d::pickWaterMultPath(const Point2d& rcv, Array1d<Point2d>& path,
				      int& i0, int& i1, int& ir0, int& ir1) const
{
    if (!is_water_mult_solved) error("GraphSolver2d::pickWaterMultPath - not solved");
    if (!water_seafloor || !water_surface)
	error("GraphSolver2d::pickWaterMultPath - interfaces missing");

    // Same as pickReflPath: keep graph node chains; pin bounce points on
    // seafloor (ir0,ir1) and surface (i0,i1). Do not replace legs with chords.
    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);
    int cur = smesh.nearest(rcv);
    int prev;
    int nadd=0;
    while ((prev=prev_w2(cur)) != -1){
	if (prev==0) error("GraphSolver2d::pickWaterMultPath - broken w2 chain");
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
    }
    double bx = smesh.nodePos(cur).x();
    Point2d onbot(bx, water_seafloor->z(bx));
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(onbot+rcv));
    tmp_path.push_back(onbot); ir0 = (int)tmp_path.size();
    tmp_path.push_back(onbot);
    tmp_path.push_back(onbot); ir1 = (int)tmp_path.size();

    nadd=0;
    while ((prev=prev_w1(cur)) != -1){
	if (prev==0) error("GraphSolver2d::pickWaterMultPath - broken w1 chain");
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
    }
    double sx = smesh.nodePos(cur).x();
    Point2d onsurf(sx, water_surface->z(sx));
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(onsurf+onbot));
    tmp_path.push_back(onsurf); i0 = (int)tmp_path.size();
    tmp_path.push_back(onsurf);
    tmp_path.push_back(onsurf); i1 = (int)tmp_path.size();

    nadd=0;
    while ((prev=prev_w0(cur)) != cur){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
    }
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(src+onsurf));
    tmp_path.push_back(src);

    if (refine_if_long){
	int old_i=1, nadd_ir=0, nadd_i=0;
	list<Point2d>::iterator pt=tmp_path.begin();
	Point2d prev_p = *pt;
	pt++; old_i++;
	while(pt!=tmp_path.end()){
	    double seg_len = prev_p.distance(*pt);
	    if (seg_len>crit_len){
		int ndiv = int(seg_len/crit_len+1.0);
		double frac = 1.0/ndiv;
		for (int j=1; j<ndiv; j++){
		    double ratio = frac*j;
		    Point2d new_p = (1.0-ratio)*prev_p+ratio*(*pt);
		    tmp_path.insert(pt,new_p);
		    if (old_i<=ir0) nadd_ir++;
		    if (old_i<=i0) nadd_i++;
		}
	    }
	    prev_p = *pt++; old_i++;
	}
	ir0 += nadd_ir;
	ir1 += nadd_ir;
	i0 += nadd_i;
	i1 += nadd_i;
    }
    list_to_path(tmp_path, path);
}

void GraphSolver2d::solve_recv_peg_refr(const Interface2d& seafloor, const Interface2d& surface)
{
    if (!is_water_solved)
	error("GraphSolver2d::solve_recv_peg_refr - call solve_water first");
    if (!is_water_mult_solved)
	error("GraphSolver2d::solve_recv_peg_refr - call solve_water_mult first");
    const double ttime_inf = 1e30;
    water_seafloor = &seafloor;
    water_surface = &surface;

    Array1d<int> botnodes;
    smesh.nearest(seafloor, botnodes);

    B_rp_crust.resize(0);
    C_w.resize(0);
    for (int i=1; i<=nnodes; i++){
	if (!node_in_crust(i, seafloor)) continue;
	ttime_rp_crust(i) = ttime_inf;
	prev_rp_crust(i) = 0;
	C_w.push_back(i);
    }
    for (int n=1; n<=botnodes.size(); n++){
	int ib = botnodes(n);
	if (ttime_w1(ib) >= ttime_inf/2.0) continue;
	Index2d id = smesh.nodeIndex(ib);
	int k = id.k();
	if (k >= smesh.Nz()) continue;
	int ic = smesh.nodeIndex(id.i(), k+1);
	if (ic < 1 || ic > nnodes) continue;
	if (!node_in_crust(ic, seafloor)) continue;
	double tseed = ttime_w1(ib)+smesh.calc_ttime(ib, ic);
	if (tseed >= ttime_rp_crust(ic)) continue;
	bool in_c = false;
	for (nodeIterator pC=C_w.begin(); pC!=C_w.end(); ){
	    if (*pC==ic){
		in_c = true;
		nodeIterator ep=pC;
		pC++;
		C_w.erase(ep);
	    }else{
		pC++;
	    }
	}
	ttime_rp_crust(ic) = tseed;
	prev_rp_crust(ic) = ib;
	if (in_c) B_rp_crust.push(ic);
    }
    dijkstra_crust(B_rp_crust, prev_rp_crust, ttime_rp_crust, seafloor, C_w);

    B_rp2.resize(0);
    C_w.resize(0);
    for (int i=1; i<=nnodes; i++){
	if (!node_in_water_col(i, seafloor)) continue;
	bool is_bot=false;
	for (int n=1; n<=botnodes.size(); n++){
	    if (botnodes(n)==i){ is_bot=true; break; }
	}
	if (is_bot){
	    Index2d id = smesh.nodeIndex(i);
	    int ic = (id.k() < smesh.Nz()) ? smesh.nodeIndex(id.i(), id.k()+1) : 0;
	    double tseed = ttime_inf;
	    int prv = -1;
	    if (ic>=1 && ic<=nnodes && ttime_rp_crust(ic) < ttime_inf/2.0){
		tseed = ttime_rp_crust(ic)+smesh.calc_ttime(ic, i);
		prv = ic;
	    }
	    prev_rp2(i) = prv;
	    ttime_rp2(i) = tseed;
	    B_rp2.push(i);
	}else{
	    ttime_rp2(i) = ttime_inf;
	    prev_rp2(i) = 0;
	    C_w.push_back(i);
	}
    }
    dijkstra_water(B_rp2, prev_rp2, ttime_rp2, seafloor, C_w);
    is_recv_peg_refr_solved = true;
}

void GraphSolver2d::pickRecvPegRefrPath(const Point2d& rcv, Array1d<Point2d>& path,
					int& is0, int& is1, int& ib0, int& ib1,
					int& ie0, int& ie1) const
{
    if (!is_recv_peg_refr_solved)
	error("GraphSolver2d::pickRecvPegRefrPath - not solved");
    if (!water_seafloor || !water_surface)
	error("GraphSolver2d::pickRecvPegRefrPath - interfaces missing");

    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);
    int cur = smesh.nearest(rcv);
    int prev;
    int nadd=0;
    while (true){
	prev = prev_rp2(cur);
	if (prev<=0) break;
	if (node_in_crust(prev, *water_seafloor)) break;
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
    }
    if (prev==0) error("GraphSolver2d::pickRecvPegRefrPath - broken rp2 chain");
    double ex = smesh.nodePos(cur).x();
    Point2d onE(ex, water_seafloor->z(ex));
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(onE+rcv));
    tmp_path.push_back(onE); ie0 = (int)tmp_path.size();
    tmp_path.push_back(onE);
    tmp_path.push_back(onE); ie1 = (int)tmp_path.size();

    if (prev>0) cur = prev;
    nadd=0;
    while (true){
	prev = prev_rp_crust(cur);
	if (prev<=0) break;
	if (!node_in_crust(prev, *water_seafloor)) break;
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
    }
    double bx = smesh.nodePos(cur).x();
    Point2d onB(bx, water_seafloor->z(bx));
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(onB+onE));
    tmp_path.push_back(onB); ib0 = (int)tmp_path.size();
    tmp_path.push_back(onB);
    tmp_path.push_back(onB); ib1 = (int)tmp_path.size();

    int ibot = (prev>0) ? prev : smesh.nearest(onB);
    nadd=0;
    cur = ibot;
    while ((prev=prev_w1(cur)) != -1){
	if (prev==0) error("GraphSolver2d::pickRecvPegRefrPath - broken w1 chain");
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
    }
    double sx = smesh.nodePos(cur).x();
    Point2d onS(sx, water_surface->z(sx));
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(onS+onB));
    tmp_path.push_back(onS); is0 = (int)tmp_path.size();
    tmp_path.push_back(onS);
    tmp_path.push_back(onS); is1 = (int)tmp_path.size();

    nadd=0;
    while ((prev=prev_w0(cur)) != cur){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
    }
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(src+onS));
    tmp_path.push_back(src);

    if (refine_if_long){
	int old_i=1, nadd_ie=0, nadd_ib=0, nadd_is=0;
	list<Point2d>::iterator pt=tmp_path.begin();
	Point2d prev_p = *pt;
	pt++; old_i++;
	while(pt!=tmp_path.end()){
	    double seg_len = prev_p.distance(*pt);
	    if (seg_len>crit_len){
		int ndiv = int(seg_len/crit_len+1.0);
		double frac = 1.0/ndiv;
		for (int j=1; j<ndiv; j++){
		    double ratio = frac*j;
		    Point2d new_p = (1.0-ratio)*prev_p+ratio*(*pt);
		    tmp_path.insert(pt,new_p);
		    if (old_i<=ie0) nadd_ie++;
		    if (old_i<=ib0) nadd_ib++;
		    if (old_i<=is0) nadd_is++;
		}
	    }
	    prev_p = *pt++; old_i++;
	}
	ie0 += nadd_ie; ie1 += nadd_ie;
	ib0 += nadd_ib; ib1 += nadd_ib;
	is0 += nadd_is; is1 += nadd_is;
    }
    list_to_path(tmp_path, path);
}

void GraphSolver2d::solve_recv_peg_refl(const Interface2d& seafloor, const Interface2d& surface,
				       const Interface2d& moho)
{
    if (!is_water_solved)
	error("GraphSolver2d::solve_recv_peg_refl - call solve_water first");
    if (!is_water_mult_solved)
	error("GraphSolver2d::solve_recv_peg_refl - call solve_water_mult first");
    const double ttime_inf = 1e30;
    water_seafloor = &seafloor;
    water_surface = &surface;
    water_moho = &moho;

    Array1d<int> botnodes, mohonodes;
    smesh.nearest(seafloor, botnodes);
    smesh.nearest(moho, mohonodes);

    B_rf_down.resize(0);
    C_w.resize(0);
    for (int i=1; i<=nnodes; i++){
	Point2d p = smesh.nodePos(i);
	if (p.x()<xmin || p.x()>xmax) continue;
	bool is_moho=false;
	for (int n=1; n<=mohonodes.size(); n++){
	    if (mohonodes(n)==i){ is_moho=true; break; }
	}
	double zsf = seafloor.z(p.x());
	double zmh = moho.z(p.x());
	bool in_layer = (p.y() > zsf+1e-6 && p.y() <= zmh+1e-6) || is_moho;
	if (!in_layer) continue;
	ttime_rf_down(i) = ttime_inf;
	prev_rf_down(i) = 0;
	C_w.push_back(i);
    }
    for (int n=1; n<=botnodes.size(); n++){
	int ib = botnodes(n);
	if (ttime_w1(ib) >= ttime_inf/2.0) continue;
	Index2d id = smesh.nodeIndex(ib);
	int k = id.k();
	if (k >= smesh.Nz()) continue;
	int ic = smesh.nodeIndex(id.i(), k+1);
	if (ic < 1 || ic > nnodes) continue;
	bool in_c = false;
	for (nodeIterator pC=C_w.begin(); pC!=C_w.end(); ){
	    if (*pC==ic){
		in_c = true;
		nodeIterator ep=pC;
		pC++;
		C_w.erase(ep);
	    }else{
		pC++;
	    }
	}
	if (!in_c) continue;
	double tseed = ttime_w1(ib)+smesh.calc_ttime(ib, ic);
	if (tseed >= ttime_rf_down(ic)) continue;
	ttime_rf_down(ic) = tseed;
	prev_rf_down(ic) = ib;
	B_rf_down.push(ic);
    }
    dijkstra_region(B_rf_down, prev_rf_down, ttime_rf_down, C_w);

    B_sp.resize(0);
    C_w.resize(0);
    for (int i=1; i<=nnodes; i++){
	Point2d p = smesh.nodePos(i);
	if (p.x()<xmin || p.x()>xmax) continue;
	bool is_moho=false;
	for (int n=1; n<=mohonodes.size(); n++){
	    if (mohonodes(n)==i){ is_moho=true; break; }
	}
	if (is_moho){
	    if (ttime_rf_down(i) >= ttime_inf/2.0) continue;
	    prev_sp(i) = -1;
	    ttime_sp(i) = ttime_rf_down(i);
	    B_sp.push(i);
	}else if (p.y() <= moho.z(p.x())+1e-6){
	    ttime_sp(i) = ttime_inf;
	    prev_sp(i) = 0;
	    C_w.push_back(i);
	}
    }
    dijkstra_region(B_sp, prev_sp, ttime_sp, C_w);
    is_recv_peg_refl_solved = true;
}

void GraphSolver2d::pickRecvPegReflPath(const Point2d& rcv, Array1d<Point2d>& path,
					int& is0, int& is1, int& ib0, int& ib1,
					int& ir0, int& ir1) const
{
    if (!is_recv_peg_refl_solved)
	error("GraphSolver2d::pickRecvPegReflPath - not solved");
    if (!water_seafloor || !water_surface || !water_moho)
	error("GraphSolver2d::pickRecvPegReflPath - interfaces missing");

    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);
    int cur = smesh.nearest(rcv);
    int prev;
    int nadd=0;
    while ((prev=prev_sp(cur)) != -1){
	if (prev==0) error("GraphSolver2d::pickRecvPegReflPath - broken sp chain");
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
    }
    double mx = smesh.nodePos(cur).x();
    Point2d onM(mx, water_moho->z(mx));
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(onM+rcv));
    tmp_path.push_back(onM); ir0 = (int)tmp_path.size();
    tmp_path.push_back(onM);
    tmp_path.push_back(onM); ir1 = (int)tmp_path.size();

    nadd=0;
    while (true){
	prev = prev_rf_down(cur);
	if (prev<=0) break;
	if (!node_in_crust(prev, *water_seafloor)) break;
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
    }
    if (prev==0) error("GraphSolver2d::pickRecvPegReflPath - broken rf_down chain");
    double bx = smesh.nodePos(cur).x();
    Point2d onB(bx, water_seafloor->z(bx));
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(onB+onM));
    tmp_path.push_back(onB); ib0 = (int)tmp_path.size();
    tmp_path.push_back(onB);
    tmp_path.push_back(onB); ib1 = (int)tmp_path.size();

    int ibot = (prev>0) ? prev : smesh.nearest(onB);
    nadd=0;
    cur = ibot;
    while ((prev=prev_w1(cur)) != -1){
	if (prev==0) error("GraphSolver2d::pickRecvPegReflPath - broken w1 chain");
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
    }
    double sx = smesh.nodePos(cur).x();
    Point2d onS(sx, water_surface->z(sx));
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(onS+onB));
    tmp_path.push_back(onS); is0 = (int)tmp_path.size();
    tmp_path.push_back(onS);
    tmp_path.push_back(onS); is1 = (int)tmp_path.size();

    nadd=0;
    while ((prev=prev_w0(cur)) != cur){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
    }
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(src+onS));
    tmp_path.push_back(src);

    if (refine_if_long){
	int old_i=1, nadd_ir=0, nadd_ib=0, nadd_is=0;
	list<Point2d>::iterator pt=tmp_path.begin();
	Point2d prev_p = *pt;
	pt++; old_i++;
	while(pt!=tmp_path.end()){
	    double seg_len = prev_p.distance(*pt);
	    if (seg_len>crit_len){
		int ndiv = int(seg_len/crit_len+1.0);
		double frac = 1.0/ndiv;
		for (int j=1; j<ndiv; j++){
		    double ratio = frac*j;
		    Point2d new_p = (1.0-ratio)*prev_p+ratio*(*pt);
		    tmp_path.insert(pt,new_p);
		    if (old_i<=ir0) nadd_ir++;
		    if (old_i<=ib0) nadd_ib++;
		    if (old_i<=is0) nadd_is++;
		}
	    }
	    prev_p = *pt++; old_i++;
	}
	ir0 += nadd_ir; ir1 += nadd_ir;
	ib0 += nadd_ib; ib1 += nadd_ib;
	is0 += nadd_is; is1 += nadd_is;
    }
    list_to_path(tmp_path, path);
}

bool GraphSolver2d::node_on_or_above_conv(int inode, const Interface2d& conv) const
{
    Point2d p = smesh.nodePos(inode);
    return p.y() <= conv.z(p.x()) + 1e-6;
}

bool GraphSolver2d::node_strictly_below_conv(int inode, const Interface2d& conv) const
{
    Point2d p = smesh.nodePos(inode);
    return p.y() > conv.z(p.x()) + 1e-6;
}

bool GraphSolver2d::node_in_lid(int inode, const Interface2d& seafloor,
			       const Interface2d& conv) const
{
    Point2d p = smesh.nodePos(inode);
    if (p.x()<xmin || p.x()>xmax) return false;
    double zsf = seafloor.z(p.x());
    double zc = conv.z(p.x());
    return p.y() >= zsf - 1e-6 && p.y() <= zc + 1e-6;
}

bool GraphSolver2d::node_in_psm_slab(int inode, const Interface2d& conv,
				    const Interface2d& moho) const
{
    Point2d p = smesh.nodePos(inode);
    if (p.x()<xmin || p.x()>xmax) return false;
    double zc = conv.z(p.x());
    double zm = moho.z(p.x());
    return p.y() > zc + 1e-6 && p.y() <= zm + 1e-6;
}

double GraphSolver2d::edge_tt(int a, int b) const
{
    if (edge_psx_leg != 0 && psp_convp){
	const bool want_s = (edge_psx_leg == 2 || edge_psx_leg == 3);
	double t = smesh.calc_ttime_psx(a, b, *psp_convp, edge_psx_leg);
	if (want_s) t *= ttime_scale;
	return t;
    }
    if (edge_use_vp && smesh.hasDualVs()){
	if (edge_vp_mixed_conv && psp_convp)
	    return smesh.calc_ttime_converse_dual(a, b, *psp_convp);
	return smesh.calc_ttime_vp(a, b);
    }
    return smesh.calc_ttime(a, b)*ttime_scale;
}

void GraphSolver2d::solve_pps(const Point2d& s, const Interface2d& conv,
			      const Interface2d& seafloor, double kappa)
{
    solve_psx(s, conv, seafloor, kappa, false);
}

void GraphSolver2d::solve_pss(const Point2d& s, const Interface2d& conv,
			      const Interface2d& seafloor, double kappa)
{
    solve_psp(s, conv, kappa, &seafloor, true);
}

void GraphSolver2d::solve_psx(const Point2d& s, const Interface2d& conv,
			     const Interface2d& seafloor, double kappa,
			     bool below_is_s)
{
    const double ttime_inf = 1e30;
    if (kappa<=0.0) error("GraphSolver2d::solve_psx - kappa must be positive.");
    src = s;
    psp_convp = &conv;
    is_psp_solved = false;
    is_psp_moho_solved = false;
    is_psp_peg_solved = false;
    is_psp_ss_solved = false;
    psx_two_leg = true;
    ttime_scale = 1.0;
    edge_use_vp = false;
    edge_vp_mixed_conv = false;
    edge_psx_leg = 0;

    Array1d<int> itcnodes;
    smesh.nearest(conv, itcnodes);
    std::vector<char> is_itc(nnodes + 1, 0);
    for (int n=1; n<=itcnodes.size(); n++)
	is_itc[itcnodes(n)] = 1;

    int Nsrc = smesh.nearest(src);
    if (!node_in_lid(Nsrc, seafloor, conv) && !is_itc[Nsrc])
	error("GraphSolver2d::solve_psx - source must sit in the lid (seafloor to conv).");

    for (int i=1; i<=nnodes; i++){
	ttime_psp_a(i) = ttime_inf;
	ttime_psp_b(i) = ttime_inf;
	ttime_psp_c(i) = ttime_inf;
	prev_psp_a(i) = 0;
	prev_psp_b(i) = 0;
	prev_psp_c(i) = 0;
    }

    // Two legs, one star. PPS: S(OBS→C in lid) + P(C→shot through lid+below+water).
    // Mixed lid S is already Vs = Vp/kappa inside calc_ttime_psx.
    ttime_scale = 1.0;
    edge_use_vp = false;
    edge_psx_leg = 3;
    B_psp_a.resize(0);
    prev_psp_a(Nsrc) = Nsrc;
    ttime_psp_a(Nsrc) = 0.0;
    B_psp_a.push(Nsrc);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	if (i==Nsrc) continue;
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	bool in_s = node_in_lid(i, seafloor, conv) || is_itc[i];
	if (below_is_s && node_strictly_below_conv(i, conv)) in_s = true;
	if (in_s){
	    C.push_back(i);
	    prev_psp_a(i) = Nsrc;
	}
    }
    dijkstra_region(B_psp_a, prev_psp_a, ttime_psp_a, C);

    psp_istar = 0;
    double tstar = ttime_inf;
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	double x = smesh.nodePos(itc).x();
	if (x<xmin || x>xmax) continue;
	if (ttime_psp_a(itc) < tstar){
	    tstar = ttime_psp_a(itc);
	    psp_istar = itc;
	}
    }
    if (psp_istar==0 || tstar>=ttime_inf)
	error("GraphSolver2d::solve_psx - OBS cannot reach the conversion interface as S.");

    ttime_scale = 1.0;
    edge_use_vp = false;
    edge_psx_leg = 4;
    B_psp_c.resize(0);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	bool in_p = node_on_or_above_conv(i, conv) || is_itc[i];
	if (!below_is_s && node_strictly_below_conv(i, conv)) in_p = true;
	if (!in_p) continue;
	if (is_itc[i]){
	    if (ttime_psp_a(i) >= ttime_inf) continue;
	    prev_psp_c(i) = -1;
	    ttime_psp_c(i) = ttime_psp_a(i);
	    B_psp_c.push(i);
	}else{
	    ttime_psp_c(i) = ttime_inf;
	    prev_psp_c(i) = 0;
	    C.push_back(i);
	}
    }
    dijkstra_region(B_psp_c, prev_psp_c, ttime_psp_c, C);
    ttime_scale = 1.0;
    edge_use_vp = false;
    edge_psx_leg = 0;
    is_psp_solved = true;
}

void GraphSolver2d::solve_psp(const Point2d& s, const Interface2d& conv,
			      double kappa, const Interface2d* seafloor,
			      bool lid_a_is_s)
{
    const double ttime_inf = 1e30;
    if (kappa<=0.0) kappa = 1.0;
    src = s;
    psp_convp = &conv;
    is_psp_solved = false;
    is_psp_moho_solved = false;
    is_psp_peg_solved = false;
    is_psp_ss_solved = false;
    psx_two_leg = false;
    ttime_scale = 1.0;
    edge_use_vp = false;
    edge_vp_mixed_conv = false;
    edge_psx_leg = 0;
    const double k_s = 1.0;
    const double k_below = 1.0;

    Array1d<int> itcnodes;
    smesh.nearest(conv, itcnodes);
    std::vector<char> is_itc(nnodes + 1, 0);
    for (int n=1; n<=itcnodes.size(); n++)
	is_itc[itcnodes(n)] = 1;

    int Nsrc = smesh.nearest(src);
    if (lid_a_is_s){
	if (seafloor==0)
	    error("GraphSolver2d::solve_psp - PSS requires seafloor.");
	if (!node_in_lid(Nsrc, *seafloor, conv) && !is_itc[Nsrc])
	    error("GraphSolver2d::solve_psp - PSS source must sit in the lid.");
    }else if (!node_on_or_above_conv(Nsrc, conv) && !is_itc[Nsrc]){
	error("GraphSolver2d::solve_psp - source is below the conversion interface.");
    }

    // Windowed Dijkstra must not see leftover 0s on nodes outside
    // [xmin,xmax] — those used to steal psp_istar to the mesh edge
    // (OBS in the middle of the line → cannot inject S).
    for (int i=1; i<=nnodes; i++){
	ttime_psp_a(i) = ttime_inf;
	ttime_psp_b(i) = ttime_inf;
	ttime_psp_c(i) = ttime_inf;
	prev_psp_a(i) = 0;
	prev_psp_b(i) = 0;
	prev_psp_c(i) = 0;
    }

    // Stage A: PSP = P above; PSS = S in the lid only. Interface nodes are
    // terminals — they must not become a headwave that starves the below S leg.
    if (lid_a_is_s){
	ttime_scale = k_s;
	edge_use_vp = false;
	edge_vp_mixed_conv = false;
	edge_psx_leg = 3;
    }else{
	ttime_scale = 1.0;
	edge_use_vp = false;
	edge_vp_mixed_conv = false;
	edge_psx_leg = 1;
    }
    B_psp_a.resize(0);
    prev_psp_a(Nsrc) = Nsrc;
    ttime_psp_a(Nsrc) = 0.0;
    B_psp_a.push(Nsrc);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	if (i==Nsrc) continue;
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	if (is_itc[i]) continue;
	bool in_a = lid_a_is_s
	    ? node_in_lid(i, *seafloor, conv)
	    : node_on_or_above_conv(i, conv);
	if (in_a){
	    C.push_back(i);
	    prev_psp_a(i) = Nsrc;
	}
    }
    dijkstra_region(B_psp_a, prev_psp_a, ttime_psp_a, C);
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	double x = smesh.nodePos(itc).x();
	if (x<xmin || x>xmax) continue;
	double best = ttime_inf;
	int bestp = -1;
	ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	for (int j=1; j<=nnodes; j++){
	    if (is_itc[j]) continue;
	    bool in_a = lid_a_is_s
		? node_in_lid(j, *seafloor, conv)
		: node_on_or_above_conv(j, conv);
	    if (!in_a) continue;
	    if (ttime_psp_a(j) >= ttime_inf) continue;
	    if (!fstar.isIn(smesh.nodeIndex(j))) continue;
	    double cand = ttime_psp_a(j)+edge_tt(j,itc);
	    if (cand < best){
		best = cand;
		bestp = j;
	    }
	}
	if (bestp != -1){
	    ttime_psp_a(itc) = best;
	    prev_psp_a(itc) = bestp;
	}
    }

    // Reachability only: conversion points (B,C) are chosen later by the
    // joint min of t_P(src→B)+t_S(B→C)+t_P(C→rcv). Snell is not imposed.
    psp_istar = 0;
    double tstar = ttime_inf;
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	double x = smesh.nodePos(itc).x();
	if (x<xmin || x>xmax) continue;
	if (ttime_psp_a(itc) < tstar){
	    tstar = ttime_psp_a(itc);
	    psp_istar = itc;
	}
    }
    if (psp_istar==0 || tstar>=ttime_inf)
	error("GraphSolver2d::solve_psp - source cannot reach the conversion interface.");

    ttime_scale = k_below;
    edge_use_vp = false;
    edge_vp_mixed_conv = false;
    edge_psx_leg = 2;
    B_psp_b.resize(0);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	if (is_itc[i]){
	    prev_psp_b(i) = -1;
	    ttime_psp_b(i) = ttime_inf;
	}else if (node_strictly_below_conv(i, conv)){
	    ttime_psp_b(i) = ttime_inf;
	    prev_psp_b(i) = 0;
	    C.push_back(i);
	}
    }
    {
	for (int n=1; n<=itcnodes.size(); n++){
	    int itc = itcnodes(n);
	    if (itc<1 || itc>nnodes) continue;
	    double x = smesh.nodePos(itc).x();
	    if (x<xmin || x>xmax) continue;
	    if (ttime_psp_a(itc) >= ttime_inf) continue;
	    ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	    for (nodeIterator pC=C.begin(); pC!=C.end(); ++pC){
		int j=*pC;
		if (!fstar.isIn(smesh.nodeIndex(j))) continue;
		double cand = ttime_psp_a(itc)+edge_tt(itc,j);
		if (cand < ttime_psp_b(j)){
		    ttime_psp_b(j) = cand;
		    prev_psp_b(j) = itc;
		}
	    }
	}
    }
    {
	nodeIterator pC=C.begin();
	nodeIterator epC;
	while(pC!=C.end()){
	    if (ttime_psp_b(*pC) < ttime_inf){
		B_psp_b.push(*pC);
		epC = pC;
		pC++;
		C.erase(epC);
	    }else{
		pC++;
	    }
	}
    }
    if (B_psp_b.size()==0)
	error("GraphSolver2d::solve_psp - cannot inject S below the conversion interface.");
    dijkstra_region(B_psp_b, prev_psp_b, ttime_psp_b, C);

    int n_returned = 0;
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	double best = ttime_inf;
	int bestp = -1;
	ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	for (int j=1; j<=nnodes; j++){
	    if (!node_strictly_below_conv(j, conv)) continue;
	    if (ttime_psp_b(j) >= ttime_inf) continue;
	    if (!fstar.isIn(smesh.nodeIndex(j))) continue;
	    double cand = ttime_psp_b(j)+edge_tt(j,itc);
	    if (cand < best){
		best = cand;
		bestp = j;
	    }
	}
	if (bestp != -1){
	    ttime_psp_b(itc) = best;
	    prev_psp_b(itc) = bestp;
	    n_returned++;
	}
    }
    if (n_returned==0)
	error("GraphSolver2d::solve_psp - no converted path returned to the interface from below.");

    // Upgoing P: interface nodes carry t_P(src→B)+t_S(B→C), so the receiver
    // attaches to the C that minimises the full PSP time (not nearest-P).
    ttime_scale = 1.0;
    edge_use_vp = false;
    edge_vp_mixed_conv = false;
    edge_psx_leg = 1;
    B_psp_c.resize(0);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	if (is_itc[i]){
	    if (ttime_psp_b(i) >= ttime_inf) continue;
	    prev_psp_c(i) = -1;
	    ttime_psp_c(i) = ttime_psp_b(i);
	}else if (node_on_or_above_conv(i, conv)){
	    ttime_psp_c(i) = ttime_inf;
	    prev_psp_c(i) = 0;
	    C.push_back(i);
	}
    }
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	if (ttime_psp_b(itc) >= ttime_inf) continue;
	ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	for (nodeIterator pC=C.begin(); pC!=C.end(); ++pC){
	    int j=*pC;
	    if (!fstar.isIn(smesh.nodeIndex(j))) continue;
	    double cand = ttime_psp_b(itc)+edge_tt(itc,j);
	    if (cand < ttime_psp_c(j)){
		ttime_psp_c(j) = cand;
		prev_psp_c(j) = itc;
	    }
	}
    }
    {
	nodeIterator pC=C.begin();
	nodeIterator epC;
	while(pC!=C.end()){
	    if (ttime_psp_c(*pC) < ttime_inf){
		B_psp_c.push(*pC);
		epC = pC;
		pC++;
		C.erase(epC);
	    }else{
		pC++;
	    }
	}
    }
    if (B_psp_c.size()==0)
	error("GraphSolver2d::solve_psp - cannot inject P above the conversion interface.");
    dijkstra_region(B_psp_c, prev_psp_c, ttime_psp_c, C);
    edge_use_vp = false;
    edge_vp_mixed_conv = false;
    edge_psx_leg = 0;
    is_psp_solved = true;
}

static void psp_refine_if_long(list<Point2d>& tmp_path, double crit_len,
			       int* pins, int npins)
{
    int old_i=1;
    std::vector<int> nadd(npins, 0);
    list<Point2d>::iterator pt=tmp_path.begin();
    Point2d prev_p = *pt;
    pt++; old_i++;
    while(pt!=tmp_path.end()){
	double seg_len = prev_p.distance(*pt);
	if (seg_len>crit_len){
	    int ndiv = int(seg_len/crit_len+1.0);
	    double frac = 1.0/ndiv;
	    for (int j=1; j<ndiv; j++){
		double ratio = frac*j;
		Point2d new_p = (1.0-ratio)*prev_p+ratio*(*pt);
		tmp_path.insert(pt,new_p);
		for (int k=0; k<npins; k++){
		    if (old_i<=pins[k]) nadd[k]++;
		}
	    }
	}
	prev_p = *pt++; old_i++;
    }
    for (int k=0; k<npins; k++) pins[k] += nadd[k];
}

void GraphSolver2d::pickPspPath(const Point2d& rcv, Array1d<Point2d>& path,
				int& id0, int& id1, int& iu0, int& iu1) const
{
    if (!is_psp_solved) error("GraphSolver2d::pickPspPath - not solved");
    if (!psp_convp) error("GraphSolver2d::pickPspPath - conversion interface missing");
    const Interface2d& conv = *psp_convp;
    const double ttime_inf = 1e29;

    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);
    int cur = smesh.nearest(rcv);
    if (ttime_psp_c(cur) > ttime_inf)
	error("GraphSolver2d::pickPspPath - receiver not reached by converted path");

    int prev;
    int nadd=0;
    while ((prev=prev_psp_c(cur)) != -1){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspPath - cycle in stage C");
    }
    if (nadd>0) tmp_path.pop_back();
    Point2d on_up(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
    if (nadd==1) tmp_path.push_back(0.5*(on_up+rcv));
    tmp_path.push_back(on_up); iu0 = (int)tmp_path.size();
    tmp_path.push_back(on_up);
    tmp_path.push_back(on_up); iu1 = (int)tmp_path.size();

    Point2d on_dn = on_up;
    if (psx_two_leg){
	id0 = iu1;
	id1 = iu1;
    }else{
	nadd=0;
	prev = prev_psp_b(cur);
	while (prev>0 && prev!=cur && node_strictly_below_conv(prev, conv)){
	    tmp_path.push_back(smesh.nodePos(prev));
	    nadd++;
	    cur = prev;
	    prev = prev_psp_b(cur);
	    if (nadd>nnodes) error("GraphSolver2d::pickPspPath - cycle in stage B");
	}
	if (prev>0 && prev!=cur) cur = prev;
	if (nadd>1) tmp_path.pop_back();
	on_dn = Point2d(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
	if (nadd==1) tmp_path.push_back(0.5*(on_dn+on_up));
	tmp_path.push_back(on_dn); id0 = (int)tmp_path.size();
	tmp_path.push_back(on_dn);
	tmp_path.push_back(on_dn); id1 = (int)tmp_path.size();
    }

    nadd=0;
    while ((prev=prev_psp_a(cur)) != cur){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspPath - cycle in stage A");
    }
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(src+on_dn));
    tmp_path.push_back(src);

    if (refine_if_long){
	int pins[4] = {iu0, iu1, id0, id1};
	psp_refine_if_long(tmp_path, crit_len, pins, 4);
	iu0=pins[0]; iu1=pins[1]; id0=pins[2]; id1=pins[3];
    }
    list_to_path(tmp_path, path);
}

void GraphSolver2d::pickPspPathThruWater(const Point2d& rcv, Array1d<Point2d>& path,
					 int& i0, int& i1,
					 int& id0, int& id1, int& iu0, int& iu1) const
{
    if (!is_psp_solved) error("GraphSolver2d::pickPspPathThruWater - not solved");
    if (!psp_convp) error("GraphSolver2d::pickPspPathThruWater - conversion interface missing");
    const Interface2d& conv = *psp_convp;
    const double ttime_inf = 1e29;

    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);
    int cur = smesh.nearest(rcv);
    findRayEntryPoint(rcv, cur, ttime_psp_c);
    if (ttime_psp_c(cur) > ttime_inf)
	error("GraphSolver2d::pickPspPathThruWater - water entry not reached by converted path");

    Point2d entryp=smesh.nodePos(cur);
    tmp_path.push_back(entryp);
    tmp_path.push_back(entryp);
    tmp_path.push_back(entryp);
    i0 = 2; i1 = 4;

    int prev;
    int nadd=0;
    while ((prev=prev_psp_c(cur)) != -1){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspPathThruWater - cycle in stage C");
    }
    if (nadd>0) tmp_path.pop_back();
    Point2d on_up(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
    if (nadd==1) tmp_path.push_back(0.5*(on_up+entryp));
    tmp_path.push_back(on_up); iu0 = (int)tmp_path.size();
    tmp_path.push_back(on_up);
    tmp_path.push_back(on_up); iu1 = (int)tmp_path.size();

    Point2d on_dn = on_up;
    if (psx_two_leg){
	id0 = iu1;
	id1 = iu1;
    }else{
	nadd=0;
	prev = prev_psp_b(cur);
	while (prev>0 && prev!=cur && node_strictly_below_conv(prev, conv)){
	    tmp_path.push_back(smesh.nodePos(prev));
	    nadd++;
	    cur = prev;
	    prev = prev_psp_b(cur);
	    if (nadd>nnodes) error("GraphSolver2d::pickPspPathThruWater - cycle in stage B");
	}
	if (prev>0 && prev!=cur) cur = prev;
	if (nadd>1) tmp_path.pop_back();
	on_dn = Point2d(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
	if (nadd==1) tmp_path.push_back(0.5*(on_dn+on_up));
	tmp_path.push_back(on_dn); id0 = (int)tmp_path.size();
	tmp_path.push_back(on_dn);
	tmp_path.push_back(on_dn); id1 = (int)tmp_path.size();
    }

    nadd=0;
    while ((prev=prev_psp_a(cur)) != cur){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspPathThruWater - cycle in stage A");
    }
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(src+on_dn));
    tmp_path.push_back(src);

    if (refine_if_long){
	int pins[6] = {i0, i1, iu0, iu1, id0, id1};
	psp_refine_if_long(tmp_path, crit_len, pins, 6);
	i0=pins[0]; i1=pins[1]; iu0=pins[2]; iu1=pins[3]; id0=pins[4]; id1=pins[5];
    }
    list_to_path(tmp_path, path);
}

void GraphSolver2d::solve_pss_moho(const Point2d& s, const Interface2d& conv,
				   const Interface2d& moho, const Interface2d& seafloor,
				   double kappa)
{
    solve_psp_moho(s, conv, moho, kappa, &seafloor, true);
}

void GraphSolver2d::solve_psp_moho(const Point2d& s, const Interface2d& conv,
				   const Interface2d& moho, double kappa,
				   const Interface2d* seafloor, bool lid_a_is_s)
{
    const double ttime_inf = 1e30;
    if (kappa<=0.0) kappa = 1.0;
    src = s;
    psp_convp = &conv;
    psp_mohop = &moho;
    is_psp_solved = false;
    is_psp_moho_solved = false;
    is_psp_peg_solved = false;
    is_psp_ss_solved = false;
    psx_two_leg = false;
    ttime_scale = 1.0;
    edge_use_vp = false;
    edge_vp_mixed_conv = false;
    edge_psx_leg = 0;
    const double k_s = 1.0;
    const double k_below = 1.0;

    Array1d<int> itcnodes, itmnodes;
    smesh.nearest(conv, itcnodes);
    smesh.nearest(moho, itmnodes);
    std::vector<char> is_itc(nnodes + 1, 0);
    std::vector<char> is_itm(nnodes + 1, 0);
    for (int n=1; n<=itcnodes.size(); n++)
	is_itc[itcnodes(n)] = 1;
    for (int n=1; n<=itmnodes.size(); n++)
	is_itm[itmnodes(n)] = 1;

    int Nsrc = smesh.nearest(src);
    if (lid_a_is_s){
	if (seafloor==0)
	    error("GraphSolver2d::solve_psp_moho - PSS-Moho requires seafloor.");
	if (!node_in_lid(Nsrc, *seafloor, conv) && !is_itc[Nsrc])
	    error("GraphSolver2d::solve_psp_moho - PSS-Moho source must sit in the lid.");
    }else if (!node_on_or_above_conv(Nsrc, conv) && !is_itc[Nsrc]){
	error("GraphSolver2d::solve_psp_moho - source is below the conversion interface.");
    }

    for (int i=1; i<=nnodes; i++){
	ttime_psp_a(i) = ttime_inf;
	ttime_psp_b(i) = ttime_inf;
	ttime_psp_c(i) = ttime_inf;
	ttime_psm_u(i) = ttime_inf;
	prev_psp_a(i) = 0;
	prev_psp_b(i) = 0;
	prev_psp_c(i) = 0;
	prev_psm_u(i) = 0;
    }

    // Stage A: same lid as raytype 6/8 (P or S). Does not touch 0/1 arrays.
    if (lid_a_is_s){
	ttime_scale = k_s;
	edge_use_vp = false;
	edge_vp_mixed_conv = false;
	edge_psx_leg = 3;
    }else{
	ttime_scale = 1.0;
	edge_use_vp = false;
	edge_vp_mixed_conv = false;
	edge_psx_leg = 1;
    }
    B_psp_a.resize(0);
    prev_psp_a(Nsrc) = Nsrc;
    ttime_psp_a(Nsrc) = 0.0;
    B_psp_a.push(Nsrc);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	if (i==Nsrc) continue;
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	if (is_itc[i]) continue;
	bool in_a = lid_a_is_s
	    ? node_in_lid(i, *seafloor, conv)
	    : node_on_or_above_conv(i, conv);
	if (in_a){
	    C.push_back(i);
	    prev_psp_a(i) = Nsrc;
	}
    }
    dijkstra_region(B_psp_a, prev_psp_a, ttime_psp_a, C);
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	double x = smesh.nodePos(itc).x();
	if (x<xmin || x>xmax) continue;
	double best = ttime_inf;
	int bestp = -1;
	ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	for (int j=1; j<=nnodes; j++){
	    if (is_itc[j]) continue;
	    bool in_a = lid_a_is_s
		? node_in_lid(j, *seafloor, conv)
		: node_on_or_above_conv(j, conv);
	    if (!in_a) continue;
	    if (ttime_psp_a(j) >= ttime_inf) continue;
	    if (!fstar.isIn(smesh.nodeIndex(j))) continue;
	    double cand = ttime_psp_a(j)+edge_tt(j,itc);
	    if (cand < best){
		best = cand;
		bestp = j;
	    }
	}
	if (bestp != -1){
	    ttime_psp_a(itc) = best;
	    prev_psp_a(itc) = bestp;
	}
    }

    psp_istar = 0;
    double tstar = ttime_inf;
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	double x = smesh.nodePos(itc).x();
	if (x<xmin || x>xmax) continue;
	if (ttime_psp_a(itc) < tstar){
	    tstar = ttime_psp_a(itc);
	    psp_istar = itc;
	}
    }
    if (psp_istar==0 || tstar>=ttime_inf)
	error("GraphSolver2d::solve_psp_moho - source cannot reach the conversion interface.");

    // Stage B-down: S in conv–Moho slab only. prev_psp_b is independent of type 1.
    ttime_scale = k_below;
    edge_use_vp = false;
    edge_vp_mixed_conv = false;
    edge_psx_leg = 2;
    B_psp_b.resize(0);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	if (is_itc[i]){
	    prev_psp_b(i) = -1;
	    ttime_psp_b(i) = ttime_inf;
	    continue;
	}
	bool in_slab = node_in_psm_slab(i, conv, moho);
	if (!in_slab && is_itm[i] && node_strictly_below_conv(i, conv))
	    in_slab = true;
	if (in_slab){
	    ttime_psp_b(i) = ttime_inf;
	    prev_psp_b(i) = 0;
	    C.push_back(i);
	}
    }
    {
	for (int n=1; n<=itcnodes.size(); n++){
	    int itc = itcnodes(n);
	    if (itc<1 || itc>nnodes) continue;
	    double x = smesh.nodePos(itc).x();
	    if (x<xmin || x>xmax) continue;
	    if (ttime_psp_a(itc) >= ttime_inf) continue;
	    ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	    for (nodeIterator pC=C.begin(); pC!=C.end(); ++pC){
		int j=*pC;
		if (!fstar.isIn(smesh.nodeIndex(j))) continue;
		double cand = ttime_psp_a(itc)+edge_tt(itc,j);
		if (cand < ttime_psp_b(j)){
		    ttime_psp_b(j) = cand;
		    prev_psp_b(j) = itc;
		}
	    }
	}
    }
    {
	nodeIterator pC=C.begin();
	nodeIterator epC;
	while(pC!=C.end()){
	    if (ttime_psp_b(*pC) < ttime_inf){
		B_psp_b.push(*pC);
		epC = pC;
		pC++;
		C.erase(epC);
	    }else{
		pC++;
	    }
	}
    }
    if (B_psp_b.size()==0)
	error("GraphSolver2d::solve_psp_moho - cannot inject S below the conversion interface.");
    dijkstra_region(B_psp_b, prev_psp_b, ttime_psp_b, C);

    int n_moho = 0;
    for (int n=1; n<=itmnodes.size(); n++){
	int itm = itmnodes(n);
	if (itm<1 || itm>nnodes) continue;
	double x = smesh.nodePos(itm).x();
	if (x<xmin || x>xmax) continue;
	if (ttime_psp_b(itm) < ttime_inf){
	    n_moho++;
	    continue;
	}
	double best = ttime_inf;
	int bestp = -1;
	ForwardStar2d fstar(smesh.nodeIndex(itm), fs_xorder, fs_zorder);
	for (int j=1; j<=nnodes; j++){
	    if (!node_in_psm_slab(j, conv, moho)) continue;
	    if (ttime_psp_b(j) >= ttime_inf) continue;
	    if (!fstar.isIn(smesh.nodeIndex(j))) continue;
	    double cand = ttime_psp_b(j)+edge_tt(j,itm);
	    if (cand < best){
		best = cand;
		bestp = j;
	    }
	}
	if (bestp != -1){
	    ttime_psp_b(itm) = best;
	    prev_psp_b(itm) = bestp;
	    n_moho++;
	}
    }
    if (n_moho==0)
	error("GraphSolver2d::solve_psp_moho - S cannot reach the Moho.");

    // Stage B-up: S from Moho back to conv. prev_psm_u is independent of type 1.
    B_psm_u.resize(0);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	if (is_itm[i]){
	    if (ttime_psp_b(i) >= ttime_inf) continue;
	    prev_psm_u(i) = -1;
	    ttime_psm_u(i) = ttime_psp_b(i);
	    B_psm_u.push(i);
	}else if (is_itc[i]){
	    prev_psm_u(i) = 0;
	    ttime_psm_u(i) = ttime_inf;
	}else if (node_in_psm_slab(i, conv, moho)){
	    ttime_psm_u(i) = ttime_inf;
	    prev_psm_u(i) = 0;
	    C.push_back(i);
	}
    }
    if (B_psm_u.size()==0)
	error("GraphSolver2d::solve_psp_moho - no Moho nodes to start the upgoing S.");
    dijkstra_region(B_psm_u, prev_psm_u, ttime_psm_u, C);

    int n_returned = 0;
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	double best = ttime_inf;
	int bestp = -1;
	ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	for (int j=1; j<=nnodes; j++){
	    if (!node_in_psm_slab(j, conv, moho) && !is_itm[j]) continue;
	    if (ttime_psm_u(j) >= ttime_inf) continue;
	    if (!fstar.isIn(smesh.nodeIndex(j))) continue;
	    double cand = ttime_psm_u(j)+edge_tt(j,itc);
	    if (cand < best){
		best = cand;
		bestp = j;
	    }
	}
	if (bestp != -1){
	    ttime_psm_u(itc) = best;
	    prev_psm_u(itc) = bestp;
	    n_returned++;
	}
    }
    if (n_returned==0)
	error("GraphSolver2d::solve_psp_moho - no reflected S path returned to the conversion interface.");

    // Stage C: P above conv, seeded by t_A + t_S(down) + t_S(up) at conv.
    ttime_scale = 1.0;
    edge_use_vp = false;
    edge_vp_mixed_conv = false;
    edge_psx_leg = 1;
    B_psp_c.resize(0);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	if (is_itc[i]){
	    if (ttime_psm_u(i) >= ttime_inf) continue;
	    prev_psp_c(i) = -1;
	    ttime_psp_c(i) = ttime_psm_u(i);
	}else if (node_on_or_above_conv(i, conv)){
	    ttime_psp_c(i) = ttime_inf;
	    prev_psp_c(i) = 0;
	    C.push_back(i);
	}
    }
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	if (ttime_psm_u(itc) >= ttime_inf) continue;
	ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	for (nodeIterator pC=C.begin(); pC!=C.end(); ++pC){
	    int j=*pC;
	    if (!fstar.isIn(smesh.nodeIndex(j))) continue;
	    double cand = ttime_psm_u(itc)+edge_tt(itc,j);
	    if (cand < ttime_psp_c(j)){
		ttime_psp_c(j) = cand;
		prev_psp_c(j) = itc;
	    }
	}
    }
    {
	nodeIterator pC=C.begin();
	nodeIterator epC;
	while(pC!=C.end()){
	    if (ttime_psp_c(*pC) < ttime_inf){
		B_psp_c.push(*pC);
		epC = pC;
		pC++;
		C.erase(epC);
	    }else{
		pC++;
	    }
	}
    }
    if (B_psp_c.size()==0)
	error("GraphSolver2d::solve_psp_moho - cannot inject P above the conversion interface.");
    dijkstra_region(B_psp_c, prev_psp_c, ttime_psp_c, C);
    edge_use_vp = false;
    edge_vp_mixed_conv = false;
    edge_psx_leg = 0;
    is_psp_moho_solved = true;
}

void GraphSolver2d::pickPspMohoPath(const Point2d& rcv, Array1d<Point2d>& path,
				    int& id0, int& id1, int& ir0, int& ir1,
				    int& iu0, int& iu1) const
{
    if (!is_psp_moho_solved) error("GraphSolver2d::pickPspMohoPath - not solved");
    if (!psp_convp || !psp_mohop)
	error("GraphSolver2d::pickPspMohoPath - conversion or Moho interface missing");
    const Interface2d& conv = *psp_convp;
    const Interface2d& moho = *psp_mohop;
    const double ttime_inf = 1e29;

    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);
    int cur = smesh.nearest(rcv);
    if (ttime_psp_c(cur) > ttime_inf)
	error("GraphSolver2d::pickPspMohoPath - receiver not reached by reflected converted path");

    int prev;
    int nadd=0;
    while ((prev=prev_psp_c(cur)) != -1){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspMohoPath - cycle in stage C");
    }
    if (nadd>0) tmp_path.pop_back();
    Point2d on_up(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
    if (nadd==1) tmp_path.push_back(0.5*(on_up+rcv));
    tmp_path.push_back(on_up); iu0 = (int)tmp_path.size();
    tmp_path.push_back(on_up);
    tmp_path.push_back(on_up); iu1 = (int)tmp_path.size();

    nadd=0;
    prev = prev_psm_u(cur);
    if (prev==0) error("GraphSolver2d::pickPspMohoPath - broken S-up chain at conversion");
    while (prev != -1){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	prev = prev_psm_u(cur);
	if (nadd>nnodes) error("GraphSolver2d::pickPspMohoPath - cycle in S-up");
    }
    if (nadd>0) tmp_path.pop_back();
    Point2d on_moho(smesh.nodePos(cur).x(), moho.z(smesh.nodePos(cur).x()));
    if (nadd==1) tmp_path.push_back(0.5*(on_up+on_moho));
    tmp_path.push_back(on_moho); ir0 = (int)tmp_path.size();
    tmp_path.push_back(on_moho);
    tmp_path.push_back(on_moho); ir1 = (int)tmp_path.size();

    nadd=0;
    prev = prev_psp_b(cur);
    while (prev>0 && prev!=cur &&
	   (node_in_psm_slab(prev, conv, moho) ||
	    smesh.nodePos(prev).y() > conv.z(smesh.nodePos(prev).x())+1e-6)){
	if (prev_psp_b(prev)==-1) break;
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	prev = prev_psp_b(cur);
	if (nadd>nnodes) error("GraphSolver2d::pickPspMohoPath - cycle in S-down");
    }
    if (prev>0 && prev!=cur) cur = prev;
    if (nadd>1) tmp_path.pop_back();
    Point2d on_dn(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
    if (nadd==1) tmp_path.push_back(0.5*(on_moho+on_dn));
    tmp_path.push_back(on_dn); id0 = (int)tmp_path.size();
    tmp_path.push_back(on_dn);
    tmp_path.push_back(on_dn); id1 = (int)tmp_path.size();

    nadd=0;
    while ((prev=prev_psp_a(cur)) != cur){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspMohoPath - cycle in stage A");
    }
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(src+on_dn));
    tmp_path.push_back(src);

    if (refine_if_long){
	int pins[6] = {iu0, iu1, ir0, ir1, id0, id1};
	psp_refine_if_long(tmp_path, crit_len, pins, 6);
	iu0=pins[0]; iu1=pins[1]; ir0=pins[2]; ir1=pins[3]; id0=pins[4]; id1=pins[5];
    }
    list_to_path(tmp_path, path);
}

void GraphSolver2d::pickPspMohoPathThruWater(const Point2d& rcv, Array1d<Point2d>& path,
					     int& i0, int& i1, int& id0, int& id1,
					     int& ir0, int& ir1, int& iu0, int& iu1) const
{
    if (!is_psp_moho_solved) error("GraphSolver2d::pickPspMohoPathThruWater - not solved");
    if (!psp_convp || !psp_mohop)
	error("GraphSolver2d::pickPspMohoPathThruWater - conversion or Moho interface missing");
    const Interface2d& conv = *psp_convp;
    const Interface2d& moho = *psp_mohop;
    const double ttime_inf = 1e29;

    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);
    int cur = smesh.nearest(rcv);
    findRayEntryPoint(rcv, cur, ttime_psp_c);
    if (ttime_psp_c(cur) > ttime_inf)
	error("GraphSolver2d::pickPspMohoPathThruWater - water entry not reached");

    Point2d entryp=smesh.nodePos(cur);
    tmp_path.push_back(entryp);
    tmp_path.push_back(entryp);
    tmp_path.push_back(entryp);
    i0 = 2; i1 = 4;

    int prev;
    int nadd=0;
    while ((prev=prev_psp_c(cur)) != -1){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspMohoPathThruWater - cycle in stage C");
    }
    if (nadd>0) tmp_path.pop_back();
    Point2d on_up(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
    if (nadd==1) tmp_path.push_back(0.5*(on_up+entryp));
    tmp_path.push_back(on_up); iu0 = (int)tmp_path.size();
    tmp_path.push_back(on_up);
    tmp_path.push_back(on_up); iu1 = (int)tmp_path.size();

    nadd=0;
    prev = prev_psm_u(cur);
    if (prev==0) error("GraphSolver2d::pickPspMohoPathThruWater - broken S-up chain at conversion");
    while (prev != -1){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	prev = prev_psm_u(cur);
	if (nadd>nnodes) error("GraphSolver2d::pickPspMohoPathThruWater - cycle in S-up");
    }
    if (nadd>0) tmp_path.pop_back();
    Point2d on_moho(smesh.nodePos(cur).x(), moho.z(smesh.nodePos(cur).x()));
    if (nadd==1) tmp_path.push_back(0.5*(on_up+on_moho));
    tmp_path.push_back(on_moho); ir0 = (int)tmp_path.size();
    tmp_path.push_back(on_moho);
    tmp_path.push_back(on_moho); ir1 = (int)tmp_path.size();

    nadd=0;
    prev = prev_psp_b(cur);
    while (prev>0 && prev!=cur &&
	   (node_in_psm_slab(prev, conv, moho) ||
	    smesh.nodePos(prev).y() > conv.z(smesh.nodePos(prev).x())+1e-6)){
	if (prev_psp_b(prev)==-1) break;
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	prev = prev_psp_b(cur);
	if (nadd>nnodes) error("GraphSolver2d::pickPspMohoPathThruWater - cycle in S-down");
    }
    if (prev>0 && prev!=cur) cur = prev;
    if (nadd>1) tmp_path.pop_back();
    Point2d on_dn(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
    if (nadd==1) tmp_path.push_back(0.5*(on_moho+on_dn));
    tmp_path.push_back(on_dn); id0 = (int)tmp_path.size();
    tmp_path.push_back(on_dn);
    tmp_path.push_back(on_dn); id1 = (int)tmp_path.size();

    nadd=0;
    while ((prev=prev_psp_a(cur)) != cur){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspMohoPathThruWater - cycle in stage A");
    }
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(src+on_dn));
    tmp_path.push_back(src);

    if (refine_if_long){
	int pins[8] = {i0, i1, iu0, iu1, ir0, ir1, id0, id1};
	psp_refine_if_long(tmp_path, crit_len, pins, 8);
	i0=pins[0]; i1=pins[1]; iu0=pins[2]; iu1=pins[3];
	ir0=pins[4]; ir1=pins[5]; id0=pins[6]; id1=pins[7];
    }
    list_to_path(tmp_path, path);
}

void GraphSolver2d::solve_pps_peg(const Point2d& s, const Interface2d& conv,
				  const Interface2d& seafloor, double kappa)
{
    const double ttime_inf = 1e30;
    if (!is_water_solved)
	error("GraphSolver2d::solve_pps_peg - call solve_water first");
    if (!is_water_mult_solved)
	error("GraphSolver2d::solve_pps_peg - call solve_water_mult first");
    if (kappa<=0.0) error("GraphSolver2d::solve_pps_peg - kappa must be positive.");
    src = s;
    psp_convp = &conv;
    water_seafloor = &seafloor;
    is_psp_solved = false;
    is_psp_moho_solved = false;
    is_psp_peg_solved = false;
    is_psp_ss_solved = false;
    psx_two_leg = true;
    ttime_scale = 1.0;
    edge_use_vp = false;
    edge_vp_mixed_conv = false;
    edge_psx_leg = 0;

    Array1d<int> itcnodes, botnodes;
    smesh.nearest(conv, itcnodes);
    smesh.nearest(seafloor, botnodes);
    std::vector<char> is_itc(nnodes + 1, 0);
    std::vector<char> is_bot(nnodes + 1, 0);
    for (int n=1; n<=itcnodes.size(); n++)
	is_itc[itcnodes(n)] = 1;
    for (int n=1; n<=botnodes.size(); n++)
	is_bot[botnodes(n)] = 1;

    for (int i=1; i<=nnodes; i++){
	ttime_psp_a(i) = ttime_inf;
	ttime_psp_b(i) = ttime_inf;
	ttime_psp_c(i) = ttime_inf;
	prev_psp_a(i) = 0;
	prev_psp_b(i) = 0;
	prev_psp_c(i) = 0;
    }

    ttime_scale = 1.0;
    edge_psx_leg = 3;
    B_psp_a.resize(0);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	if (is_bot[i]) continue;
	bool in_s = node_in_lid(i, seafloor, conv) || is_itc[i];
	if (in_s){
	    C.push_back(i);
	    prev_psp_a(i) = 0;
	}
    }
    for (int n=1; n<=botnodes.size(); n++){
	int ib = botnodes(n);
	if (ib<1 || ib>nnodes) continue;
	if (ttime_w1(ib) >= ttime_inf/2.0) continue;
	Index2d id = smesh.nodeIndex(ib);
	if (id.k() >= smesh.Nz()) continue;
	int ic = smesh.nodeIndex(id.i(), id.k()+1);
	if (ic<1 || ic>nnodes) continue;
	if (!node_in_lid(ic, seafloor, conv) && !is_itc[ic]) continue;
	double tseed = ttime_w1(ib)+edge_tt(ib, ic);
	if (tseed >= ttime_psp_a(ic)) continue;
	bool in_c = false;
	for (nodeIterator pC=C.begin(); pC!=C.end(); ){
	    if (*pC==ic){
		in_c = true;
		nodeIterator ep=pC;
		pC++;
		C.erase(ep);
	    }else{
		pC++;
	    }
	}
	ttime_psp_a(ic) = tseed;
	prev_psp_a(ic) = ib;
	if (in_c) B_psp_a.push(ic);
    }
    if (B_psp_a.size()==0)
	error("GraphSolver2d::solve_pps_peg - cannot inject lid S from seafloor after water peg.");
    dijkstra_region(B_psp_a, prev_psp_a, ttime_psp_a, C);

    ttime_scale = 1.0;
    edge_psx_leg = 4;
    B_psp_c.resize(0);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	bool in_p = node_on_or_above_conv(i, conv) || is_itc[i]
	    || node_strictly_below_conv(i, conv);
	if (!in_p) continue;
	if (is_itc[i]){
	    if (ttime_psp_a(i) >= ttime_inf) continue;
	    prev_psp_c(i) = -1;
	    ttime_psp_c(i) = ttime_psp_a(i);
	    B_psp_c.push(i);
	}else{
	    ttime_psp_c(i) = ttime_inf;
	    prev_psp_c(i) = 0;
	    C.push_back(i);
	}
    }
    if (B_psp_c.size()==0)
	error("GraphSolver2d::solve_pps_peg - OBS peg cannot reach the conversion interface as S.");
    dijkstra_region(B_psp_c, prev_psp_c, ttime_psp_c, C);
    edge_psx_leg = 0;
    is_psp_peg_solved = true;
}

void GraphSolver2d::solve_pss_peg(const Point2d& s, const Interface2d& conv,
				  const Interface2d& seafloor, double kappa)
{
    solve_psp_peg(s, conv, seafloor, kappa, true);
}

void GraphSolver2d::solve_psp_peg(const Point2d& s, const Interface2d& conv,
				  const Interface2d& seafloor, double kappa,
				  bool lid_a_is_s)
{
    const double ttime_inf = 1e30;
    if (!is_water_solved)
	error("GraphSolver2d::solve_psp_peg - call solve_water first");
    if (!is_water_mult_solved)
	error("GraphSolver2d::solve_psp_peg - call solve_water_mult first");
    if (kappa<=0.0) kappa = 1.0;
    src = s;
    psp_convp = &conv;
    water_seafloor = &seafloor;
    is_psp_solved = false;
    is_psp_moho_solved = false;
    is_psp_peg_solved = false;
    is_psp_ss_solved = false;
    psx_two_leg = false;
    ttime_scale = 1.0;
    edge_use_vp = false;
    edge_vp_mixed_conv = false;
    edge_psx_leg = 0;

    Array1d<int> itcnodes, botnodes;
    smesh.nearest(conv, itcnodes);
    smesh.nearest(seafloor, botnodes);
    std::vector<char> is_itc(nnodes + 1, 0);
    std::vector<char> is_bot(nnodes + 1, 0);
    for (int n=1; n<=itcnodes.size(); n++)
	is_itc[itcnodes(n)] = 1;
    for (int n=1; n<=botnodes.size(); n++)
	is_bot[botnodes(n)] = 1;

    for (int i=1; i<=nnodes; i++){
	ttime_psp_a(i) = ttime_inf;
	ttime_psp_b(i) = ttime_inf;
	ttime_psp_c(i) = ttime_inf;
	prev_psp_a(i) = 0;
	prev_psp_b(i) = 0;
	prev_psp_c(i) = 0;
    }

    if (lid_a_is_s){
	ttime_scale = 1.0;
	edge_psx_leg = 3;
    }else{
	ttime_scale = 1.0;
	edge_psx_leg = 1;
    }
    B_psp_a.resize(0);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	if (is_itc[i] || is_bot[i]) continue;
	if (node_in_lid(i, seafloor, conv)){
	    C.push_back(i);
	    prev_psp_a(i) = 0;
	}
    }
    for (int n=1; n<=botnodes.size(); n++){
	int ib = botnodes(n);
	if (ib<1 || ib>nnodes) continue;
	if (ttime_w1(ib) >= ttime_inf/2.0) continue;
	Index2d id = smesh.nodeIndex(ib);
	if (id.k() >= smesh.Nz()) continue;
	int ic = smesh.nodeIndex(id.i(), id.k()+1);
	if (ic<1 || ic>nnodes) continue;
	if (!node_in_lid(ic, seafloor, conv) && !is_itc[ic]) continue;
	double tseed = ttime_w1(ib)+edge_tt(ib, ic);
	if (tseed >= ttime_psp_a(ic)) continue;
	bool in_c = false;
	for (nodeIterator pC=C.begin(); pC!=C.end(); ){
	    if (*pC==ic){
		in_c = true;
		nodeIterator ep=pC;
		pC++;
		C.erase(ep);
	    }else{
		pC++;
	    }
	}
	ttime_psp_a(ic) = tseed;
	prev_psp_a(ic) = ib;
	if (in_c || is_itc[ic]) B_psp_a.push(ic);
    }
    if (B_psp_a.size()==0)
	error("GraphSolver2d::solve_psp_peg - cannot inject lid phase from seafloor after water peg.");
    dijkstra_region(B_psp_a, prev_psp_a, ttime_psp_a, C);
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	double x = smesh.nodePos(itc).x();
	if (x<xmin || x>xmax) continue;
	double best = ttime_inf;
	int bestp = -1;
	ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	for (int j=1; j<=nnodes; j++){
	    if (is_itc[j]) continue;
	    if (!node_in_lid(j, seafloor, conv)) continue;
	    if (ttime_psp_a(j) >= ttime_inf) continue;
	    if (!fstar.isIn(smesh.nodeIndex(j))) continue;
	    double cand = ttime_psp_a(j)+edge_tt(j,itc);
	    if (cand < best){
		best = cand;
		bestp = j;
	    }
	}
	if (bestp != -1){
	    ttime_psp_a(itc) = best;
	    prev_psp_a(itc) = bestp;
	}
    }

    psp_istar = 0;
    double tstar = ttime_inf;
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	double x = smesh.nodePos(itc).x();
	if (x<xmin || x>xmax) continue;
	if (ttime_psp_a(itc) < tstar){
	    tstar = ttime_psp_a(itc);
	    psp_istar = itc;
	}
    }
    if (psp_istar==0 || tstar>=ttime_inf)
	error("GraphSolver2d::solve_psp_peg - peg path cannot reach the conversion interface.");

    ttime_scale = 1.0;
    edge_use_vp = false;
    edge_vp_mixed_conv = false;
    edge_psx_leg = 2;
    B_psp_b.resize(0);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	if (is_itc[i]){
	    prev_psp_b(i) = -1;
	    ttime_psp_b(i) = ttime_inf;
	}else if (node_strictly_below_conv(i, conv)){
	    ttime_psp_b(i) = ttime_inf;
	    prev_psp_b(i) = 0;
	    C.push_back(i);
	}
    }
    {
	for (int n=1; n<=itcnodes.size(); n++){
	    int itc = itcnodes(n);
	    if (itc<1 || itc>nnodes) continue;
	    double x = smesh.nodePos(itc).x();
	    if (x<xmin || x>xmax) continue;
	    if (ttime_psp_a(itc) >= ttime_inf) continue;
	    ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	    for (nodeIterator pC=C.begin(); pC!=C.end(); ++pC){
		int j=*pC;
		if (!fstar.isIn(smesh.nodeIndex(j))) continue;
		double cand = ttime_psp_a(itc)+edge_tt(itc,j);
		if (cand < ttime_psp_b(j)){
		    ttime_psp_b(j) = cand;
		    prev_psp_b(j) = itc;
		}
	    }
	}
    }
    {
	nodeIterator pC=C.begin();
	nodeIterator epC;
	while(pC!=C.end()){
	    if (ttime_psp_b(*pC) < ttime_inf){
		B_psp_b.push(*pC);
		epC = pC;
		pC++;
		C.erase(epC);
	    }else{
		pC++;
	    }
	}
    }
    if (B_psp_b.size()==0)
	error("GraphSolver2d::solve_psp_peg - cannot inject S below the conversion interface.");
    dijkstra_region(B_psp_b, prev_psp_b, ttime_psp_b, C);

    int n_returned = 0;
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	double best = ttime_inf;
	int bestp = -1;
	ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	for (int j=1; j<=nnodes; j++){
	    if (!node_strictly_below_conv(j, conv)) continue;
	    if (ttime_psp_b(j) >= ttime_inf) continue;
	    if (!fstar.isIn(smesh.nodeIndex(j))) continue;
	    double cand = ttime_psp_b(j)+edge_tt(j,itc);
	    if (cand < best){
		best = cand;
		bestp = j;
	    }
	}
	if (bestp != -1){
	    ttime_psp_b(itc) = best;
	    prev_psp_b(itc) = bestp;
	    n_returned++;
	}
    }
    if (n_returned==0)
	error("GraphSolver2d::solve_psp_peg - no converted path returned to the interface from below.");

    ttime_scale = 1.0;
    edge_use_vp = false;
    edge_vp_mixed_conv = false;
    edge_psx_leg = 1;
    B_psp_c.resize(0);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	if (is_itc[i]){
	    if (ttime_psp_b(i) >= ttime_inf) continue;
	    prev_psp_c(i) = -1;
	    ttime_psp_c(i) = ttime_psp_b(i);
	}else if (node_on_or_above_conv(i, conv)){
	    ttime_psp_c(i) = ttime_inf;
	    prev_psp_c(i) = 0;
	    C.push_back(i);
	}
    }
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	if (ttime_psp_b(itc) >= ttime_inf) continue;
	ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	for (nodeIterator pC=C.begin(); pC!=C.end(); ++pC){
	    int j=*pC;
	    if (!fstar.isIn(smesh.nodeIndex(j))) continue;
	    double cand = ttime_psp_b(itc)+edge_tt(itc,j);
	    if (cand < ttime_psp_c(j)){
		ttime_psp_c(j) = cand;
		prev_psp_c(j) = itc;
	    }
	}
    }
    {
	nodeIterator pC=C.begin();
	nodeIterator epC;
	while(pC!=C.end()){
	    if (ttime_psp_c(*pC) < ttime_inf){
		B_psp_c.push(*pC);
		epC = pC;
		pC++;
		C.erase(epC);
	    }else{
		pC++;
	    }
	}
    }
    if (B_psp_c.size()==0)
	error("GraphSolver2d::solve_psp_peg - cannot inject P above the conversion interface.");
    dijkstra_region(B_psp_c, prev_psp_c, ttime_psp_c, C);
    edge_use_vp = false;
    edge_vp_mixed_conv = false;
    edge_psx_leg = 0;
    is_psp_peg_solved = true;
}

void GraphSolver2d::pickPspPegPath(const Point2d& rcv, Array1d<Point2d>& path,
				   int& id0, int& id1, int& ib0, int& ib1,
				   int& is0, int& is1, int& iu0, int& iu1) const
{
    if (!is_psp_peg_solved) error("GraphSolver2d::pickPspPegPath - not solved");
    if (!psp_convp || !water_seafloor || !water_surface)
	error("GraphSolver2d::pickPspPegPath - interfaces missing");
    const Interface2d& conv = *psp_convp;
    const Interface2d& seafloor = *water_seafloor;
    const double ttime_inf = 1e29;

    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);
    int cur = smesh.nearest(rcv);
    if (ttime_psp_c(cur) > ttime_inf)
	error("GraphSolver2d::pickPspPegPath - receiver not reached by peg converted path");

    int prev;
    int nadd=0;
    while ((prev=prev_psp_c(cur)) != -1){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspPegPath - cycle in stage C");
    }
    if (nadd>0) tmp_path.pop_back();
    Point2d on_up(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
    if (nadd==1) tmp_path.push_back(0.5*(on_up+rcv));
    tmp_path.push_back(on_up); iu0 = (int)tmp_path.size();
    tmp_path.push_back(on_up);
    tmp_path.push_back(on_up); iu1 = (int)tmp_path.size();

    Point2d on_dn = on_up;
    if (psx_two_leg){
	id0 = iu1;
	id1 = iu1;
    }else{
	nadd=0;
	prev = prev_psp_b(cur);
	while (prev>0 && prev!=cur && node_strictly_below_conv(prev, conv)){
	    tmp_path.push_back(smesh.nodePos(prev));
	    nadd++;
	    cur = prev;
	    prev = prev_psp_b(cur);
	    if (nadd>nnodes) error("GraphSolver2d::pickPspPegPath - cycle in stage B");
	}
	if (prev>0 && prev!=cur) cur = prev;
	if (nadd>1) tmp_path.pop_back();
	on_dn = Point2d(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
	if (nadd==1) tmp_path.push_back(0.5*(on_dn+on_up));
	tmp_path.push_back(on_dn); id0 = (int)tmp_path.size();
	tmp_path.push_back(on_dn);
	tmp_path.push_back(on_dn); id1 = (int)tmp_path.size();
    }

    nadd=0;
    prev = prev_psp_a(cur);
    while (prev>0 && prev!=cur &&
	   node_in_lid(prev, seafloor, conv) &&
	   smesh.nodePos(prev).y() > seafloor.z(smesh.nodePos(prev).x())+1e-6){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	prev = prev_psp_a(cur);
	if (nadd>nnodes) error("GraphSolver2d::pickPspPegPath - cycle in stage A");
    }
    if (prev>0 && prev!=cur) cur = prev;
    if (nadd>1) tmp_path.pop_back();
    double bx = smesh.nodePos(cur).x();
    Point2d onB(bx, seafloor.z(bx));
    if (nadd==1) tmp_path.push_back(0.5*(onB+on_dn));
    tmp_path.push_back(onB); ib0 = (int)tmp_path.size();
    tmp_path.push_back(onB);
    tmp_path.push_back(onB); ib1 = (int)tmp_path.size();

    nadd=0;
    while ((prev=prev_w1(cur)) != -1){
	if (prev==0) error("GraphSolver2d::pickPspPegPath - broken w1 chain");
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspPegPath - cycle in w1");
    }
    double sx = smesh.nodePos(cur).x();
    Point2d onS(sx, water_surface->z(sx));
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(onS+onB));
    tmp_path.push_back(onS); is0 = (int)tmp_path.size();
    tmp_path.push_back(onS);
    tmp_path.push_back(onS); is1 = (int)tmp_path.size();

    nadd=0;
    while ((prev=prev_w0(cur)) != cur){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspPegPath - cycle in w0");
    }
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(src+onS));
    tmp_path.push_back(src);

    if (refine_if_long){
	int pins[8] = {iu0, iu1, id0, id1, ib0, ib1, is0, is1};
	psp_refine_if_long(tmp_path, crit_len, pins, 8);
	iu0=pins[0]; iu1=pins[1]; id0=pins[2]; id1=pins[3];
	ib0=pins[4]; ib1=pins[5]; is0=pins[6]; is1=pins[7];
    }
    list_to_path(tmp_path, path);
}

void GraphSolver2d::pickPspPegPathThruWater(const Point2d& rcv, Array1d<Point2d>& path,
					    int& i0, int& i1, int& id0, int& id1,
					    int& ib0, int& ib1, int& is0, int& is1,
					    int& iu0, int& iu1) const
{
    if (!is_psp_peg_solved) error("GraphSolver2d::pickPspPegPathThruWater - not solved");
    if (!psp_convp || !water_seafloor || !water_surface)
	error("GraphSolver2d::pickPspPegPathThruWater - interfaces missing");
    const Interface2d& conv = *psp_convp;
    const Interface2d& seafloor = *water_seafloor;
    const double ttime_inf = 1e29;

    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);
    int cur = smesh.nearest(rcv);
    findRayEntryPoint(rcv, cur, ttime_psp_c);
    if (ttime_psp_c(cur) > ttime_inf)
	error("GraphSolver2d::pickPspPegPathThruWater - water entry not reached");

    Point2d entryp=smesh.nodePos(cur);
    tmp_path.push_back(entryp);
    tmp_path.push_back(entryp);
    tmp_path.push_back(entryp);
    i0 = 2; i1 = 4;

    int prev;
    int nadd=0;
    while ((prev=prev_psp_c(cur)) != -1){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspPegPathThruWater - cycle in stage C");
    }
    if (nadd>0) tmp_path.pop_back();
    Point2d on_up(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
    if (nadd==1) tmp_path.push_back(0.5*(on_up+entryp));
    tmp_path.push_back(on_up); iu0 = (int)tmp_path.size();
    tmp_path.push_back(on_up);
    tmp_path.push_back(on_up); iu1 = (int)tmp_path.size();

    Point2d on_dn = on_up;
    if (psx_two_leg){
	id0 = iu1;
	id1 = iu1;
    }else{
	nadd=0;
	prev = prev_psp_b(cur);
	while (prev>0 && prev!=cur && node_strictly_below_conv(prev, conv)){
	    tmp_path.push_back(smesh.nodePos(prev));
	    nadd++;
	    cur = prev;
	    prev = prev_psp_b(cur);
	    if (nadd>nnodes) error("GraphSolver2d::pickPspPegPathThruWater - cycle in stage B");
	}
	if (prev>0 && prev!=cur) cur = prev;
	if (nadd>1) tmp_path.pop_back();
	on_dn = Point2d(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
	if (nadd==1) tmp_path.push_back(0.5*(on_dn+on_up));
	tmp_path.push_back(on_dn); id0 = (int)tmp_path.size();
	tmp_path.push_back(on_dn);
	tmp_path.push_back(on_dn); id1 = (int)tmp_path.size();
    }

    nadd=0;
    prev = prev_psp_a(cur);
    while (prev>0 && prev!=cur &&
	   node_in_lid(prev, seafloor, conv) &&
	   smesh.nodePos(prev).y() > seafloor.z(smesh.nodePos(prev).x())+1e-6){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	prev = prev_psp_a(cur);
	if (nadd>nnodes) error("GraphSolver2d::pickPspPegPathThruWater - cycle in stage A");
    }
    if (prev>0 && prev!=cur) cur = prev;
    if (nadd>1) tmp_path.pop_back();
    double bx = smesh.nodePos(cur).x();
    Point2d onB(bx, seafloor.z(bx));
    if (nadd==1) tmp_path.push_back(0.5*(onB+on_dn));
    tmp_path.push_back(onB); ib0 = (int)tmp_path.size();
    tmp_path.push_back(onB);
    tmp_path.push_back(onB); ib1 = (int)tmp_path.size();

    int ibot = cur;
    nadd=0;
    cur = ibot;
    while ((prev=prev_w1(cur)) != -1){
	if (prev==0) error("GraphSolver2d::pickPspPegPathThruWater - broken w1 chain");
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspPegPathThruWater - cycle in w1");
    }
    double sx = smesh.nodePos(cur).x();
    Point2d onS(sx, water_surface->z(sx));
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(onS+onB));
    tmp_path.push_back(onS); is0 = (int)tmp_path.size();
    tmp_path.push_back(onS);
    tmp_path.push_back(onS); is1 = (int)tmp_path.size();

    nadd=0;
    while ((prev=prev_w0(cur)) != cur){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspPegPathThruWater - cycle in w0");
    }
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(src+onS));
    tmp_path.push_back(src);

    if (refine_if_long){
	int pins[10] = {i0, i1, iu0, iu1, id0, id1, ib0, ib1, is0, is1};
	psp_refine_if_long(tmp_path, crit_len, pins, 10);
	i0=pins[0]; i1=pins[1]; iu0=pins[2]; iu1=pins[3];
	id0=pins[4]; id1=pins[5]; ib0=pins[6]; ib1=pins[7];
	is0=pins[8]; is1=pins[9];
    }
    list_to_path(tmp_path, path);
}

void GraphSolver2d::solve_pps_ss(const Point2d& s, const Interface2d& conv,
				 const Interface2d& seafloor, double kappa)
{
    solve_psx_ss(s, conv, seafloor, kappa, false);
}

void GraphSolver2d::solve_pss_ss(const Point2d& s, const Interface2d& conv,
				 const Interface2d& seafloor, double kappa)
{
    solve_psx_ss(s, conv, seafloor, kappa, true);
}

void GraphSolver2d::solve_psx_ss(const Point2d& s, const Interface2d& conv,
				 const Interface2d& seafloor, double kappa,
				 bool below_is_s)
{
    const double ttime_inf = 1e30;
    if (kappa<=0.0) error("GraphSolver2d::solve_psx_ss - kappa must be positive.");
    src = s;
    psp_convp = &conv;
    water_seafloor = &seafloor;
    is_psp_solved = false;
    is_psp_moho_solved = false;
    is_psp_peg_solved = false;
    is_psp_ss_solved = false;
    psx_two_leg = !below_is_s;
    ttime_scale = 1.0;
    edge_use_vp = false;
    edge_vp_mixed_conv = false;
    edge_psx_leg = 0;

    Array1d<int> itcnodes, botnodes;
    smesh.nearest(conv, itcnodes);
    smesh.nearest(seafloor, botnodes);
    std::vector<char> is_itc(nnodes + 1, 0);
    std::vector<char> is_bot(nnodes + 1, 0);
    for (int n=1; n<=itcnodes.size(); n++)
	is_itc[itcnodes(n)] = 1;
    for (int n=1; n<=botnodes.size(); n++)
	is_bot[botnodes(n)] = 1;

    int Nsrc = smesh.nearest(src);
    if (!node_in_lid(Nsrc, seafloor, conv) && !is_itc[Nsrc])
	error("GraphSolver2d::solve_psx_ss - source must sit in the lid.");

    for (int i=1; i<=nnodes; i++){
	ttime_psp_a(i) = ttime_inf;
	ttime_psp_b(i) = ttime_inf;
	ttime_psp_c(i) = ttime_inf;
	ttime_ss_dn(i) = ttime_inf;
	ttime_ss_up(i) = ttime_inf;
	prev_psp_a(i) = 0;
	prev_psp_b(i) = 0;
	prev_psp_c(i) = 0;
	prev_ss_dn(i) = 0;
	prev_ss_up(i) = 0;
    }

    // Stage A: S OBS → lid / conv (same as PPS/PSS primary).
    ttime_scale = 1.0;
    edge_psx_leg = 3;
    B_psp_a.resize(0);
    prev_psp_a(Nsrc) = Nsrc;
    ttime_psp_a(Nsrc) = 0.0;
    B_psp_a.push(Nsrc);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	if (i==Nsrc) continue;
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	if (node_in_lid(i, seafloor, conv) || is_itc[i]){
	    C.push_back(i);
	    prev_psp_a(i) = Nsrc;
	}
    }
    dijkstra_region(B_psp_a, prev_psp_a, ttime_psp_a, C);

    psp_istar = 0;
    double tstar = ttime_inf;
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	double x = smesh.nodePos(itc).x();
	if (x<xmin || x>xmax) continue;
	if (ttime_psp_a(itc) < tstar){
	    tstar = ttime_psp_a(itc);
	    psp_istar = itc;
	}
    }
    if (psp_istar==0 || tstar>=ttime_inf)
	error("GraphSolver2d::solve_psx_ss - OBS cannot reach the conversion interface as S.");

    // Stage ss_dn: S reflects at conv, stays in the lid, goes to seafloor.
    ttime_scale = 1.0;
    edge_psx_leg = 3;
    B_ss_dn.resize(0);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	if (is_itc[i]){
	    if (ttime_psp_a(i) >= ttime_inf) continue;
	    prev_ss_dn(i) = -1;
	    ttime_ss_dn(i) = ttime_psp_a(i);
	}else if (node_in_lid(i, seafloor, conv)){
	    ttime_ss_dn(i) = ttime_inf;
	    prev_ss_dn(i) = 0;
	    C.push_back(i);
	}
    }
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	double x = smesh.nodePos(itc).x();
	if (x<xmin || x>xmax) continue;
	if (ttime_psp_a(itc) >= ttime_inf) continue;
	ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	for (nodeIterator pC=C.begin(); pC!=C.end(); ++pC){
	    int j=*pC;
	    if (!fstar.isIn(smesh.nodeIndex(j))) continue;
	    double cand = ttime_psp_a(itc)+edge_tt(itc,j);
	    if (cand < ttime_ss_dn(j)){
		ttime_ss_dn(j) = cand;
		prev_ss_dn(j) = itc;
	    }
	}
    }
    {
	nodeIterator pC=C.begin();
	nodeIterator epC;
	while(pC!=C.end()){
	    if (ttime_ss_dn(*pC) < ttime_inf){
		B_ss_dn.push(*pC);
		epC = pC;
		pC++;
		C.erase(epC);
	    }else{
		pC++;
	    }
	}
    }
    if (B_ss_dn.size()==0)
	error("GraphSolver2d::solve_psx_ss - cannot inject lid S from conv toward seafloor.");
    dijkstra_region(B_ss_dn, prev_ss_dn, ttime_ss_dn, C);

    int n_bot = 0;
    for (int n=1; n<=botnodes.size(); n++){
	int ib = botnodes(n);
	if (ib<1 || ib>nnodes) continue;
	if (ttime_ss_dn(ib) < ttime_inf) n_bot++;
    }
    if (n_bot==0)
	error("GraphSolver2d::solve_psx_ss - lid S bounce cannot reach the seafloor.");

    // Stage ss_up: SS at seafloor, S back through lid to conv.
    ttime_scale = 1.0;
    edge_psx_leg = 3;
    B_ss_up.resize(0);
    C.resize(0);
    for (int i=1; i<=nnodes; i++){
	double x = smesh.nodePos(i).x();
	if (x<xmin || x>xmax) continue;
	if (is_bot[i]) continue;
	if (node_in_lid(i, seafloor, conv) || is_itc[i]){
	    C.push_back(i);
	    prev_ss_up(i) = 0;
	}
    }
    for (int n=1; n<=botnodes.size(); n++){
	int ib = botnodes(n);
	if (ib<1 || ib>nnodes) continue;
	if (ttime_ss_dn(ib) >= ttime_inf) continue;
	prev_ss_up(ib) = -1;
	ttime_ss_up(ib) = ttime_ss_dn(ib);
	Index2d id = smesh.nodeIndex(ib);
	if (id.k() >= smesh.Nz()) continue;
	int ic = smesh.nodeIndex(id.i(), id.k()+1);
	if (ic<1 || ic>nnodes) continue;
	if (!node_in_lid(ic, seafloor, conv) && !is_itc[ic]) continue;
	double tseed = ttime_ss_dn(ib)+edge_tt(ib, ic);
	if (tseed >= ttime_ss_up(ic)) continue;
	bool in_c = false;
	for (nodeIterator pC=C.begin(); pC!=C.end(); ){
	    if (*pC==ic){
		in_c = true;
		nodeIterator ep=pC;
		pC++;
		C.erase(ep);
	    }else{
		pC++;
	    }
	}
	ttime_ss_up(ic) = tseed;
	prev_ss_up(ic) = ib;
	if (in_c || is_itc[ic]) B_ss_up.push(ic);
    }
    if (B_ss_up.size()==0)
	error("GraphSolver2d::solve_psx_ss - cannot inject lid S from seafloor after SS.");
    dijkstra_region(B_ss_up, prev_ss_up, ttime_ss_up, C);
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	double x = smesh.nodePos(itc).x();
	if (x<xmin || x>xmax) continue;
	double best = ttime_inf;
	int bestp = -1;
	ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	for (int j=1; j<=nnodes; j++){
	    if (is_itc[j]) continue;
	    if (!node_in_lid(j, seafloor, conv)) continue;
	    if (ttime_ss_up(j) >= ttime_inf) continue;
	    if (!fstar.isIn(smesh.nodeIndex(j))) continue;
	    double cand = ttime_ss_up(j)+edge_tt(j,itc);
	    if (cand < best){
		best = cand;
		bestp = j;
	    }
	}
	if (bestp != -1){
	    ttime_ss_up(itc) = best;
	    prev_ss_up(itc) = bestp;
	}
    }
    int n_c2 = 0;
    for (int n=1; n<=itcnodes.size(); n++){
	int itc = itcnodes(n);
	if (itc<1 || itc>nnodes) continue;
	if (ttime_ss_up(itc) < ttime_inf) n_c2++;
    }
    if (n_c2==0)
	error("GraphSolver2d::solve_psx_ss - SS bounce cannot return to the conversion interface.");

    if (!below_is_s){
	ttime_scale = 1.0;
	edge_psx_leg = 4;
	B_psp_c.resize(0);
	C.resize(0);
	for (int i=1; i<=nnodes; i++){
	    double x = smesh.nodePos(i).x();
	    if (x<xmin || x>xmax) continue;
	    bool in_p = node_on_or_above_conv(i, conv) || is_itc[i]
		|| node_strictly_below_conv(i, conv);
	    if (!in_p) continue;
	    if (is_itc[i]){
		if (ttime_ss_up(i) >= ttime_inf) continue;
		prev_psp_c(i) = -1;
		ttime_psp_c(i) = ttime_ss_up(i);
		B_psp_c.push(i);
	    }else{
		ttime_psp_c(i) = ttime_inf;
		prev_psp_c(i) = 0;
		C.push_back(i);
	    }
	}
	if (B_psp_c.size()==0)
	    error("GraphSolver2d::solve_psx_ss - PPS SS cannot convert to P at conv.");
	dijkstra_region(B_psp_c, prev_psp_c, ttime_psp_c, C);
    }else{
	ttime_scale = 1.0;
	edge_psx_leg = 2;
	B_psp_b.resize(0);
	C.resize(0);
	for (int i=1; i<=nnodes; i++){
	    double x = smesh.nodePos(i).x();
	    if (x<xmin || x>xmax) continue;
	    if (is_itc[i]){
		prev_psp_b(i) = -1;
		ttime_psp_b(i) = ttime_inf;
	    }else if (node_strictly_below_conv(i, conv)){
		ttime_psp_b(i) = ttime_inf;
		prev_psp_b(i) = 0;
		C.push_back(i);
	    }
	}
	for (int n=1; n<=itcnodes.size(); n++){
	    int itc = itcnodes(n);
	    if (itc<1 || itc>nnodes) continue;
	    double x = smesh.nodePos(itc).x();
	    if (x<xmin || x>xmax) continue;
	    if (ttime_ss_up(itc) >= ttime_inf) continue;
	    ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	    for (nodeIterator pC=C.begin(); pC!=C.end(); ++pC){
		int j=*pC;
		if (!fstar.isIn(smesh.nodeIndex(j))) continue;
		double cand = ttime_ss_up(itc)+edge_tt(itc,j);
		if (cand < ttime_psp_b(j)){
		    ttime_psp_b(j) = cand;
		    prev_psp_b(j) = itc;
		}
	    }
	}
	{
	    nodeIterator pC=C.begin();
	    nodeIterator epC;
	    while(pC!=C.end()){
		if (ttime_psp_b(*pC) < ttime_inf){
		    B_psp_b.push(*pC);
		    epC = pC;
		    pC++;
		    C.erase(epC);
		}else{
		    pC++;
		}
	    }
	}
	if (B_psp_b.size()==0)
	    error("GraphSolver2d::solve_psx_ss - cannot inject S below conv after lid SS.");
	dijkstra_region(B_psp_b, prev_psp_b, ttime_psp_b, C);

	int n_returned = 0;
	for (int n=1; n<=itcnodes.size(); n++){
	    int itc = itcnodes(n);
	    if (itc<1 || itc>nnodes) continue;
	    double best = ttime_inf;
	    int bestp = -1;
	    ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	    for (int j=1; j<=nnodes; j++){
		if (!node_strictly_below_conv(j, conv)) continue;
		if (ttime_psp_b(j) >= ttime_inf) continue;
		if (!fstar.isIn(smesh.nodeIndex(j))) continue;
		double cand = ttime_psp_b(j)+edge_tt(j,itc);
		if (cand < best){
		    best = cand;
		    bestp = j;
		}
	    }
	    if (bestp != -1){
		ttime_psp_b(itc) = best;
		prev_psp_b(itc) = bestp;
		n_returned++;
	    }
	}
	if (n_returned==0)
	    error("GraphSolver2d::solve_psx_ss - no S path returned to conv from below.");

	ttime_scale = 1.0;
	edge_psx_leg = 1;
	B_psp_c.resize(0);
	C.resize(0);
	for (int i=1; i<=nnodes; i++){
	    double x = smesh.nodePos(i).x();
	    if (x<xmin || x>xmax) continue;
	    if (is_itc[i]){
		if (ttime_psp_b(i) >= ttime_inf) continue;
		prev_psp_c(i) = -1;
		ttime_psp_c(i) = ttime_psp_b(i);
	    }else if (node_on_or_above_conv(i, conv)){
		ttime_psp_c(i) = ttime_inf;
		prev_psp_c(i) = 0;
		C.push_back(i);
	    }
	}
	for (int n=1; n<=itcnodes.size(); n++){
	    int itc = itcnodes(n);
	    if (itc<1 || itc>nnodes) continue;
	    if (ttime_psp_b(itc) >= ttime_inf) continue;
	    ForwardStar2d fstar(smesh.nodeIndex(itc), fs_xorder, fs_zorder);
	    for (nodeIterator pC=C.begin(); pC!=C.end(); ++pC){
		int j=*pC;
		if (!fstar.isIn(smesh.nodeIndex(j))) continue;
		double cand = ttime_psp_b(itc)+edge_tt(itc,j);
		if (cand < ttime_psp_c(j)){
		    ttime_psp_c(j) = cand;
		    prev_psp_c(j) = itc;
		}
	    }
	}
	{
	    nodeIterator pC=C.begin();
	    nodeIterator epC;
	    while(pC!=C.end()){
		if (ttime_psp_c(*pC) < ttime_inf){
		    B_psp_c.push(*pC);
		    epC = pC;
		    pC++;
		    C.erase(epC);
		}else{
		    pC++;
		}
	    }
	}
	if (B_psp_c.size()==0)
	    error("GraphSolver2d::solve_psx_ss - cannot inject P above conv after lid SS.");
	dijkstra_region(B_psp_c, prev_psp_c, ttime_psp_c, C);
    }
    edge_psx_leg = 0;
    is_psp_ss_solved = true;
}

void GraphSolver2d::pickPspSsPath(const Point2d& rcv, Array1d<Point2d>& path,
				  int& id0, int& id1, int& ib0, int& ib1,
				  int& ic0, int& ic1, int& iu0, int& iu1) const
{
    if (!is_psp_ss_solved) error("GraphSolver2d::pickPspSsPath - not solved");
    if (!psp_convp || !water_seafloor)
	error("GraphSolver2d::pickPspSsPath - interfaces missing");
    const Interface2d& conv = *psp_convp;
    const Interface2d& seafloor = *water_seafloor;
    const double ttime_inf = 1e29;

    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);
    int cur = smesh.nearest(rcv);
    if (ttime_psp_c(cur) > ttime_inf)
	error("GraphSolver2d::pickPspSsPath - receiver not reached");

    int prev;
    int nadd=0;
    while ((prev=prev_psp_c(cur)) != -1){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspSsPath - cycle in stage C");
    }
    if (nadd>0) tmp_path.pop_back();
    Point2d on_up(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
    if (nadd==1) tmp_path.push_back(0.5*(on_up+rcv));
    tmp_path.push_back(on_up); iu0 = (int)tmp_path.size();
    tmp_path.push_back(on_up);
    tmp_path.push_back(on_up); iu1 = (int)tmp_path.size();

    Point2d on_dn = on_up;
    if (psx_two_leg){
	id0 = iu1;
	id1 = iu1;
    }else{
	nadd=0;
	prev = prev_psp_b(cur);
	while (prev>0 && prev!=cur && node_strictly_below_conv(prev, conv)){
	    tmp_path.push_back(smesh.nodePos(prev));
	    nadd++;
	    cur = prev;
	    prev = prev_psp_b(cur);
	    if (nadd>nnodes) error("GraphSolver2d::pickPspSsPath - cycle in stage B");
	}
	if (prev>0 && prev!=cur) cur = prev;
	if (nadd>1) tmp_path.pop_back();
	on_dn = Point2d(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
	if (nadd==1) tmp_path.push_back(0.5*(on_dn+on_up));
	tmp_path.push_back(on_dn); id0 = (int)tmp_path.size();
	tmp_path.push_back(on_dn);
	tmp_path.push_back(on_dn); id1 = (int)tmp_path.size();
    }

    nadd=0;
    prev = prev_ss_up(cur);
    while (prev>0 && prev!=cur &&
	   node_in_lid(prev, seafloor, conv) &&
	   smesh.nodePos(prev).y() > seafloor.z(smesh.nodePos(prev).x())+1e-6){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	prev = prev_ss_up(cur);
	if (nadd>nnodes) error("GraphSolver2d::pickPspSsPath - cycle in ss_up");
    }
    if (prev>0 && prev!=cur) cur = prev;
    if (nadd>1) tmp_path.pop_back();
    double bx = smesh.nodePos(cur).x();
    Point2d onB(bx, seafloor.z(bx));
    if (nadd==1) tmp_path.push_back(0.5*(onB+on_dn));
    tmp_path.push_back(onB); ib0 = (int)tmp_path.size();
    tmp_path.push_back(onB);
    tmp_path.push_back(onB); ib1 = (int)tmp_path.size();

    nadd=0;
    prev = prev_ss_dn(cur);
    while (prev>0 && prev!=cur && node_in_lid(prev, seafloor, conv)){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	prev = prev_ss_dn(cur);
	if (nadd>nnodes) error("GraphSolver2d::pickPspSsPath - cycle in ss_dn");
    }
    if (prev>0 && prev!=cur) cur = prev;
    if (nadd>1) tmp_path.pop_back();
    Point2d onC1(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
    if (nadd==1) tmp_path.push_back(0.5*(onC1+onB));
    tmp_path.push_back(onC1); ic0 = (int)tmp_path.size();
    tmp_path.push_back(onC1);
    tmp_path.push_back(onC1); ic1 = (int)tmp_path.size();

    nadd=0;
    prev = prev_psp_a(cur);
    while (prev>0 && prev!=cur){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	prev = prev_psp_a(cur);
	if (nadd>nnodes) error("GraphSolver2d::pickPspSsPath - cycle in stage A");
    }
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(src+onC1));
    tmp_path.push_back(src);

    if (refine_if_long){
	int pins[8] = {iu0, iu1, id0, id1, ib0, ib1, ic0, ic1};
	psp_refine_if_long(tmp_path, crit_len, pins, 8);
	iu0=pins[0]; iu1=pins[1]; id0=pins[2]; id1=pins[3];
	ib0=pins[4]; ib1=pins[5]; ic0=pins[6]; ic1=pins[7];
    }
    list_to_path(tmp_path, path);
}

void GraphSolver2d::pickPspSsPathThruWater(const Point2d& rcv, Array1d<Point2d>& path,
					   int& i0, int& i1, int& id0, int& id1,
					   int& ib0, int& ib1, int& ic0, int& ic1,
					   int& iu0, int& iu1) const
{
    if (!is_psp_ss_solved) error("GraphSolver2d::pickPspSsPathThruWater - not solved");
    if (!psp_convp || !water_seafloor)
	error("GraphSolver2d::pickPspSsPathThruWater - interfaces missing");
    const Interface2d& conv = *psp_convp;
    const Interface2d& seafloor = *water_seafloor;
    const double ttime_inf = 1e29;

    list<Point2d> tmp_path;
    tmp_path.push_back(rcv);
    int cur = smesh.nearest(rcv);
    findRayEntryPoint(rcv, cur, ttime_psp_c);
    if (ttime_psp_c(cur) > ttime_inf)
	error("GraphSolver2d::pickPspSsPathThruWater - water entry not reached");

    Point2d entryp=smesh.nodePos(cur);
    tmp_path.push_back(entryp);
    tmp_path.push_back(entryp);
    tmp_path.push_back(entryp);
    i0 = 2; i1 = 4;

    int prev;
    int nadd=0;
    while ((prev=prev_psp_c(cur)) != -1){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	if (nadd>nnodes) error("GraphSolver2d::pickPspSsPathThruWater - cycle in stage C");
    }
    if (nadd>0) tmp_path.pop_back();
    Point2d on_up(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
    if (nadd==1) tmp_path.push_back(0.5*(on_up+entryp));
    tmp_path.push_back(on_up); iu0 = (int)tmp_path.size();
    tmp_path.push_back(on_up);
    tmp_path.push_back(on_up); iu1 = (int)tmp_path.size();

    Point2d on_dn = on_up;
    if (psx_two_leg){
	id0 = iu1;
	id1 = iu1;
    }else{
	nadd=0;
	prev = prev_psp_b(cur);
	while (prev>0 && prev!=cur && node_strictly_below_conv(prev, conv)){
	    tmp_path.push_back(smesh.nodePos(prev));
	    nadd++;
	    cur = prev;
	    prev = prev_psp_b(cur);
	    if (nadd>nnodes) error("GraphSolver2d::pickPspSsPathThruWater - cycle in stage B");
	}
	if (prev>0 && prev!=cur) cur = prev;
	if (nadd>1) tmp_path.pop_back();
	on_dn = Point2d(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
	if (nadd==1) tmp_path.push_back(0.5*(on_dn+on_up));
	tmp_path.push_back(on_dn); id0 = (int)tmp_path.size();
	tmp_path.push_back(on_dn);
	tmp_path.push_back(on_dn); id1 = (int)tmp_path.size();
    }

    nadd=0;
    prev = prev_ss_up(cur);
    while (prev>0 && prev!=cur &&
	   node_in_lid(prev, seafloor, conv) &&
	   smesh.nodePos(prev).y() > seafloor.z(smesh.nodePos(prev).x())+1e-6){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	prev = prev_ss_up(cur);
	if (nadd>nnodes) error("GraphSolver2d::pickPspSsPathThruWater - cycle in ss_up");
    }
    if (prev>0 && prev!=cur) cur = prev;
    if (nadd>1) tmp_path.pop_back();
    double bx = smesh.nodePos(cur).x();
    Point2d onB(bx, seafloor.z(bx));
    if (nadd==1) tmp_path.push_back(0.5*(onB+on_dn));
    tmp_path.push_back(onB); ib0 = (int)tmp_path.size();
    tmp_path.push_back(onB);
    tmp_path.push_back(onB); ib1 = (int)tmp_path.size();

    nadd=0;
    prev = prev_ss_dn(cur);
    while (prev>0 && prev!=cur && node_in_lid(prev, seafloor, conv)){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	prev = prev_ss_dn(cur);
	if (nadd>nnodes) error("GraphSolver2d::pickPspSsPathThruWater - cycle in ss_dn");
    }
    if (prev>0 && prev!=cur) cur = prev;
    if (nadd>1) tmp_path.pop_back();
    Point2d onC1(smesh.nodePos(cur).x(), conv.z(smesh.nodePos(cur).x()));
    if (nadd==1) tmp_path.push_back(0.5*(onC1+onB));
    tmp_path.push_back(onC1); ic0 = (int)tmp_path.size();
    tmp_path.push_back(onC1);
    tmp_path.push_back(onC1); ic1 = (int)tmp_path.size();

    nadd=0;
    prev = prev_psp_a(cur);
    while (prev>0 && prev!=cur){
	tmp_path.push_back(smesh.nodePos(prev));
	nadd++;
	cur = prev;
	prev = prev_psp_a(cur);
	if (nadd>nnodes) error("GraphSolver2d::pickPspSsPathThruWater - cycle in stage A");
    }
    if (nadd>0) tmp_path.pop_back();
    if (nadd==1) tmp_path.push_back(0.5*(src+onC1));
    tmp_path.push_back(src);

    if (refine_if_long){
	int pins[10] = {i0, i1, iu0, iu1, id0, id1, ib0, ib1, ic0, ic1};
	psp_refine_if_long(tmp_path, crit_len, pins, 10);
	i0=pins[0]; i1=pins[1]; iu0=pins[2]; iu1=pins[3];
	id0=pins[4]; id1=pins[5]; ib0=pins[6]; ib1=pins[7];
	ic0=pins[8]; ic1=pins[9];
    }
    list_to_path(tmp_path, path);
}

void clampPathToWaterCol(Array1d<Point2d>& path,
			 const Interface2d& z_lo, const Interface2d& z_hi)
{
    for (int i=1; i<=path.size(); i++){
	double x = path(i).x();
	double z = path(i).y();
	double lo = z_lo.z(x);
	double hi = z_hi.z(x);
	if (lo > hi){
	    double tmp = lo; lo = hi; hi = tmp;
	}
	if (z < lo) z = lo;
	if (z > hi) z = hi;
	path(i) = Point2d(x, z);
    }
}

//
// forward star
//
ForwardStar2d::ForwardStar2d(const Index2d& id, int ix, int iz)
    : orig(id)
{
    if (ix>0 && iz>0){ xorder=ix; zorder=iz; }
    else{ error("ForwardStar2d::non-positive order detected."); }
}

bool ForwardStar2d::isIn(const Index2d& node) const
{
    int diff;
    int di = (diff=orig.i()-node.i()) > 0 ? diff : -diff;
    int dk = (diff=orig.k()-node.k()) > 0 ? diff : -diff;

    if (di>xorder || dk>zorder) return false;
    if (di==1 || dk==1) return true;
    if (di==0 || dk==0) return false; // NB: (0,1),(1,0) are already returned true.

    // now di && dk must be >=2 
    int mod = dk>di ? dk%di : di%dk;
    return mod==0 ? false : true;
}
