/*
 * graph.h - graph method related classes
 *
 * Jun Korenaga, MIT/WHOI
 * January 1999
 */

#ifndef _TOMO_GRAPH_H_
#define _TOMO_GRAPH_H_

#include <list>
#include <array.h> // from mconv source
#include <geom.h>
#include "heap_deque.h"
#include "index.h"
#include "interface.h"
#include "geom.h"

class TTCmp {
public:
    TTCmp(Array1d<double>& a) : data(a) {}
    bool operator()(int i, int j) const { return data(i)>data(j); }
private:
    Array1d<double>& data;
};

class GraphSolver2d {
public:
    GraphSolver2d(const SlownessMesh2d&, int xorder, int zorder);

    int xOrder() const { return fs_xorder; }
    int zOrder() const { return fs_zorder; }
    void solve(const Point2d& src);
    void solve_conv(const Point2d& src, const Interface2d& itf);
    void solve_refl(const Point2d& src, const Interface2d& itf);
    void solve_refl_conv(const Point2d& src, const Interface2d& itf, const Interface2d& itc);
    void do_refl_downward();
    bool reflDownward() const { return solve_refl_downward; }
    void limitRange(double xmin, double xmax);
    void delimitRange();
    void refineIfLong(double);
    double critLength() const { return crit_len; }
    
    void pickPath(const Point2d& rcv, Array1d<Point2d>& path) const;
    void pickPathThruWater(const Point2d& rcv, Array1d<Point2d>& path,
			   int&, int&) const;
    void pickReflPath(const Point2d& rcv, Array1d<Point2d>& path,
		      int&, int&) const;
    void pickReflPathThruWater(const Point2d& rcv, Array1d<Point2d>& path,
			       int&, int&, int&, int&) const;

    // Water-column phases (raytype 2/3). Independent arrays — does not
    // touch prev_node / prev_node_up / prev_node_down used by code 0/1.
    void solve_water(const Point2d& src, const Interface2d& seafloor);
    void solve_water_mult(const Interface2d& seafloor, const Interface2d& surface);
    void pickWaterDirectPath(const Point2d& rcv, Array1d<Point2d>& path,
			     int& ir0, int& ir1) const;
    void pickWaterMultPath(const Point2d& rcv, Array1d<Point2d>& path,
			   int& i0, int& i1, int& ir0, int& ir1) const;

    // Receiver-side 1st-order peg-leg (raytype 4/5). Reuses water w0/w1
    // (OBS → surface → seafloor); does not overwrite code 0/1 prev_node.
    void solve_recv_peg_refr(const Interface2d& seafloor, const Interface2d& surface);
    void pickRecvPegRefrPath(const Point2d& rcv, Array1d<Point2d>& path,
			     int& is0, int& is1, int& ib0, int& ib1,
			     int& ie0, int& ie1) const;
    void solve_recv_peg_refl(const Interface2d& seafloor, const Interface2d& surface,
			    const Interface2d& moho);
    void pickRecvPegReflPath(const Point2d& rcv, Array1d<Point2d>& path,
			     int& is0, int& is1, int& ib0, int& ib1,
			     int& ir0, int& ir1) const;

    // Reduced PSP (raytype 6): P above conv, S below, P above again.
    // Conversion points minimise the whole-path travel time t_P+t_S+t_P.
    // Independent arrays — does not overwrite code 0/1 prev_node.
    // Dual/mixed: graph edges use one phase at both ends (P=Vp, S=Vs).
    void solve_psp(const Point2d& src, const Interface2d& conv,
		   double kappa=1.0,
		   const Interface2d* seafloor=0, bool lid_a_is_s=false);
    void pickPspPath(const Point2d& rcv, Array1d<Point2d>& path,
		     int& id0, int& id1, int& iu0, int& iu1) const;
    void pickPspPathThruWater(const Point2d& rcv, Array1d<Point2d>& path,
			      int& i0, int& i1,
			      int& id0, int& id1, int& iu0, int& iu1) const;

    // PPS (7): two legs, one star — S in lid OBS→C, then P (lid+below+water) C→shot.
    // PSS (8): same three legs as PSP, but OBS→C is S in the lid (not P).
    // Below-interface S is strictly below; interface nodes are terminals only.
    // S legs use t *= kappa unless smesh.hasDualVs().
    void solve_pps(const Point2d& src, const Interface2d& conv,
		   const Interface2d& seafloor, double kappa);
    void solve_pss(const Point2d& src, const Interface2d& conv,
		   const Interface2d& seafloor, double kappa);

    // Moho-reflected PSP (12) / PSS (13): same lid legs as 6/8, but the
    // below-conv S leg reflects on -F instead of turning. Independent of
    // prev_node_down/up (raytype 1). Water stays P.
    void solve_psp_moho(const Point2d& src, const Interface2d& conv,
			const Interface2d& moho, double kappa=1.0,
			const Interface2d* seafloor=0, bool lid_a_is_s=false);
    void solve_pss_moho(const Point2d& src, const Interface2d& conv,
			const Interface2d& moho, const Interface2d& seafloor,
			double kappa);
    void pickPspMohoPath(const Point2d& rcv, Array1d<Point2d>& path,
			 int& id0, int& id1, int& ir0, int& ir1,
			 int& iu0, int& iu1) const;
    void pickPspMohoPathThruWater(const Point2d& rcv, Array1d<Point2d>& path,
				  int& i0, int& i1, int& id0, int& id1,
				  int& ir0, int& ir1, int& iu0, int& iu1) const;

    // Receiver-side 1st-order water peg of 6/7/8 (raytype 9/14/15).
    // Water column is P only (reuses w0/w1). PPS/PSS convert S↔P at the seafloor.
    // Independent of prev_node used by 0/1 and of rp_crust used by 4/5.
    void solve_psp_peg(const Point2d& src, const Interface2d& conv,
		       const Interface2d& seafloor, double kappa=1.0,
		       bool lid_a_is_s=false);
    void solve_pps_peg(const Point2d& src, const Interface2d& conv,
		       const Interface2d& seafloor, double kappa);
    void solve_pss_peg(const Point2d& src, const Interface2d& conv,
		       const Interface2d& seafloor, double kappa);
    void pickPspPegPath(const Point2d& rcv, Array1d<Point2d>& path,
		       int& id0, int& id1, int& ib0, int& ib1,
		       int& is0, int& is1, int& iu0, int& iu1) const;
    void pickPspPegPathThruWater(const Point2d& rcv, Array1d<Point2d>& path,
				 int& i0, int& i1, int& id0, int& id1,
				 int& ib0, int& ib1, int& is0, int& is1,
				 int& iu0, int& iu1) const;

    // Lid SS peg of PPS/PSS (raytype 10/11): S stays in the lid, reflects
    // at conv then at the seafloor, then continues as 7/8. No water S.
    void solve_pps_ss(const Point2d& src, const Interface2d& conv,
		      const Interface2d& seafloor, double kappa);
    void solve_pss_ss(const Point2d& src, const Interface2d& conv,
		      const Interface2d& seafloor, double kappa);
    void pickPspSsPath(const Point2d& rcv, Array1d<Point2d>& path,
		       int& id0, int& id1, int& ib0, int& ib1,
		       int& ic0, int& ic1, int& iu0, int& iu1) const;
    void pickPspSsPathThruWater(const Point2d& rcv, Array1d<Point2d>& path,
				int& i0, int& i1, int& id0, int& id1,
				int& ib0, int& ib1, int& ic0, int& ic1,
				int& iu0, int& iu1) const;
    
    void printPath(ostream&) const;

private:
    void findRayEntryPoint(const Point2d& rcv, int& cur, 
			   const Array1d<double>& nodetime) const;
    bool node_in_water_col(int inode, const Interface2d& seafloor) const;
    bool node_in_crust(int inode, const Interface2d& seafloor) const;
    void dijkstra_water(heap_deque<int,TTCmp>& Bq, Array1d<int>& prv,
			Array1d<double>& tt, const Interface2d& seafloor,
			list<int>& Cq);
    void dijkstra_crust(heap_deque<int,TTCmp>& Bq, Array1d<int>& prv,
			Array1d<double>& tt, const Interface2d& seafloor,
			list<int>& Cq);
    void dijkstra_region(heap_deque<int,TTCmp>& Bq, Array1d<int>& prv,
			 Array1d<double>& tt, list<int>& Cq);
    bool node_on_or_above_conv(int inode, const Interface2d& conv) const;
    bool node_strictly_below_conv(int inode, const Interface2d& conv) const;
    bool node_in_lid(int inode, const Interface2d& seafloor,
		     const Interface2d& conv) const;
    bool node_in_psm_slab(int inode, const Interface2d& conv,
			  const Interface2d& moho) const;
    void solve_psx(const Point2d& src, const Interface2d& conv,
		   const Interface2d& seafloor, double kappa, bool below_is_s);
    void solve_psx_ss(const Point2d& src, const Interface2d& conv,
		      const Interface2d& seafloor, double kappa, bool below_is_s);
    double edge_tt(int a, int b) const;
    void list_to_path(const list<Point2d>& tmp_path, Array1d<Point2d>& path) const;
    
    const SlownessMesh2d& smesh;
    int nnodes;
    bool is_solved, is_refl_solved, solve_refl_downward;
    bool is_water_solved, is_water_mult_solved;
    bool is_recv_peg_refr_solved, is_recv_peg_refl_solved;
    bool is_psp_solved, is_psp_moho_solved, is_psp_peg_solved, is_psp_ss_solved;
    bool psx_two_leg;
    Point2d src;
    int fs_xorder, fs_zorder;
    Array1d<int> prev_node, prev_node_down, prev_node_up;
    Array1d<double> ttime, ttime_down, ttime_up;
    Array1d<int> prev_w0, prev_w1, prev_w2;
    Array1d<double> ttime_w0, ttime_w1, ttime_w2;
    list<int> C;
    list<int> C_w;
    TTCmp cmp, cmp_down, cmp_up;
    heap_deque<int,TTCmp> B, B_down, B_up;
    TTCmp cmp_w0, cmp_w1, cmp_w2;
    heap_deque<int,TTCmp> B_w0, B_w1, B_w2;
    Array1d<int> prev_rp_crust, prev_rp2, prev_sp, prev_rf_down;
    Array1d<double> ttime_rp_crust, ttime_rp2, ttime_sp, ttime_rf_down;
    TTCmp cmp_rp_crust, cmp_rp2, cmp_sp, cmp_rf_down;
    heap_deque<int,TTCmp> B_rp_crust, B_rp2, B_sp, B_rf_down;
    Array1d<int> prev_psp_a, prev_psp_b, prev_psp_c, prev_psm_u;
    Array1d<double> ttime_psp_a, ttime_psp_b, ttime_psp_c, ttime_psm_u;
    TTCmp cmp_psp_a, cmp_psp_b, cmp_psp_c, cmp_psm_u;
    heap_deque<int,TTCmp> B_psp_a, B_psp_b, B_psp_c, B_psm_u;
    Array1d<int> prev_ss_dn, prev_ss_up;
    Array1d<double> ttime_ss_dn, ttime_ss_up;
    TTCmp cmp_ss_dn, cmp_ss_up;
    heap_deque<int,TTCmp> B_ss_dn, B_ss_up;
    const Interface2d* water_seafloor;
    const Interface2d* water_surface;
    const Interface2d* water_moho;
    const Interface2d* psp_convp;
    const Interface2d* psp_mohop;
    int psp_istar;
    const Interface2d* itfp,itcp;
    typedef list<int>::const_iterator nodeBrowser;
    typedef list<int>::iterator nodeIterator;
    typedef heap_deque<int,TTCmp>::const_iterator heapBrowser;

    const double local_search_radius;
    double xmin, xmax;
    bool refine_if_long;
    double crit_len;
    double ttime_scale;
    bool edge_use_vp;
    bool edge_vp_mixed_conv;
    int edge_psx_leg; // 0=off, 1=P lid, 2=S below, 3=S lid, 4=P auto
};

// Clip a water-column path to z_lo (sea surface) … z_hi (seafloor).
void clampPathToWaterCol(Array1d<Point2d>& path,
			 const Interface2d& z_lo, const Interface2d& z_hi);

class ForwardStar2d {
public:
    ForwardStar2d(const Index2d& id, int ix, int iz);
    bool isIn(const Index2d&) const;

private:
    const Index2d& orig;
    int xorder, zorder;
};

#endif /* _TOMO_GRAPH_H_ */
