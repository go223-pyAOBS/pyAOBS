/*
 * smesh.h - slowness mesh interface
 *
 * Jun Korenaga, MIT/WHOI
 * December 1998
 */

#ifndef _TOMO_SMESH_H_
#define _TOMO_SMESH_H_

#include <list>
#include <array.h> // from mconv source
#include <geom.h>
#include "heap_deque.h"
#include "index.h"

class Interface2d; // forward declaration

// -k<k> or -k<k_lid>/<k_below>  (lid = above conv, below = on/below conv)
bool parseVpVsKappa(const char* s, double& k_lid, double& k_below);

class SlownessMesh2d {
public:
    SlownessMesh2d(const char*); // file input

    void set(const Array1d<double>&);
    void get(Array1d<double>&) const;
    void vget(Array1d<double>&) const;
    // Dual field: get/set remain Vs (pgrid). These read/write frozen Vp.
    void setVp(const Array1d<double>&);
    void getVp(Array1d<double>&) const;

    // PPS/PSS Vs inversion: freeze Vp, invert Vs (init Vs = Vp/kappa).
    // After this, get/set/outMesh/at/calc_ttime use the Vs field.
    void enableDualVs(double kappa);
    void enableDualVs(double k_lid, double k_below, const Interface2d* conv);
    void resetVsFromVpBelow(double kappa, const Interface2d& conv);
    // All non-water nodes: Vs = Vp/kappa (lid vs below). Dual field only.
    void resetVsFromVp(double k_lid, double k_below, const Interface2d* conv);
    void loadDualVs(const char* vsfn);
    bool hasDualVs() const { return dual_vs; }
    // Fold leftover (type 6, no -k): lid Vp, below Vs. Dual ignores this.
    void setMixKappa(double k);
    void setMixKappa(double k_lid, double k_below);
    double calc_ttime_vp(int node_src, int node_rcv) const;
    // Dual PSP P-leg leftover: lid Vp, on/below Vs. Prefer calc_ttime_psx.
    double calc_ttime_converse_dual(int node_src, int node_rcv,
				   const Interface2d& conv) const;
    // PSP/PSX graph edges: both ends the same phase. No (uP+uS)/2.
    // want_s selects S vs P; the node/leg side is in calc_ttime_on psx_leg.
    double calc_ttime_psx(int node_src, int node_rcv, const Interface2d& conv,
			 bool want_s) const;
    double calc_ttime_psx(int node_src, int node_rcv, const Interface2d& conv,
			 int psx_mode) const;
    double atVp(const Point2d& pos) const;
    double atVp(const Point2d& pos, Index2d& guess) const;
    double atVp(const Point2d& pos, Index2d& guess,
		double& dudx, double& dudz) const;
    // PSP/PSX bending: bilinear, same phase at every corner.
    // sample_below from the point: lid vs basement of that phase.
    double atPsx(const Point2d& pos, const Interface2d& conv, bool want_s,
		 bool sample_below, Index2d& guess) const;
    double atPsx(const Point2d& pos, const Interface2d& conv, bool want_s,
		 bool sample_below, Index2d& guess, double& dudx, double& dudz) const;
    void outMeshVp(ostream&) const;
    
    // general bookkeeping functions
    int numNodes() const { return nnodes; }
    int Nx() const { return nx; }
    int Nz() const { return nz; }
    double xmin() const;
    double xmax() const;
    double zmin() const;
    double zmax() const;
    const Index2d& nodeIndex(int i) const;
    int nodeIndex(int i, int k) const;
    Point2d nodePos(int i) const;

    // slowness interpolation
    double at(const Point2d& pos) const;
    double at(const Point2d& pos, Index2d& guess) const;
    double at(const Point2d& pos, Index2d& guess,
	      double& dudx, double& dudz) const;
    double atWater() const { return p_water; }

    // for graph traveltime calculation
    double calc_ttime(int node_src, int node_rcv) const; 
									   
    // cell-oriented functions
    int numCells() const { return ncells; }
    void cellNodes(int, int&, int&, int&, int&) const;
    int cellIndex(int i, int k) const { return index2cell(i,k); }
    void cellGradientKernel(int, Array2d<double>&, double, double) const;
    void cellNormKernel(int, Array2d<double>&) const;
    int locateInCell(const Point2d&, Index2d&,
		     int&, int&, int&, int&,
		     double&, double&, double&, double&) const;
    void nodalCellVolume(Array1d<double>&,
			 Array1d<double>&, Array1d<Point2d>&) const; // for gravity
    
    // miscellaneous
    int nearest(const Point2d& src) const;
    void nearest(const Interface2d& itf, Array1d<int>& inodes) const;
    bool inWater(const Point2d& pos) const;
    bool inAir(const Point2d& pos) const;

    // output functions
    void outMesh(ostream&) const;
    void printElements(ostream&) const;
    void printVGrid(ostream&,bool) const;
    void printVGrid(ostream&,
		    double,double,double,double,double,double) const;
    void printMaskGrid(ostream&, const Array1d<int>&) const;
    void printMaskGrid(ostream&, const Array1d<double>&) const;

    friend class Interface2d;
    
private:
    double calc_ttime_on(int node_src, int node_rcv,
			const Array2d<double>& pg,
			const Interface2d* mix_conv=0, int psx_leg=0) const;
    double p_on_node(int i, int k, const Array2d<double>& pg,
		     const Interface2d* mix_conv, int psx_leg=0) const;
    // Mixed fold: lid stores Vp, below stores Vs. Reconstruct the other
    // phase at this node (lid S = Vp/kappa, below P = Vs*kappa). Dual: the
    // dedicated grid. Conversion nodes then match dual without hopping a cell.
    double psx_node_phase_vel(int i, int k, const Interface2d& conv,
			      bool want_s) const;
    double psx_corner_vel(int i, int k, const Interface2d& conv,
			  bool want_s, bool sample_below) const;
    double atPsx_interp(const Point2d& pos, const Interface2d& conv,
			bool want_s, bool sample_below, Index2d& guess,
			double* dudx, double* dudz) const;
    void upperleft(const Point2d& pos, Index2d& guess) const;
    void calc_local(const Point2d& pos, int, int,
		    double&, double&, double&, double&) const;
    void commonGradientKernel();
    void commonNormKernel();
    bool in_water(const Point2d& pos, const Index2d& guess) const;
    bool in_air(const Point2d& pos, const Index2d& guess) const;
//    double almost_exact_ttime(double v1, double v2,
//			      double dpath) const;

    int nx, nz, nnodes, ncells;
    double p_water; /* slowness of water column */
    double p_air; /* slowness of air column */
    Array2d<double> pgrid, vgrid;
    Array2d<double> pgrid_vp, vgrid_vp;
    bool dual_vs;
    double mix_kappa;
    double mix_kappa_below;
    Array2d<int> ser_index, index2cell;
    Array1d<Index2d> node_index;
    Array1d<int> cell_index;
    Array1d<double> xpos, topo, zpos;
    Array1d<double> rdx_vec, rdz_vec, b_vec;
    Array1d<double> dx_vec, dz_vec;
    Array2d<double> Sm_H1, Sm_H2, Sm_V, T_common;
    const double eps;
};

#endif /* _TOMO_SMESH_H_ */
