/*
 * inverse.h
 *
 * Jun Korenaga, MIT/WHOI
 * January 1999
 */

#ifndef _TOMO_INVERSE_H_
#define _TOMO_INVERSE_H_

#include <iostream>
#include <fstream>
#include <map>
#include <array.h>
#include "smesh.h"
#include "graph.h"
#include "betaspline.h"
#include "bend.h"
#include "sparse_rect.h"
#include "interface.h"
#include "corrlen.h"
#include "jgrav.h"

class TomographicInversion2d {
public:
    TomographicInversion2d(SlownessMesh2d& m, const char* datafn,
			   int xorder, int zorder, double crit_len,
			   int nintp, double cg_tol, double br_tol);
    void solve(int niter);
    void doRobust(double);
    void removeOutliers();
    void setLSQR_TOL(double);
    void addRefl(Interface2d* intfp);
    void addSeafloor(Interface2d* intfp);
    void invertWaterOnly();
    void invertCrustOnly();
    void freezeRefl();
    void doFullRefl();
    void setReflWeight(double);
    
    void SmoothVelocity(const char*, double, double, double,
			bool logscale=false);
    void SmoothVelocityVs(double w);
    void SmoothDepth(double, double, double,
		     bool logscale=false);
    void SmoothDepth(const char*, double, double, double,
		     bool logscale=false);
    void applyFilter(const char*);
    void DampVelocity(double);
    void DampDepth(double);
    void FixDamping(double, double);
    void Squeezing(const char*);
    void targetChisq(double);
    void doJumping();
    void addonGravity(double, const Array1d<double>&, const Array1d<double>&,
		      AddonGravityInversion2d*, double);
    void addonGravity(double, const Array1d<double>&, const Array1d<double>&,
		      AddonGravityInversion2d*, double, const char*);

    void outStepwise(const char*, int level);
    void outFinal(const char*, int level);
    void setLogfile(const char*);
    void outMask(const char*);
    void setVerbose(int);

    void doConvert(Interface2d* convtp);
    void setKappa(double k);
    void setKappa(double k_lid, double k_below);
    void enableTraveltimeDiff();
    void enableFreezeLid();
    void enablePssBelowOnly();
    void enableFreezeBelow();
    void enableStrategy();

private:
    typedef Array1d< map<int,double> > sparseMat;

    void read_file(const char* ifn);

    void reset_kernel();
    void add_kernel(int, const Array1d<Point2d>&);
    void add_kernel_refl(int, const Array1d<Point2d>&, int, int);
    void calc_averaging_matrix();
    void calc_damping_matrix();
    void calc_refl_averaging_matrix();
    void calc_refl_damping_matrix();
    void add_global(const Array2d<double>&, const Array1d<int>&,
		    sparseMat&);
    void calc_sens_weights();
    double sensVelW(int inode, bool for_vs) const;
    double sensDepW(int inode) const;
    double sensCouple(double wi, double wj) const;
    void assignSensWeights(const Array1d<double>& s, Array1d<double>& w,
			   bool for_vs, bool joint_vp, double med_out[3]);
    void applyDmodel(double alpha,
		     const Array1d<double>& base_v,
		     const Array1d<double>& base_vp,
		     const Array1d<double>& base_d,
		     bool write_out);
    void restoreModel(const Array1d<double>& base_v,
		      const Array1d<double>& base_vp,
		      const Array1d<double>& base_d);
    void writeCurrentModels();
    int _solve(bool,double,bool,double,bool,double,bool,double);
    void auto_damping(int&, int&, double&, double&);
    void fixed_damping(int&, int&, double&, double&);
    double auto_damping_depth(double, int&, int&);
    double auto_damping_vel(double, int&, int&);
    double calc_ave_dmv();
    double calc_ave_dmd();
    void calc_Lm(double&, double&, double&);
    double calc_chi();
    const Interface2d* waterBottom() const;
    bool kernelRestricted() const;
    bool kernelAllowNode(int inode) const;
    bool kernelRestrictedVp() const;
    bool kernelAllowNodeVp(int inode) const;
    int nVelUnknowns() const;
    int convFaceDomain(int inode) const;
    void fill_vel_averaging(sparseMat& Rh, sparseMat& Rv,
			    const Array1d<double>& scale, bool for_vs);
    void fill_vel_damping(sparseMat& Tmat, bool for_vs);
    // 0=no split; 1=lid (strictly above conv); 2=on/below conv.
    // PSS/PPS invert both, but smooth/damp must not cross conv.
    int psxVelDomain(int inode) const;

    SlownessMesh2d& smesh;
    GraphSolver2d graph;
    BetaSpline2d betasp;
    BendingSolver2d bend;
    
    Array1d<Point2d> src;
    Array1d< Array1d<Point2d> > rcv;
    Array1d< Array1d<int> > raytype;
    Array1d< Array1d<double> > obs_ttime;
    Array1d< Array1d<double> > obs_dt;
    Array1d< Array1d<double> > res_ttime;
    Array1d<double> r_dt_vec, path_wt;
    Array1d<double> path_length;

    int nrefl;
    Array1d<int> start_i, end_i;
    Array1d<const Interface2d*> interp;
    const Interface2d *bathyp;
    Interface2d *reflp,*convp,*seafloorp;
    double refl_weight;
    bool do_full_refl,do_convert,freeze_refl;
    bool invert_water_only, invert_crust_only;
    bool invert_vs_psx;
    bool invert_joint_vpvs; // unused for staged joint (never [Vp|Vs] LSQR)
    bool joint_staged; // PPP then Vs-only; each stage is single-field
    int joint_vp_only_iters; // joint: first N iters update Vp only
    bool freeze_psx_lid; // lid frozen in kernel/smooth/damp/dm (PSP-only, or TOMO2D_INV_FREEZE_LID)
    bool freeze_below; // PPP lid-only: freeze on/below conv in kernel/smooth/damp/dm
    bool pss_below_only; // type 8 writes below S only; lid stays free (PPS)
    double vpvs_kappa;
    double vpvs_kappa_below;
    bool do_ttdiff; // PPS−PPP (+ PSP pairs if type 6 present); freeze Vp
    bool ttdiff_skip_pss_abs; // default false: PSS abs stays in A
    bool ttdiff_pps_only; // strategy stage 2: only PPS−PPP
    int ndata_abs;
    struct TtDiffPair { int isrc, ia, ib, gid_a, gid_b; };
    Array1d<TtDiffPair> ttdiff_pairs;
    void setupTraveltimeDiffRows();

    // Auto PPP → lid Vs → far-offset PSS→PSP corr → below Vs.
    bool do_strategy;
    bool strategy_force; // -ts
    bool strategy_full_vp; // stage 1: ignore conv mask (still honor -w)
    bool strategy_skip_update; // forward-only iter before stage 3
    bool strategy_rescale;
    int strategy_stage; // 0 off, 1 PPP, 2 lid, 3 below
    double strategy_dstar;
    int strategy_n_pick, strategy_n_corr;
    Array1d< Array1d<int> > ray_use; // 1=trace/invert this row
    Array1d< Array1d<int> > ray_kind; // 0=observed, 1=PSP placeholder
    void strategyEnsurePspSlots();
    void strategyRebuildDataArrays();
    void strategySetRayUseByCode(int c0, int c1=-1, int c2=-1);
    void configureStrategyStage(int stage);
    void strategyAfterPpp();
    void applyStrategyCorr();
    bool strategyRayOn(int isrc, int ircv) const;

    int nnodev, nnoded, ndata, ndata_valid, nx, nz;
    double rms_tres[2], init_chi[2], rms_tres_total, init_chi_total;
    int ndata_in[2];
    bool robust;
    double crit_chi;
    sparseMat A, Rv_h, Rv_v, Rd, Tv, Td;
    sparseMat Rv_h_vs, Rv_v_vs, Tv_vs;
    Array1d<double> data_vec, total_data_vec;
    Array1d<double> modelv, modelvp, modeld, dmodel_total;
    int itermax_LSQR;
    double LSQR_ATOL;

    bool jumping;
    Array1d<double> dmodel_total_sum, mvscale, mvscale_vp, mdscale;
    bool sens_weight;
    double sens_kappa, sens_eps;
    Array1d<double> sens_w_vp, sens_w_vs, sens_w_d;
    bool line_search;
    double ls_c, ls_rho, ls_amin;
    bool use_lm;
    double lm_lambda, lm_up, lm_down, lm_lmin, lm_lmax;
    double lm_rho_accept, lm_rho_good;

    bool smooth_velocity, logscale_vel;
    double wsv_min, wsv_max, dwsv;
    double wsv_vs; // <0: Vs uses weight_s_v. Joint: -Ss
    double weight_s_v, weight_s_vs;
    CorrelationLength2d *corr_vel_p;
    bool do_filter;
    Interface2d *uboundp;

    bool smooth_depth, logscale_dep;
    double wsd_min, wsd_max, dwsd;
    double weight_s_d;

    CorrelationLength1d *corr_dep_p;
    bool damp_velocity, damp_depth;
    double target_dv, target_dd;
    bool damping_is_fixed; 
    double weight_d_v, weight_d_d;
    bool do_squeezing;
    DampingWeight2d *damping_wt_p;

    int nnode_total;
    Array1d<int> nodev_hit;
    Array1d<int> tmp_node, tmp_nodev, tmp_nodev_vs, tmp_data;
    Array1d<int> tmp_nodedc, tmp_nodedr;
    bool out_mask;
    Array1d<double> dws;
    ofstream* vmesh_os_p;

    double target_chisq;
    
    bool printLog;
    ofstream* log_os_p;
    int verbose_level;
    bool printTransient, printFinal;
    const char* out_root;
    int out_level;
    int dump_iter, dump_iset;
    bool dump_is_final;

    bool gravity, out_grav_dws;
    AddonGravityInversion2d *ginv;
    int ngravdata;
    sparseMat B;
    double weight_grav, grav_z0, rms_grav;
    Array1d<int> tmp_gravdata;
    Array1d<double> grav_x, obs_grav, res_grav;
    ofstream *grav_dws_osp;
    
    // variables of temporary use
    Array1d<Point2d> path, Q;
    Array1d<const Point2d*> pp;

    ofstream* dump_os_p;

};

#endif /* _TOMO_INVERSE_H_ */
