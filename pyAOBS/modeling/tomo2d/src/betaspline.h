/*
 * betaspline.h - beta spline interface
 *
 * Jun Korenaga, MIT/WHOI
 * January 1999
 */

#ifndef _TOMO_BETASPLINE_H_
#define _TOMO_BETASPLINE_H_

#include <list>
#include <array.h> // from mconv
#include <geom.h>

class BetaSpline2d {
public:
    BetaSpline2d(double beta1, double beta2, int nintp);

    int numIntp() const { return nintp; }
    void resetNIntp(int);
    void interpolate(const Point2d&, const Point2d&,
		     const Point2d&, const Point2d&,
		     Array1d<Point2d>&) const;
    void interpolate(const Point2d&, const Point2d&,
		     const Point2d&, const Point2d&,
		     Array1d<Point2d>&, Array1d<Point2d>&) const;

    // valid i range = -2,-1,0,1
    double coeff_b(int i, int iintp) const { return b[i+2](iintp); }
    double coeff_dbdu(int i, int iintp) const { return dbdu[i+2](iintp); }
    
private:
    void calc_c(double, double);
    void calc_u();
    void calc_B();
    void calc_dBdu();
    
    int nintp;
    Array1d<double> u, b[4], dbdu[4];
    double c[4][4];
};

// helper functions
void makeBSpoints(const Array1d<Point2d>& orig, Array1d<const Point2d*>& pp);
void makeBSpoints(const list<Point2d>& orig, Array1d<const Point2d*>& pp);
void printCurve(ostream& os,
		const Array1d<Point2d>& orig, const BetaSpline2d& bs);
void printCurve(ostream& os,
		const list<Point2d>& orig, const BetaSpline2d& bs);
// Independent splines on [1,iu1], [iu1,id0], [id1,np] so P/S hinges are not rounded.
void printCurve(ostream& os,
		const Array1d<Point2d>& orig, const BetaSpline2d& bs,
		int iu1, int id0, int id1);

class Interface2d;
void printCurve(ostream& os,
		const Array1d<Point2d>& orig, const BetaSpline2d& bs,
		const Interface2d& z_lo, const Interface2d& z_hi);
    
#endif /* _TOMO_BETASPLINE_H_ */
