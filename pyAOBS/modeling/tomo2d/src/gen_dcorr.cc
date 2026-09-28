/*
 * gen_dcorr.cc
 *
 * usage: gen_dcorr [ options ]
 *
 * 写出 tt_inverse -CD 所用的 1D 相关长度（CorrelationLength1d）：
 *   每行  x(km)  Lh(km)
 *
 * <均匀>
 *     -ALh -Dxmin/xmax [ -Nnx ]
 *     缺 -N 时只写两端点（与手工 dcorr 两行文件相同）
 *
 * <Zelt 划区，原理同 gen_vcorr 的水平相关长度>
 *     -Aanorm/norm -Cv.in/ilayer -Edx [ -Fitop/ibot ] [ -Rrefl ]
 *     沿界面 x 取样；若深度落在 -F 上下界面之间则用 anorm，否则 norm。
 *     有 -R 时用反射面节点；否则用 Zelt 层 ilayer 的地形节点。
 *
 * <从二维 vcorr 取样，即 readme 3.3：无 -CD 时程序在反射点上取速度水平相关长度>
 *     -Vcorr_v_fn -Rrefl_file
 *
 * Jun Korenaga 相关长度约定 / pyAOBS gen_dcorr
 */

#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <array.h>
#include <util.h>
#include <geom.h>
#include "zeltform.h"
#include "corrlen.h"
#include "error.h"

using namespace std;

static double interp_at(const Array1d<double>& xs, const Array1d<double>& vs, double x)
{
    int n = xs.size();
    if (n < 1) return 0.0;
    if (n == 1 || x <= xs(1)) return vs(1);
    if (x >= xs(n)) return vs(n);
    for (int i = 1; i < n; i++){
	if (x >= xs(i) && x <= xs(i + 1)){
	    double den = xs(i + 1) - xs(i);
	    double r = (den == 0.0) ? 0.0 : (x - xs(i)) / den;
	    return vs(i) + r * (vs(i + 1) - vs(i));
	}
    }
    return vs(n);
}

static void read_xy_file(const char* fn, Array1d<double>& x, Array1d<double>& z)
{
    int n = countLines(fn);
    if (n < 1) error("gen_dcorr: empty file");
    x.resize(n);
    z.resize(n);
    ifstream in(fn);
    if (!in) error("gen_dcorr: cannot open file");
    for (int i = 1; i <= n; i++){
	if (!(in >> x(i) >> z(i))){
	    error("gen_dcorr: expected two columns (x z or x Lh-source)");
	}
    }
}

int main(int argc, char **argv)
{
    bool getA = false, getD = false, getN = false, getE = false;
    bool getV = false, getR = false, readZelt = false, dampfl = false;
    bool err = false;
    bool twoA = false;
    double lh = 0.0, anorm = 0.0, norm = 0.0;
    double xmin = 0.0, xmax = 0.0, dx = 0.0;
    int nx = 2, ilayer = 0, itop = 0, ibot = 0;
    char zeltfn[MaxStr];
    char *vcorrfn = 0, *reflfn = 0;

    for (int i = 1; i < argc; i++){
	if (argv[i][0] == '-'){
	    switch (argv[i][1]){
	    case 'A':
		if (sscanf(&argv[i][2], "%lf/%lf", &anorm, &norm) == 2){
		    getA = true;
		    twoA = true;
		}else if (sscanf(&argv[i][2], "%lf", &lh) == 1){
		    getA = true;
		    twoA = false;
		    anorm = lh;
		    norm = lh;
		}else{
		    cerr << "invalid -A option; give Lh or anorm/norm\n";
		    err = true;
		}
		break;
	    case 'D':
		getD = true;
		if (sscanf(&argv[i][2], "%lf/%lf", &xmin, &xmax) != 2){
		    xmax = atof(&argv[i][2]);
		    xmin = 0.0;
		}
		break;
	    case 'N':
		getN = true;
		nx = atoi(&argv[i][2]);
		break;
	    case 'C':
		if (sscanf(&argv[i][2], "%[^/]/%d", zeltfn, &ilayer) == 2){
		    readZelt = true;
		}else{
		    cerr << "invalid -C option\n";
		    err = true;
		}
		break;
	    case 'F':
		if (sscanf(&argv[i][2], "%d/%d", &itop, &ibot) == 2){
		    dampfl = true;
		}else{
		    cerr << "invalid -F option; give TOP and BOTM layer\n";
		    err = true;
		}
		break;
	    case 'E':
		getE = true;
		dx = atof(&argv[i][2]);
		break;
	    case 'V':
		getV = true;
		vcorrfn = &argv[i][2];
		break;
	    case 'R':
		getR = true;
		reflfn = &argv[i][2];
		break;
	    default:
		err = true;
		break;
	    }
	}else{
	    err = true;
	}
    }

    bool fromVcorr = getV && getR;
    if (getV != getR){
	cerr << "-V and -R must be used together to sample 2D vcorr.\n";
	err = true;
    }
    if (fromVcorr){
	/* ok */
    }else if (readZelt){
	if (!getA || !twoA){
	    cerr << "zelt mode needs -Aanorm/norm\n";
	    err = true;
	}
	if (!getE){
	    cerr << "zelt mode needs -Edx\n";
	    err = true;
	}
    }else if (getA && getD){
	if (twoA){
	    cerr << "-Aanorm/norm needs zelt -C/-E (or use a single -ALh)\n";
	    err = true;
	}
	if (getN && nx < 2){
	    cerr << "-N must be >= 2\n";
	    err = true;
	}
	if (xmax <= xmin){
	    cerr << "invalid -D range\n";
	    err = true;
	}
    }else{
	err = true;
    }
    if (err) error("usage: gen_dcorr [ -options ]");

    if (fromVcorr){
	CorrelationLength2d cv(vcorrfn);
	Array1d<double> x, z;
	read_xy_file(reflfn, x, z);
	int n = x.size();
	for (int i = 1; i <= n; i++){
	    double Lh, Lv;
	    cv.at(Point2d(x(i), z(i)), Lh, Lv);
	    cout << x(i) << " " << Lh << '\n';
	}
	return 0;
    }

    Array1d<double> x, z, x_top, x_bot, damptop, dampbot;
    if (readZelt){
	ZeltVelocityModel2d zelt(zeltfn);
	if (dampfl){
	    zelt.getTopo(itop, dx, x_top, damptop);
	    zelt.getTopo(ibot, dx, x_bot, dampbot);
	}
	if (getR){
	    read_xy_file(reflfn, x, z);
	}else{
	    Array1d<double> topo;
	    zelt.getTopo(ilayer, dx, x, topo);
	    z.resize(x.size());
	    for (int i = 1; i <= x.size(); i++) z(i) = topo(i);
	}
	int n = x.size();
	for (int i = 1; i <= n; i++){
	    double Lh = norm;
	    if (dampfl){
		double zt = interp_at(x_top, damptop, x(i));
		double zb = interp_at(x_bot, dampbot, x(i));
		if (z(i) > zt && z(i) < (zb - 0.10)) Lh = anorm;
	    }
	    cout << x(i) << " " << Lh << '\n';
	}
	return 0;
    }

    if (!getN) nx = 2;
    if (nx < 2) nx = 2;
    double step = (xmax - xmin) / (nx - 1);
    for (int i = 0; i < nx; i++){
	double xi = xmin + i * step;
	cout << xi << " " << lh << '\n';
    }
    return 0;
}
