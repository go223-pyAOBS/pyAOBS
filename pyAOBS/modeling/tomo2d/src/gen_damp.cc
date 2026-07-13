/*
 * gen_damp.cc
 *
 * usage: gen_damp [ options ]
 *
 * <velocity setting options>
 *     [uniform gradient]
 *             -Anormal/abnormal -Bgradient
 *     [Zelt's v.in]
 *             -Cv.in/ilayer [ -Filayer/blayer [ -d<dump> ]]
 *             
 * <grid generation options>
 *     [uniform spacing grid]:
 *             -Nnx/nz -Dxmax/zmax
 *     [variable spacing grid]:
 *             -Xxfile -Zzfile -Ttfile
 *     [Zelt-based grid]:
 *             -Edx -Zzfile 
 *
 * Jun Korenaga, MIT/WHOI
 * January 1999
 * BY Huang Haibo, MAY 2018
 */

#include <iostream>
#include <fstream>
#include <cstdio>
#include <list>
#include <cstdlib>
#include <array.h>
#include <util.h>
#include "zeltform.h"

int main(int argc, char **argv)
{
    bool getA=false, getB=false, readZelt=false, getN=false;
    bool getD=false, getE=false, getXfn=false, getZfn=false, getTopo=false;
    bool getWater=false, dampfl=false, err=false;
    bool zeltdump=false;
    double v0, vgrad, xmax, zmax, dx, norm_d,anorm_d,wcol=0.0;
    int nx, nz, ilayer, imoho, itop,ibot;
    char *xfn, *zfn, *tfn, zeltfn[MaxStr], reflfn[MaxStr];
    char *zdfn; 
    double v_water = 1.50,depth;
    double v_air = 0.330;
    
    for (int i=1; i<argc; i++){
	if (argv[i][0] == '-'){
	    switch(argv[i][1]){
	    case 'A':
		if (sscanf(&argv[i][2], "%lf/%lf", &anorm_d, &norm_d) != 2){
		    cerr << "invalid -A option;give abnormal and normal damp values\n";
		}
		break;
	    case 'C':
		if (sscanf(&argv[i][2], "%[^/]/%d",&zeltfn, &ilayer) == 2){
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
		    cerr << "invalid -F option;give TOP and BOTM damp layer\n";
		    err = true;
		}
		break;
	    case 'N':
		getN = true;
		if (sscanf(&argv[i][2], "%d/%d", &nx, &nz) != 2){
		    cerr << "invalid -N option\n";
		}
		break;
	    case 'D':
		getD = true;
		if (sscanf(&argv[i][2], "%lf/%lf", &xmax, &zmax) != 2){
		    cerr << "invalid -D option\n";
		}
		break;
	    case 'E':
		getE = true;
		dx = atof(&argv[i][2]);
		break;
	    case 'X':
		xfn = &argv[i][2];
		getXfn = true;
	    case 'Z':
		zfn = &argv[i][2];
		getZfn = true;
		break;
	    case 'd':
		zdfn = &argv[i][2];
		zeltdump = true;
		break;
	    case 'T':
		tfn = &argv[i][2];
		getTopo = true;
		break;
	    case 'W':
		getWater = true;
		wcol = atof(&argv[i][2]);
		break;
	    case 'Q':
		v_water = atof(&argv[i][2]);
		break;
	    case 'R':
		v_air = atof(&argv[i][2]);
		break;
	    default:
		err = true;
		break;
	    }
	}else{
	    err = true;
	}
    }

    int velopt=0, gridopt=0;
    bool uniGradient=false;
    bool uniGrid=false, varGrid=false;
    if (readZelt){ velopt++; gridopt++; } 
    if (getN && getD){ uniGrid=true; gridopt++; }
    if (getXfn && getZfn){ varGrid=true; gridopt++; }

    if (velopt!=1 || gridopt!=1){
	cerr << "too many options!\n";
	err=true;
    }
    if (getTopo && !varGrid){
	cerr << "for -T, also use -X & -Z.\n";
	err=true;
    }
    if (readZelt && !(getZfn && getE)){
	cerr << "incomplete Zelt options.\n";
	err=true;
    }
    if (err) error("usage: gen_smesh [ -options ]");

    Array1d<double> x, topo, z, x_top, x_bot,damptop,dampbot;
    ZeltVelocityModel2d* pzelt;
    if (uniGrid){
	x.resize(nx); topo.resize(nx); z.resize(nz);
	double dx = xmax/(nx-1);
	double dz = zmax/(nz-1);
	for (int i=1; i<=nx; i++) x(i) = (i-1)*dx;
	for (int i=1; i<=nz; i++) z(i) = (i-1)*dz;
    }else if (varGrid){
	nx = countLines(xfn);
	nz = countLines(zfn);
	x.resize(nx); topo.resize(nx); z.resize(nz);
	ifstream xin(xfn), zin(zfn);
	for (int i=1; i<=nx; i++) xin >> x(i);
	for (int i=1; i<=nz; i++) zin >> z(i);
	if (getTopo){
	    int ntopo = countLines(tfn);
	    if (ntopo != nx){
		error("gen_smesh::size mismatch for topo file");
	    }
	    ifstream tin(tfn);
	    for (int i=1; i<=nx; i++) tin >> topo(i);
	}
    }else if (readZelt){
	nz = countLines(zfn);
	z.resize(nz);
	ifstream zin(zfn);
	for (int i=1; i<=nz; i++) zin >> z(i);
	pzelt = new ZeltVelocityModel2d(zeltfn);
	if (dampfl){
	    pzelt->getTopo(itop, dx, x_top, damptop);
	    pzelt->getTopo(ibot, dx, x_bot, dampbot);
	}
	pzelt->getTopo(ilayer, dx, x, topo);
	nx = x.size();
	if (zeltdump) pzelt->dumpNodes(zdfn);
    }

    cout << nx << " " << nz << '\n';
    for (int i=1; i<=nx; i++)  cout << x(i) << " ";
    cout << '\n';
    for (int i=1; i<=nx; i++)  cout << topo(i) << " ";
    cout << '\n';
    for (int i=1; i<=nz; i++)  cout << z(i) << " ";
    cout << '\n';

    for (int i=1; i<=nx; i++){
	for (int k=1; k<=nz; k++){
	    double v;
	    if (readZelt){
		// note: this 'eps' is essential to stabilize
		// interpolation used for Zelt model.
		// (e.g., to skip null cells, to get subseafloor
		//  velocity not seawater velocity for the surface velocity)
		const double eps=1e-10;
		depth=topo(i)+z(k)+eps;
		v = norm_d;
		if(depth > damptop(i) && depth < (dampbot(i)-0.10)) v = anorm_d;
    }
	    cout << v << " ";
	}
	cout << '\n';
    }
}
