/* Copyright (c) Colorado School of Mines, 1994.*/
/* All rights reserved.                       */

/* SURCHWN: $Revision: 1.0 $ ; $Date: 95/01/30 14:58:58 $		*/

#include "su.h"
#include "segy.h"

/*********************** self documentation **********************/
char *sdoc[] = {
"								",
" SURCHWN -change header word using one or two header word fields",
"        -change is applicable only between traces inidicated by",
"         e and f. Others remain unchanged                      ",
"								",
" surchw <stdin >stdout [optional parameters]			",
"								",
" Required parameters:						",
"	none							",
"								",
" Optional parameters:						",
"	key1=cdp	output key 				",
"	key2=cdp	input key  				",
"	key3=cdp	input key  				",
"	key4=none	range key  				",
"	a=0		overall shift 				",
"	b=1		scale on first input key 		",
"	c=0		scale on second input key 		",
"	d=1		overall scale 				",
"	e=-99999	start range 				",
"	f= 99999	end   range 				",
"								",
" The value of header word key1 is computed from the values of	",
" key2 and key3 by:						",
"								",
"	val(key1) = (a + b * val(key2) + c * val(key3)) / d	",
"								",
" Examples:							",
" Shift cdp numbers by -1:					",
"	surchw <data >outdata a=-1				",
"								",
" Shift cdp numbers by -1 for traces 102 - 203:			",
"	surchw <data >outdata a=-1 key4=ep e=102 f=203		",
"								",
" Add 1000 to tracr value:					",
" 	surchw key1=tracr key2=tracr a=1000 <infile >outfile	",
"								",
" We set the receiver point (gx) field by summing the offset	",
" and shot point (sx) fields and then we set the cdp field by	",
" averaging the sx and gx fields (we choose to use the actual	",
" locations for the cdp fields instead of the conventional	",
" 1, 2, 3, ... enumeration):					",
"	surchw <indata key1=gx key2=offset key3=sx b=1 c=1 |	",
"	surchw key1=cdp key2=gx key3=sx b=1 c=1 d=2 >outdata	",
"								",
NULL};

/* Credits:
 *	SEP: Einar
 *	CWP: Jack
 *
 *	GEOMAR: J. Bialas	apply change of header word to specific
 *				range of traces
 *
 * Caveat:
 *	The constants a, b, c, d, e, f are read in as doubles.
 *	It is implicitly assumed that the data types of the
 *	keys 1-3 are the same.
 */
/**************** end self doc ***********************************/


segy tr;

main(int argc, char **argv)
{
	cwp_String key1, key2, key3, key4;
	cwp_String type1;
	cwp_String type4;
	int index1, index2, index3, index4;
	int go, trace;
	Value val1, val2, val3;
	Value val4;
	double a, c, b, d, e, f;
	void changeval(cwp_String type1,
		Value *valp1, Value *valp2, Value *valp3,
		double a, double b, double c, double d);
	int checkrange(cwp_String type4, Value *valp4, double e, double f);


	/* Initialize */
	initargs(argc, argv);
	requestdoc(1);


	/* Get parameters */
	if (!getparstring("key1", &key1))	key1 = "cdp";
	if (!getparstring("key2", &key2))	key2 = "cdp";
	if (!getparstring("key3", &key3))	key3 = "cdp";
	if (!getparstring("key4", &key4))	key4 = "cdp";
	if (!getpardouble("a"   , &a))	a = 0;
	if (!getpardouble("b"   , &b))	b = 1;
	if (!getpardouble("c"   , &c))	c = 0;
	if (!getpardouble("d"   , &d))	d = 1;
	if (!getpardouble("e"   , &e))  e = -99999;
	if (!getpardouble("f"   , &f))  f =  99999;

	type1  = hdtype(key1);
	type4  = hdtype(key4);
	index1 = getindex(key1);
	index2 = getindex(key2); 
	index3 = getindex(key3);
	index4 = getindex(key4);

	while (gettr(&tr)) {
		gethval(&tr, index4, &val4);
		trace = vtoi (type4, val4);

/*
		if (key4 != "none") {
			go = checkrange(type4, &val4, e, f);
		}
		else if (key4 == "none") {
			go = TRUE;
		}
		if (go) {
*/

		if (e <= (float)trace && (float)trace <= f) {
			gethval(&tr, index2, &val2);
			gethval(&tr, index3, &val3);
			changeval(type1, &val1, &val2, &val3, a, b, c, d);
			puthval(&tr, index1, &val1);
		}
		puttr(&tr);
	}

	return EXIT_SUCCESS;
}

/*
checkrange(cwp_String type4, Value *valp4, double e, double f)
{
	Value g;
	switch (*type4) {
	case 's':
		err("can't check range by char header word");
	break;
	case 'h':
		g = valp4->h;
		if ((e <= g) && (g <= f)) return TRUE;
		if ((g < e)  || (f < g)) return FALSE;
	break;
	case 'u':
		g = valp4->h;
		if ((e <= g) && (g <= f)) return TRUE;
		if ((g < e)  || (f < g)) return FALSE;
	break;
	case 'l':
		g = valp4->h;
		if (e <= g && g <= f) return TRUE;
		if (g < e  || f < g) return FALSE;
	break;
	case 'v':
		g = valp4->h;
		if (e <= g && g <= f) return TRUE;
		if (g < e  || f < g) return FALSE;
	break;
	case 'i':
		g = valp4->h;
		if (e <= g && g <= f) return TRUE;
		if (g < e  || f < g) return FALSE;
	break;
	case 'p':
		g = valp4->h;
		if (e <= g && g <= f) return TRUE;
		if (g < e  || f < g) return FALSE;
	break;
	case 'f':
		g = valp4->h;
		if (e <= g && g <= f) return TRUE;
		if (g < e  || f < g) return FALSE;
	break;
	case 'd':
		g = valp4->h;
		if (e <= g && g <= f) return TRUE;
		if (g < e  || f < g) return FALSE;
	break;
	default:
		err("unknown type %s", type4);
	break;
	}

*/
void changeval(cwp_String type1, Value *valp1, Value *valp2, Value *valp3,
		double a, double b, double c, double d)
{
	switch (*type1) {
	case 's':
		err("can't change char header word");
	break;
	case 'h':
		valp1->h = (a + b * valp2->h + c * valp3->h)/d;
	break;
	case 'u':
		valp1->u = (a + b * valp2->u + c * valp3->u)/d;
	break;
	case 'l':
		valp1->l = (a + b * valp2->l + c * valp3->l)/d;
	break;
	case 'v':
		valp1->v = (a + b * valp2->v + c * valp3->v)/d;
	break;
	case 'i':
		valp1->i = (a + b * valp2->i + c * valp3->i)/d;
	break;
	case 'p':
		valp1->p = (a + b * valp2->p + c * valp3->p)/d;
	break;
	case 'f':
		valp1->f = (a + b * valp2->f + c * valp3->f)/d;
	break;
	case 'd':
		valp1->d = (a + b * valp2->d + c * valp3->d)/d;
	break;
	default:
		err("unknown type %s", type1);
	break;
	}
}
