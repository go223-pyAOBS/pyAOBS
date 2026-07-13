/* Copyright (c) Colorado School of Mines, 1994.*/
/* Copyright (c) GEOMAR, Kiel Germany, 1995.*/
/* All rights reserved.                       */

/* SUPSWIGBX2: $Revision: 1.0 $ ; $Date: 95/01/13  	$		*/

#include "su.h"
#include "segy.h"

/*********************** self documentation *****************************/
char *sdoc[] = {
" 									",
" SUPSWIGBX2 - PostScript WIGgle-trace bitmap plot of a segy data set 	",
" 									",
" supswigbx2 <stdin >postscript file [optional parameters]		",
" 							        	",
" Optional parameters: 							",
"                                                                       ",
" key=offset 		Key header word to plot against 		",
" scale=0.001		Scaling factor multiplied to key value          ",
"			Default value set offset from meter to kilometer",
" unit=cm		scaling unix of plot axis (cm/inch)		",
" tscale=1 [unit/s]	Scale along time axis				",
" xscale=0.5 [unit/km]	Scale along X (distance) axis			",
"									",
" nbpi=300		Printer resolution, default 300 dpi		",
"									",
" d1=                                                                   ",
" f1=    								",
" 							        	",
" d1 is the sampling interval in the fast dimension.  If not getparred	",
" then for seismic time domain data d1=tr.dt/10^6 if set, else 0.010. 	",
" For other types of data d1=tr.d1 if set, else 1.0			",
" 							        	",
" f1 is the first sample in the fast dimension.  If not getparred	",
" then for seismic time domain data f1=tr.delrt/10^3 if set, else 0.0.	",
" For other types of data f1=tr.d1 if set else 0.0	 		",
" 							        	",
" Note that for seismic time domain data, the \"fast dimension\" is	",
" time and the \"slow dimension\" is usually trace number or range.	",
" 							        	",
" See the pswigb selfdoc for the remaining parameters.			",
" 							        	",
NULL};

/* Credits:
 *
 *	CWP: Dave (psimage), Jack & John (su tee shirt)
 *
 *      GEOMAR: Sanyu Ye, for unevenly space segy data set
 *
 */

/**************** end self doc *******************************************/

segy tr;


main(int argc, char **argv)
{
        cwp_String key;         /* header key word from segy.h          */
        cwp_String type;        /* type of key                          */
        int index;              /* index of key                         */
        Value val;              /* value of key                         */
        int  ival;              /* ... cast to int                      */
        char plotcmd[NALLOC];   /* build command for popen              */
        char x2str[NALLOC];     /* build string for x2 vector           */
        char valstr[BUFSIZ];    /* string to hold a key value           */
	char *style="seismic";  /* layout style	seismic t-ver, x2-horiz */ 
 	float *trbuf;		/* trace buffer			 	*/
	FILE *datafp;		/* fp for trace data file		*/
	FILE *plotfp;		/* fp for plot data			*/
        float scale;            /* factor to be multiplied to key value */
        char *unit="cm";        /* scaling unit of plot axis            */
        float x1beg, x1end;     /* plot direction along time axis	*/
        float tscale;           /* scale along time axis		*/
        float xscale;           /* scale along x (distance) axis	*/
        float xcur;             /* wiggle excursion			*/
	float d1;		/* time/depth sample rate 		*/
	float f1;		/* tmin/zmin				*/
	float wbox,hbox;	/* width and hight of image flame	*/
	int nt;			/* number of samples on trace		*/
	int ntr;		/* number of traces			*/
	int nbpi;		/* printer resolution 			*/
	int xmin,xmax;		/* minimu and maximum key value 	*/
	cwp_Bool seismic;	/* is this seismic data?		*/


	/* Initialize */
	initargs(argc, argv);
	requestdoc(1);
	

	/* Get info from first trace */
	if (!gettr(&tr)) err("can't get first trace");
	seismic = ISSEISMIC(tr.trid);
		 
	nt = tr.ns;

        /* Default parameters;  User-defined overrides */
        if (!getparstring("key"    , &key))     key = "offset";
        if (!getparfloat("scale", &scale))      scale = 0.001;
        if (!getparfloat("tscale", &tscale))    tscale = 1.0;
        if (!getparfloat("xscale", &xscale))    xscale = 0.5;
        if (!getparfloat("xcur", &xcur))    	xcur = 1.0;
        if (!getparint("nbpi", &nbpi))    	nbpi = 300;

	if (!getparfloat("d1", &d1)) {
		if (seismic) {
			if (tr.dt) {
				d1 = (float) tr.dt / 1000000.0;
			} else {
				d1 = 0.01;
				warn("tr.dt not set, assuming dt=0.010");
			}
		} else { /* non-seismic data */
			if (tr.d1) {
				d1 = tr.d1;
			} else {
				d1 = 1.0;
				warn("tr.d1 not set, assuming d1=1.0");
			}
		}
	}

	if (!getparfloat("f1", &f1)) {
		if (seismic) {
			f1 = (tr.delrt) ? (float) tr.delrt/1000.0 : 0.0;
		} else {
			f1 = (tr.f1) ? tr.f1 : 0.0;
		}
	}

        type = hdtype(key);
        index = getindex(key);


	/* Allocate trace buffer */
	trbuf = ealloc1float(nt);

	/* initialize the max and min key value */
	gethval(&tr, index, &val);
        ival = vtoi(type, val);
	xmax=xmin=ival;	

	/* Create temporary "file" to hold data */
	datafp = etmpfile();

	/* Loop over input traces & put them into the psdata file */
	ntr = 0;
	do {
		++ntr;
		efwrite(tr.data, FSIZE, nt, datafp);

                gethval(&tr, index, &val);
                ival = vtoi(type, val);
                sprintf(valstr, "%.2f,", scale*ival);
                strcat(x2str,valstr);
		xmax = MAX(xmax, ival);
		xmin = MIN(xmin, ival);

	} while (gettr(&tr));

        x2str[strlen(x2str) - 1] = NULL;        /* delete last comma */

        getparstring("style", &style);
	if(STREQ("seismic",style)) {
		wbox=xscale*fabs(scale*(xmax-xmin))*(1.0 + 2.0*xcur/(ntr-1));
		hbox=tscale*fabs(d1)*nt;
	}
	else {
		hbox=xscale*fabs(scale*(xmax-xmin))*(1.0 + 2.0*xcur/(ntr-1));
		wbox=tscale*fabs(d1)*nt;
	}

        getparstring("unit", &unit);
	if( !STREQ(unit, "inch") ) {
	    hbox=hbox/2.54;
	    wbox=wbox/2.54;
	}

	/* plot time axis default upwards */
	if (!getparfloat("x1end", &x1end)) 	x1end=f1;
	if (!getparfloat("x1beg", &x1beg)) 	x1beg=f1 + d1*nt;

	/* System call to pswigb */
        sprintf(plotcmd, "pswigb n1=%d n2=%d d1=%f f1=%f x2=%s wbox=%.3f hbox=%.3f \
		nbpi=%d, x1beg=%.3f x1end=%.3f",
        	nt, ntr, d1, f1, x2str, wbox, hbox, nbpi, x1beg, x1end);

	for (--argc, ++argv; argc; --argc, ++argv) {
		if (strncmp(*argv, "d1=", 3) && /* skip those already set */
		    strncmp(*argv, "f1=", 3) &&
		    strncmp(*argv, "d2=", 3) &&
		    strncmp(*argv, "f2=", 3) &&
		    strncmp(*argv, "nbpi=", 5)) {
		    
			strcat(plotcmd, " ");   /* put a space between args */
			strcat(plotcmd, "\"");  /* user quotes are stripped */
			strcat(plotcmd, *argv); /* add the arg */
			strcat(plotcmd, "\"");  /* user quotes are stripped */
		}
	}


	/* Open pipe; read data to buf; write buf to plot program */
	plotfp = epopen(plotcmd, "w");
	rewind(datafp);
	{ register int itr;
		for (itr = 0; itr < ntr; ++itr) {
			efread (trbuf, FSIZE, nt, datafp);
			efwrite(trbuf, FSIZE, nt, plotfp);
		}
	}


	/* Clean up */
	epclose(plotfp);
	efclose(datafp);


	return EXIT_SUCCESS;
}
