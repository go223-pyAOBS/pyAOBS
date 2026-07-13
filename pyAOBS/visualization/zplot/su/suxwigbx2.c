/* Copyright (c) Colorado School of Mines, 1994.*/
/* Copyright (c) GEOMAR, Kiel Germany, 1995.*/
/* All rights reserved.                       */


/*----------------------------------------------------------------------
 *
 * This code is part of SU.  SU stands for Seismic Unix, a processing line
 * developed at the Colorado School of Mines, partially based on Stanford
 * Exploration Project (SEP) software.  Inquiries should be addressed to:
 *
 *  Jack K. Cohen, Center for Wave Phenomena, Colorado School of Mines,
 *  Golden, CO 80401  (jkc@dix.mines.colorado)
 *----------------------------------------------------------------------
 *  Modified by Sanyu Ye, GEOMAR, University Kiel (sye@geomar.de)
 *----------------------------------------------------------------------
 */

#include "su.h"
#include "segy.h"

/*********************** self documentation *****************************/
char *sdoc[] = {
" 									",
" SUXWIGBX2 - X WIGgle-trace Bitmap plot of a segy data set	 	",
"		unevenly spaced in slow dimension X2			",
" 									",
" suxwigbx2 <stdin file [optional parameters]				",
" 							        	",
" Optional parameters: 							",
" 							        	",
"	key=offset	Key header word to plot against 		",
"	scale=0.001	Scaling factor multiplied to key value 		",
"	d1=								",
"	f1=								",
"									",
"	pick=disable	X-picker (enable/disable) (not implemented yet) ",
"									",
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
" See the xwigb selfdoc for the remaining parameters and X functions.	",
" 							        	",
NULL};
/**************** end self doc *******************************************/

/* Credits:
 *
 *	CWP: Dave (xwigb), Jack & John (su tee shirt)
 *
 *	GEOMAR: Sanyu Ye, for unevenly space segy data set
 *
 * Notes: See notes for suximage.
 */


segy tr;


main(int argc, char **argv)
{
	cwp_String key; 	/* header key word from segy.h          */
        cwp_String type;	/* type of key                          */
        char *pick="disable";	/* key to enable/disable pick on display*/
        int index;      	/* index of key                         */
        Value val;      	/* value of key                         */
        int  ival;       	/* ... cast to int                      */
	char plotcmd[NALLOC];	/* build command for popen	 	*/
	char x2str[NALLOC];	/* build string for x2 vector		*/
	char valstr[BUFSIZ];	/* string to hold a key value		*/
	float *trbuf;		/* trace buffer			 	*/
	FILE *datafp;		/* fp for trace data file		*/
	FILE *plotfp;		/* fp for plot data			*/
        float x1beg, x1end;     /* plot direction along time axis       */
	float scale; 		/* factor to be multiplied to key value */
	float d1;		/* time/depth sample rate 		*/
	float f1;		/* tmin/zmin				*/
	int nt;			/* number of samples on trace		*/
	int ntr;		/* number of traces			*/
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
	if (!getparfloat("scale", &scale)) 	scale = 0.001;

	if (!getparfloat("d1", &d1)) {
		if (seismic) {
			if (tr.dt) {
				d1 = (float) tr.dt / 1000000.0;
			} else {
				d1 = 0.004;
				warn("tr.dt not set, assuming dt=0.004");
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


	/* Create temporary "file" to hold data */
	datafp = etmpfile();


	/* Loop over input traces & put them into the data file */
	ntr = 0;
	do {
		++ntr;
		efwrite(tr.data, FSIZE, nt, datafp);

        	gethval(&tr, index, &val);
		ival = vtoi(type, val);
	        sprintf(valstr, "%.2f,", scale*ival);
		strcat(x2str,valstr);

	} while (gettr(&tr));

	x2str[strlen(x2str) - 1] = NULL; 	/* delete last comma */

        /* plot time axis default upwards */
        if (!getparfloat("x1end", &x1end))      x1end=f1;
        if (!getparfloat("x1beg", &x1beg))      x1beg=f1 + d1*nt;

	/* Set up xwigb command line */
        getparstring("pick", &pick);
	if ( STREQ(pick, "enable") )
	    sprintf(plotcmd, "newxwigb n1=%d n2=%d d1=%f f1=%f x2=%s \
			x1beg=%.3f x1end=%.3f",
			nt, ntr, d1, f1, x2str, x1beg, x1end);
	else
	    sprintf(plotcmd, "xwigb n1=%d n2=%d d1=%f f1=%f x2=%s \
                        x1beg=%.3f x1end=%.3f",
			   nt, ntr, d1, f1, x2str, x1beg, x1end);

	for (--argc, ++argv; argc; --argc, ++argv) {
		if (strncmp(*argv, "d1=", 3) && /* skip those already set */
		    strncmp(*argv, "f1=", 3)) {
		    
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
