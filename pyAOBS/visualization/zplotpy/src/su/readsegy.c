/* Copyright (c) Colorado School of Mines, 1996.*/
/* All rights reserved.                       */

/* SEGYREAD: $Revision: 1.45 $ ; $Date: 1995/10/18 15:36:44 $     */

#include "su.h"
#include "segy.h"
#include "tapesegy.h"
#include "tapebhdr.h"
#include "bheader.h"

#ifndef true
#define true 1
#endif
#ifndef false
#define false 0
#endif


/*********************** self documentation **********************/
char *sdoc[] = {
"                                                                       ",
" READSEGY - read an SEG-Y tape                                         ",
"                                                                       ",
" readsegy > stdout tape=        		                        ",
"                                                                       ",
" Required parameter:                                                   ",
"       tape=		input tape device or seg-y filename (see notes)	",
"                                                                       ",
" Optional parameters:                                                  ",
"	buff=1		for buffered device (9-track reel tape drive)	",
"			normaly also for 4mm DAT and 8mm EXABYTE drives	",
"			=0 ; for disk segy files			",
"	verbose=0	silent operation				",
"			=1 ; echo every 'vblock' traces			",
"	vblock=50	echo every 'vblock' traces under verbose option	",
"	hfile=header	file to store ebcdic block (as ascii)		",
"	bfile=binary	file to store binary block			",
"	over=0		quit if bhed format not equal 1, 2, or 3	",
"			=1 ; override and attempt conversion		",
"       conv=1		convert data to native format			",
"			=0 ; assume data is in native format		",
"	ns=bh.hns	number of samples (use if bhed ns wrong)	",
"	trmin=1		first trace to read				",
"	trmax=INT_MAX	last trace to read				",
"	endian=1	set =0 for little-endian machines(PC's,DEC,etc.)",
"	errmax=0	allowable number of consecutive tape IO errors	",
"	iobyte=0	no trailing 4-bytes integer of a record		", 
"			=1 ; trailing 4-bytes integer for disk segy file",
"			     written using unformatted Fortran I/O	",
"									",
"  Notes: Traditionally tape=/dev/rmt0.  However, in the modern world	",
"	tape device names are much less uniform.  The magic name can	",
"	often be deduced by \"ls /dev\".  Likely man pages with the	",
"	names of the tape devices are: \"mt\", \"sd\" \"st\".  Also	",
"	try \"man -k scsi\", \" man mt\", etc.	Sometimes \"mt status\"	",
"	will tell the device name.					",
"									",
"	For a seg-y diskfile use tape=filename.				",
"	Remark: a seg-y file is not the same as an su file.		",
"	A seg-y file consists of three parts: an ebcdic header,		",
"	a binary reel header, and the traces.  The traces are (usually)	",
"	in 32 bit IBM floating point format.  An SU file consists only	",
"	of the trace portion written in the native binary floats.	",
"									",
"	  type:	  sudoc readsegy   for further information		",
NULL};

/*
 * Note: If you have a tape with multiple sequences of ebcdic header,
 *	binary header,traces, use the device that
 *      invokes the no-rewind option and issue multiple segyread
 *      commands (making an appropriate shell script if you
 *      want to save all the headers).  Consider using >> if
 *      you want a single trace file in the end.  Similar
 *      considerations apply for multiple reels of tapes,
 *      but use the standard rewind on end of file.
 *                                              
 * Note: For buff=1 (default) tape is accessed with 'read', for buff=0
 *      tape is accessed with fread. We suggest that you try buff=1
 *      even with EXABYTE tapes.                                 
 * Caveat: may be slow on an 8mm streaming (EXABYTE) tapedrive
 * Warning: segyread or segywrite to 8mm tape is fragile. Allow sufficient
 *         time between successive reads and writes.
 * Warning: may return the error message "efclose: fclose failed"
 *      intermittently when segyreading/segywriting to 8mm (EXABYTE) tape
 *      even if actual segyread/segywrite is successful. However, this
 *      error message may be returned if your tape drive has a fixed 
 *      block size set.
 * Caution: When reading or writing SEG-Y tapes, the tape
 *      drive should be set to be able to read variable block length
 *      tape files.
 */

/* Credits:
 *      SEP: Einar Kjartannson
 *      CWP: Jack, Brian, Chris
 *         : John Stockwell (added 8mm tape stuff)
 * conv parameter added by:
 *      Tony Kocurko
 *      Department of Earth Sciences
 *      Memorial University of Newfoundland
 *      St. John's, Newfoundland
 * bhed format = 2,3 conversion by:
 *	Remco Romijn (Applied Geophysics, TU Delft)
 *	J.W. de Bruijn (Applied Geophysics, TU Delft)
 * iobyte added by:
 *	Sanyu Ye (sye@geomar.de, GEOMAR, University Kiel)
 *	for reading segy file written using unformatted Fortran I/O
 *--------------------------
 * Additional Notes:
 *      Brian's subroutine, ibm_to_float, which converts IBM floating
 *      point to IEEE floating point is NOT portable and must be
 *      altered for non-IEEE machines.  See the subroutine notes below.
 *
 *      A direct read by dd would suck up the entire tape; hence the
 *      dancing around with buffers and files.
 * 
 *      However, if you have created a SEGY data file by a direct read
 *      of a SEGY tape to a disk file, via dd, then use conv=0 to read
 *      the resulting file.
 * 
 */
/**************** end self doc ***********************************/

/* subroutine prototypes */
static void ibm_to_float(int from[], int to[], int n, int endian);
static void long_to_float(long from[], float to[], int n, int endian);
static void short_to_float(short from[], float to[], int n, int endian);
static void tapebhed_to_bhed(const tapebhed *tapebhptr, bhed *bhptr);
static void tapesegy_to_segy(const tapesegy *tapetrptr, segy *trptr, int ns);

tapesegy tapetr;
tapebhed tapebh;
segy tr;
bhed bh;

main(int argc, char **argv)
{
        char *tape;             /* name of raw tape device      */
        char *bfile;            /* name of binary header file   */
        char *hfile;            /* name of ascii header file    */

        int tapefd;             /* file descriptor for tape     */

        FILE *tapefp;           /* file pointer for tape        */
        FILE *binaryfp;         /* file pointer for bfile       */
        FILE *headerfp;         /* file pointer for hfile       */
        FILE *pipefp;           /* file pointer for popen write */

        size_t nsegy;    	/* size of whole trace in bytes		*/
	int i;			/* counter				*/
        int itr;                /* current trace number                 */
        int trmin;              /* first trace to read                  */
        int trmax;              /* last trace to read                   */
        int ns;                 /* number of data samples               */
        int over;               /* flag for bhed.float override         */
        int conv;               /* flag for data conversion		*/
        int verbose;            /* echo every ...			*/
        int vblock;		/* ... vblock traces with verbose=1	*/
        int buff;               /* flag for buffered/unbuffered device  */
        int endian;		/* flag for big=1 or little=0 endian	*/
	int errmax;		/* max consecutive tape io errors	*/
	int iobyte, bytepr;	/* flag for and trailing 4-byte integer	*/
	int errcount = 0;	/* counter for tape io errors		*/
        cwp_Bool nsflag;        /* flag for error in tr.ns              */
 
        char cmdbuf[BUFSIZ];    /* dd command buffer                    */
        char ebcbuf[EBCBYTES];  /* ebcdic data buffer                   */


        /* Initialize */
        initargs(argc, argv);
        requestdoc(0); /* stdin not used */


        /* Make sure stdout is a file or pipe */
        switch(filestat(STDOUT)) {
        case TTY:
                err("stdout can't be tty");
        break;
        case DIRECTORY:
                err("stdout must be a file, not a directory");
        break;
        case BADFILETYPE:
                err("stdout is illegal filetype");
        break;
        }

        /* Set filenames */
        MUSTGETPARSTRING("tape",  &tape);
        if (!getparstring("hfile", &hfile))     hfile = "header";
        if (!getparstring("bfile", &bfile))     bfile = "binary";

        
        /* Set parameters */
        if (!getparint("trmin", &trmin))        trmin = 1;
        if (!getparint("trmax", &trmax))        trmax = INT_MAX;
        if (!getparint("verbose", &verbose))    verbose = 0;
        if (!getparint("vblock", &vblock))	vblock = 50;
        if (!getparint("endian", &endian))      endian = 1;
	if (!getparint("errmax", &errmax))	errmax = 0;
	if (!getparint("iobyte", &iobyte))	iobyte = 0;
	if (!getparint("buff", &buff))		buff = 1;


        /* Override binary format value */
        if (!getparint("over", &over))          over = 0;

        /* Override conversion of IBM floating point data? */
        if (!getparint("conv", &conv))          conv = 1;


        /* Open files - first the tape */
        if (buff) tapefd = eopen(tape, O_RDONLY, 0444);
        else      tapefp = efopen(tape, "r");
        if (verbose) warn("tape opened successfully");

        /* - the ebcdic header file in ascii */
        headerfp = efopen(hfile, "w");
        if (verbose) warn("header file opened successfully");

        /* - the binary data file */
        binaryfp = efopen(bfile, "w");
        if (verbose) warn("binary file opened successfully");

        /* Read the ebcdic raw bytes from the tape into the buffer */
        if (buff) {
		if (-1 == read(tapefd, ebcbuf, EBCBYTES)) {
			if (verbose)
				warn("tape read error on ebcdic header");
			if (++errcount > errmax)
				err("exceeded maximum io errors");
		} else { /* Reset counter on successful tape IO */
			errcount = 0;
		}
	} else {
		if (iobyte) fread(&bytepr, 4, 1, tapefp);
                fread(ebcbuf, 1, EBCBYTES, tapefp);
		if (ferror(tapefp)) {
			if (verbose)
				warn("tape read error on ebcdic header");
			if (++errcount > errmax)
				err("exceeded maximum io errors");
			clearerr(tapefp);
		} else { /* Reset counter on successful tape IO */
			errcount = 0;
		}
		if (iobyte) fread(&bytepr, 4, 1, tapefp);
	}

        /* Open pipe to use dd to convert ascii to ebcdic */
        sprintf(cmdbuf, "dd ibs=3200 of=%s conv=ascii cbs=80 count=1", hfile);
        pipefp = epopen(cmdbuf, "w");

        /* Write ebcdic stream from buffer into pipe */
        efwrite(ebcbuf, EBCBYTES, 1, pipefp);


        /* Read binary header from tape to bh structure */
        if (buff) {
		if (-1 == read(tapefd, (char *) &tapebh, BNYBYTES)) {
			if (verbose)
				warn("tape read error on binary header");
			if (++errcount > errmax)
				err("exceeded maximum io errors");
		} else { /* Reset counter on successful tape IO */
			errcount = 0;
		}
	} else {
		if (iobyte) fread(&bytepr, 4, 1, tapefp);
                fread((char *) &tapebh, 1, BNYBYTES, tapefp);
		if (ferror(tapefp)) {
			if (verbose)
				warn("tape read error on binary header");
			if (++errcount > errmax)
				err("exceeded maximum io errors");
			clearerr(tapefp);
		} else { /* Reset counter on successful tape IO */
			errcount = 0;
		if (iobyte) fread(&bytepr, 4, 1, tapefp);
		}
	}

	/* Convert from bytes to ints/shorts */
	tapebhed_to_bhed(&tapebh, &bh);

	/* if little endian machine, swap bytes in binary header */
	if (endian==0) for (i = 0; i < BHED_NKEYS; ++i) swapbhval(&bh,i);
  
  	switch (bh.format) {
  	case 1:
  		bh.format = -1;   /* indicate that file is no longer SEG-Y */
  		warn("assuming IBM floating point input");
  		break;
  	case 2:
  		bh.format = -2;   /* indicate that file is no longer SEG-Y */
  		warn("assuming 4 byte integer input");
  		break;
  	case 3:
  		bh.format = -3;   /* indicate that file is no longer SEG-Y */
  		warn("assuming 2 byte integer input");
  		break;
  	default:
  		(over) ? warn("ignoring bh.format ... continue") :
  			err("format not SEGY standard (1, 2 or 3)");
  	}

        /* Compute length of trace (can't use sizeof here!) */
        if (!getparint("ns", &ns))  ns = bh.hns; /* let user override */
        if (!ns) err("samples/trace not set in binary header");

  	switch (bh.format) {
  	case -3:
  	        nsegy = ns*2 + SEGY_HDRBYTES;
  		break;
  	case -2:
  	case -1:
  	default:
  	        nsegy = ns*4 + SEGY_HDRBYTES;
  	}

        /* Write binary header from bhed structure to binary file */
        efwrite( (char *) &bh,1, BNYBYTES, binaryfp);

        /* Close binary and header files now to allow pipe into segywrite */
	efclose(binaryfp);
        if (verbose) warn("binary file closed successfully");
	efclose(headerfp);
	epclose(pipefp);
	if (verbose) warn("header file closed successfully");


        /* Read the traces */
        nsflag = false;
        itr = 0;
        while (itr < trmax) {
                int nread;

		if (buff) {
			if (-1 == 
			   (nread = read(tapefd, (char *) &tapetr, nsegy))){
				if (verbose)
				      warn("tape read error on trace %d", itr);
				if (++errcount > errmax)
				      err("exceeded maximum io errors");
			} else { /* Reset counter on successful tape IO */
				errcount = 0;
			}
		} else {
			if (iobyte) {
				fread(&bytepr, 4, 1, tapefp);
				if (!endian) swap_int_4(&bytepr); 
				if ( nsegy != bytepr ) {
					warn("number of record bytes mismatch!");
					nsegy = bytepr;  
				  	switch (bh.format) {
				  	case -3:
			  	        ns = (bytepr - SEGY_HDRBYTES)/2;
			  		break;
				  	case -2:
				  	case -1:
				  	default:
			  	        ns = (bytepr - SEGY_HDRBYTES)/4;
				  	}
				} 
			}
                 	nread = fread((char *) &tapetr, 1, nsegy, tapefp);
		 	if (ferror(tapefp)) {
				if (verbose)
				      warn("tape read error on trace %d", itr);
				if (++errcount > errmax)
				      err("exceeded maximum io errors");
				clearerr(tapefp);
			} else { /* Reset counter on successful tape IO */
				errcount = 0;
			if (iobyte) fread(&bytepr, 4, 1, tapefp);
			}
	    	}
                        
        if (!nread) break; /* middle exit loop instead of mile-long while */
	
		/* Convert from bytes to ints/shorts */
		tapesegy_to_segy(&tapetr, &tr, ns);
	
		/* If little endian machine, then swap bytes in trace header */
		if (endian==0)
			for (i = 0; i < SEGY_NKEYS; ++i) swaphval(&tr,i);
	
                /* Check tr.ns field */
                if (!nsflag && ns != tr.ns) {
                        warn("discrepant tr.ns = %d with tape/user ns = %d\n"
                                "\t... first noted on trace %d",
                                tr.ns, ns, itr + 1);
                        nsflag = true;
                }

                /* Convert and write desired traces */
                if (++itr >= trmin) {
                        /* Convert IBM floats to native floats */
                          if (conv) {
  				switch (bh.format) {
  				case -1:
                          /* Convert IBM floats to native floats */
  					ibm_to_float((int *) tr.data,
  						(int *) tr.data, ns, endian);
  					break;
  				case -2:
                          /* Convert 4 byte integers to native floats */
  					long_to_float((long *) tr.data,
  						(float *) tr.data, ns, endian);
  					break;
  				case -3:
                          /* Convert 2 byte integers to native floats */
  					short_to_float((short *) tr.data,
  						(float *) tr.data, ns, endian);
  					break;
  				}
  			}

			/* handle no ibm conversion for little endian case */
			if (conv==0 && endian==0)
                        	for (i = 0; i < ns ; ++i)
                                        swap_float_4(&tr.data[i]);

                        /* Write the trace to disk */
                        tr.ns = ns;
                        puttr(&tr);

                        /* Echo under verbose option */
                        if (verbose && itr % vblock == 0)
                                warn(" %d traces from tape", itr);
                }
        }



        /* Re-iterate error in case not seen during run */
        if (nsflag) warn("discrepancy found in header and trace ns values\n"
                "the value (%d) was used to extract traces", ns);


        /* Clean up (binary & header files already closed above) */
        (buff) ? eclose(tapefd):
                 efclose(tapefp);
        if (verbose) warn("tape closed successfully");


        return EXIT_SUCCESS;
}

static void ibm_to_float(int from[], int to[], int n, int endian)
/***********************************************************************
ibm_to_float - convert between 32 bit IBM and IEEE floating numbers
************************************************************************
Input::
from		input vector
to		output vector, can be same as input vector
endian		byte order =0 little endian (DEC, PC's)
			    =1 other systems 
************************************************************************* 
Notes:
Up to 3 bits lost on IEEE -> IBM

Assumes sizeof(int) == 4

IBM -> IEEE may overflow or underflow, taken care of by 
substituting large number or zero

Only integer shifting and masking are used.
************************************************************************* 
Credits: CWP: Brian Sumner,  c.1985
*************************************************************************/
{
    register int fconv, fmant, i, t;

    for (i=0;i<n;++i) {

	fconv = from[i];

	/* if little endian, i.e. endian=0 do this */
	if (endian==0) fconv = (fconv<<24) | ((fconv>>24)&0xff) |
		((fconv&0xff00)<<8) | ((fconv&0xff0000)>>8);

	if (fconv) {
            fmant = 0x00ffffff & fconv;
            t = (int) ((0x7f000000 & fconv) >> 22) - 130;
            while (!(fmant & 0x00800000)) { --t; fmant <<= 1; }
            if (t > 254) fconv = (0x80000000 & fconv) | 0x7f7fffff;
            else if (t <= 0) fconv = 0;
            else fconv = (0x80000000 & fconv) |(t << 23)|(0x007fffff & fmant);
        }
	to[i] = fconv;
    }
    return;
}


static void tapebhed_to_bhed(const tapebhed *tapebhptr, bhed *bhptr)
/****************************************************************************
tapebhed_to_bhed -- converts the seg-y standard 2 byte and 4 byte
	integer header fields to, respectively, the
	machine's short and int types. 
*****************************************************************************
Input:
tapbhed		pointer to array of 
*****************************************************************************
Notes:
The present implementation assumes that these types are actually the "right"
size (respectively 2 and 4 bytes), so this routine is only a placeholder for
the conversions that would be needed on a machine not using this convention.
*****************************************************************************
Author: CWP: Jack  K. Cohen, August 1994
****************************************************************************/

{
	register int i;
	Value val;
	
	/* convert binary header, field by field */
	for (i = 0; i < BHED_NKEYS; ++i) {
		gettapebhval(tapebhptr, i, &val);
		putbhval(bhptr, i, &val);
	}
}

static void tapesegy_to_segy(const tapesegy *tapetrptr, segy *trptr, int ns)
/****************************************************************************
tapesegy_to_segy -- converts the seg-y standard 2 byte and 4 byte
		    integer header fields to, respectively, the machine's
		    short and int types. 
*****************************************************************************
Input:
tapetrptr	pointer to trace in "tapesegy" (SEG-Y on tape) format
ns		number of samples per trace

Output:
trptr		pointer to trace in "segy" (SEG-Y as in  SU) format
*****************************************************************************
Notes:
Also copies float data byte by byte.  The present implementation assumes that
the integer types are actually the "right" size (respectively 2 and 4 bytes),
so this routine is only a placeholder for the conversions that would be needed
on a machine not using this convention.  The float data is preserved as
four byte fields and is later converted to internal floats by ibm_to_float
(which, in turn, makes additonal assumptions).
*****************************************************************************
Author: CWP:Jack K. Cohen,  August 1994
****************************************************************************/
{
	register int i;
	Value val;
	
	/* convert header trace header fields */
	for (i = 0; i < SEGY_NKEYS; ++i) {
		gettapehval(tapetrptr, i, &val);
		puthval(trptr, i, &val);
	}

	/* copy the optional portion */
	memcpy((char *)&(trptr->otrav)+2, tapetrptr->unass, 60);

	/* copy data portion */
	memcpy(trptr->data, tapetrptr->data, 4*SU_NFLTS);
}

static void long_to_float(long from[], float to[], int n, int endian)
/****************************************************************************
Author:	J.W. de Bruijn, May 1995
****************************************************************************/
{
  	register int i;
  
  	if (endian == 0) {
  		for (i = 0; i < n; ++i) {
  			swap_long_4(&from[i]);
  			to[i] = (float) from[i];
  		}
  	} else {
  		for (i = 0; i < n; ++i) {
  			to[i] = (float) from[i];
  		}
  	}
}
  
static void short_to_float(short from[], float to[], int n, int endian)
/****************************************************************************
Author:	J.W. de Bruijn, May 1995
****************************************************************************/
{
  	register int i;
  	float	*buf = alloc1float((size_t) n);
  
  	if (endian == 0) {
  		for (i = 0; i < n; ++i) {
  			swap_short_2(&from[i]);
  			buf[i] = (float) from[i];
  		}
  	} else {
  		for (i = 0; i < n; ++i) {
  			buf[i] = (float) from[i];
  		}
  	}
  	for (i = 0; i < n; ++i)
  		to[i] = buf[i];
  
  	free1float(buf);
}
